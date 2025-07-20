import optuna
optuna.logging.set_verbosity(optuna.logging.CRITICAL)
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def optimize_model(model_name, X, y, scoring="roc_auc", n_trials=30):
    """
    Belirtilen model için Optuna ile hiperparametre optimizasyonu uygular.
    """

    def objective(trial):
        if model_name == "KNN":
            model = KNeighborsClassifier(
                n_neighbors=trial.suggest_int("n_neighbors", 2, 50)
            )

        elif model_name == "CART":
            model = DecisionTreeClassifier(
                max_depth=trial.suggest_int("max_depth", 2, 20),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 30)
            )

        elif model_name == "RF":
            model = RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 500),
                max_depth=trial.suggest_categorical("max_depth", [8, 15, None]),
                max_features=trial.suggest_categorical("max_features", [5, 7, "sqrt", "log2", None]),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
                n_jobs=1
            )

        elif model_name == "XGBoost":
            model = XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                n_estimators=trial.suggest_int("n_estimators", 100, 300),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1),
                n_jobs=1
            )

        elif model_name == "LightGBM":
            model = LGBMClassifier(
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1),
                n_estimators=trial.suggest_int("n_estimators", 100, 300),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1),
                verbose=-1,
                n_jobs=1
            )

        elif model_name == "CatBoost":
            model = CatBoostClassifier(
                depth=trial.suggest_int("depth", 4, 8),
                iterations=trial.suggest_int("iterations", 100, 300),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1),
                verbose=False,
                thread_count=1
            )

        else:
            raise ValueError(f"Unknown model name: {model_name}")

        return cross_val_score(model, X, y, cv=3, scoring=scoring, n_jobs=1).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print(f"\n{model_name} BEST {scoring.upper()}: {study.best_value:.4f}")
    print(f"Best Params → {study.best_params}")

    # En iyi model kurulumu + öğrenme
    final_model = {
        "KNN": KNeighborsClassifier,
        "CART": DecisionTreeClassifier,
        "RF": lambda **p: RandomForestClassifier(n_jobs=1, **p),
        "XGBoost": lambda **p: XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_jobs=1, **p),
        "LightGBM": lambda **p: LGBMClassifier(verbose=-1, n_jobs=1, **p),
        "CatBoost": lambda **p: CatBoostClassifier(verbose=False, thread_count=1, **p)
    }[model_name](**study.best_params)

    final_model.fit(X, y)
    return final_model, study.best_value, study.best_params


def hyperparameter_optimization(X, y, scoring="roc_auc", n_trials=30):
    model_names = ["KNN", "CART", "RF", "XGBoost", "LightGBM", "CatBoost"]
    best_models = {}
    best_params_dict = {}
    best_scores = {}

    for name in model_names:
        final_model, best_score, best_params = optimize_model(name, X, y, scoring, n_trials)
        best_models[name] = final_model
        best_params_dict[name] = best_params
        best_scores[name] = round(best_score, 4)

    return best_models, best_params_dict, best_scores

