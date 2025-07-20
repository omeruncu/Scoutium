import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from joblib import parallel_backend


def base_models(X, y, scoring=["accuracy", "precision", "f1", "roc_auc"], n_jobs=1):
    """
    Birçok temel modeli karşılaştırır, hem modelleri hem skorlarını döner.

    Returns:
        models_dict: {'Model_Adı': model nesnesi}
        results_dict: {'Model_Adı': {'Skor_Metrik': Değer}}
    """
    print("\n📊 Base Model Karşılaştırması")

    classifiers = [
        ("LR", LogisticRegression()),
        ("KNN", KNeighborsClassifier()),
        ("SVC", SVC(probability=True)),
        ("CART", DecisionTreeClassifier()),
        ("RF", RandomForestClassifier(n_jobs=1)),
        ("Adaboost", AdaBoostClassifier()),
        ("GBM", GradientBoostingClassifier()),
        ("XGBoost", XGBClassifier(n_jobs=1, use_label_encoder=False, eval_metric='logloss')),
        ("LightGBM", LGBMClassifier(verbose=-1, n_jobs=1)),
        ("CatBoost", CatBoostClassifier(verbose=False, thread_count=1))
    ]

    models_dict = {}
    results_dict = {}

    for name, model in classifiers:
        try:
            with parallel_backend("threading"):
                cv_results = cross_validate(model, X, y, cv=3, scoring=scoring, n_jobs=n_jobs)

                metrics = {
                    score: round(cv_results[f"test_{score}"].mean(), 4)
                    for score in scoring
                }

                model.fit(X, y)

                models_dict[name] = model
                results_dict[name] = metrics

                print(f"\n{name} Skorları:")
                for score in scoring:
                    print(f"  {score.upper()} → {metrics[score]}")

        except Exception as e:
            print(f"[WARN] {name} başarısız oldu → {str(e)}")

    return models_dict, results_dict

