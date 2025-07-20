def compile_model_metrics(base_models_dict, best_models_dict, ensemble_models_dict, X, y):
    """
    Base, optimize ve ensemble modellerin metriklerini birleştirerek tek tablo üretir.
    """
    import pandas as pd
    from sklearn.model_selection import cross_validate

    results = []

    # Base modeller
    for name, model in base_models_dict.items():
        cv = cross_validate(model, X, y, cv=3, scoring=["accuracy", "precision", "f1", "roc_auc"])
        results.append({
            "Model": f"Base_{name}",
            "Accuracy": round(cv["test_accuracy"].mean(), 4),
            "Precision": round(cv["test_precision"].mean(), 4),
            "F1": round(cv["test_f1"].mean(), 4),
            "ROC_AUC": round(cv["test_roc_auc"].mean(), 4)
        })

    # Optimize edilmiş modeller
    for name, model in best_models_dict.items():
        cv = cross_validate(model, X, y, cv=3, scoring=["accuracy", "precision", "f1", "roc_auc"])
        results.append({
            "Model": f"Best_{name}",
            "Accuracy": round(cv["test_accuracy"].mean(), 4),
            "Precision": round(cv["test_precision"].mean(), 4),
            "F1": round(cv["test_f1"].mean(), 4),
            "ROC_AUC": round(cv["test_roc_auc"].mean(), 4)
        })

    # Ensemble modeller (Voting, Stacking)
    for name, model in ensemble_models_dict.items():
        cv = cross_validate(model, X, y, cv=3, scoring=["accuracy", "precision", "f1", "roc_auc"])
        results.append({
            "Model": name,
            "Accuracy": round(cv["test_accuracy"].mean(), 4),
            "Precision": round(cv["test_precision"].mean(), 4),
            "F1": round(cv["test_f1"].mean(), 4),
            "ROC_AUC": round(cv["test_roc_auc"].mean(), 4)
        })

    return pd.DataFrame(results).set_index("Model")
