import pickle
import os
import yaml

def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def extract_top_models(metrics_df, all_model_objs, top_n=3, weighted=True):
    """
    Metriklere göre en iyi modelleri seçer. weighted=True ise ağırlıklı skorlama yapar.
    """
    weights = {"ROC_AUC": 0.4, "F1": 0.3, "Accuracy": 0.2, "Precision": 0.1} if weighted else {k: 1 for k in ["ROC_AUC", "F1", "Accuracy", "Precision"]}

    metrics_df_filtered = metrics_df.dropna()
    metrics_df_filtered["weighted_score"] = sum([metrics_df_filtered[k] * w for k, w in weights.items()])
    top_model_names = metrics_df_filtered.sort_values(by="weighted_score", ascending=False).index[:top_n]

    selected_models = {name: all_model_objs[name.split("_")[-1]] for name in top_model_names if name.split("_")[-1] in all_model_objs}
    return selected_models

def save_models_from_metrics(metrics_df, all_model_objs, report_name="model_comparison", top_n=3, weighted=True):
    config = load_config()
    save_path = config["paths"]["model_output"]
    os.makedirs(save_path, exist_ok=True)

    selected_models = extract_top_models(metrics_df, all_model_objs, top_n=top_n, weighted=weighted)

    for name, model in selected_models.items():
        filename = f"{name.lower()}_model.pkl"
        filepath = os.path.join(save_path, filename)

        with open(filepath, "wb") as f:
            pickle.dump(model, f)
        print(f"[✔] {name} modeli kaydedildi → {filepath}")

