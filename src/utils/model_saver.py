import pickle
import os
import yaml

def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_models(models: dict, names: dict = None):
    """
    Verilen model sözlüğünü `.pkl` olarak config'deki dizine kaydeder.

    Parameters:
        models (dict): {'ModelKey': model_obj}
        names (dict): {'ModelKey': 'dosya_adi'} opsiyonel
    """
    config = load_config()
    save_path = config["paths"]["model_output"]
    os.makedirs(save_path, exist_ok=True)

    for key, model in models.items():
        fname = names[key] if names and key in names else f"{key}.pkl"
        filepath = os.path.join(save_path, fname)

        with open(filepath, "wb") as f:
            pickle.dump(model, f)
        print(f"[✔] Model kaydedildi → {filepath}")
