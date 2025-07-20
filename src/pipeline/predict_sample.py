import pandas as pd
import pickle
import yaml
import os
from src.utils.column_selector import get_numeric_columns
import time


def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_model(path):  # artık tam path alıyor
    with open(path, "rb") as f:
        return pickle.load(f)


def load_clean_data():
    config = load_config()
    return pd.read_csv(config["paths"]["cleaned_data"])

def predict_random_sample():
    df = load_clean_data()

    # Dinamik örnek seçimi
    sample = df.sample(1)
    numeric_cols = get_numeric_columns(df)
    numeric_cols = [col for col in numeric_cols if col != "potential_label_encoded"]
    features = sample[numeric_cols]
    info_cols = [col for col in sample.columns if col not in numeric_cols]

    print("\n🎯 Oyuncu Bilgisi:")
    print(sample[info_cols].T)

    # Tüm models klasöründeki .pkl dosyalarını tarar
    config = load_config()
    model_dir = config["paths"]["model_output"]
    model_files = [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f)) and f.endswith(".pkl")]

    print("\n🧪 Model Tahminleri:")
    for file in model_files:
        name = os.path.splitext(file)[0]  # Dosya adını model adı olarak kullan
        model_path = os.path.join(model_dir, file)
        model = load_model(model_path)

        try:
            pred = model.predict(features)[0]
            prob = model.predict_proba(features)[0][1] if hasattr(model, "predict_proba") else None
            print(f"  {name} → Tahmin: {pred} | Olasılık: {round(prob, 4) if prob is not None else 'Yok'}")
        except Exception as e:
            print(f"  [WARN] {name} tahmin üretirken hata verdi → {e}")

