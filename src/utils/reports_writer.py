import os
import yaml
import pandas as pd

def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_report(df: pd.DataFrame, report_name: str = "model_comparison"):
    """
    Model metrik tablosunu hem .csv hem .md olarak config'e bağlı path'e kaydeder.
    """
    config = load_config()
    out_path = config["paths"]["report_output"]
    os.makedirs(out_path, exist_ok=True)

    csv_path = os.path.join(out_path, f"{report_name}.csv")
    md_path  = os.path.join(out_path, f"{report_name}.md")

    df.to_csv(csv_path)
    df.to_markdown(buf=md_path)

    print(f"[✔] CSV Raporu → {csv_path}")
    print(f"[✔] Markdown Raporu → {md_path}")