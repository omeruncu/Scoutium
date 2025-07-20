import pandas as pd

def get_numeric_columns(df: pd.DataFrame) -> list:
    """
    Sayısal tipteki sütunları tespit eder ve 'id' geçenleri hariç tutar.

    Parameters:
        df (pd.DataFrame): Veri seti

    Returns:
        list: Sayısal sütun adları (id içermeyenler)
    """
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    num_cols = [col for col in numeric_cols if "id" not in col.lower()]

    print(f"[INFO] Sayısal sütunlar (id içermeyen): {num_cols}")
    return num_cols
