import pandas as pd

from sklearn.preprocessing import LabelEncoder


def encode_column(df: pd.DataFrame, config: dict, column: str) -> pd.DataFrame:
    """
    config.yaml'dan belirtilen encoding stratejisini uygular.

    Parameters:
        df (pd.DataFrame): İşlenecek veri
        config (dict): YAML'dan yüklenen ayarlar
        column (str): Encoding yapılacak sütun

    Returns:
        pd.DataFrame: Encoding uygulanmış veri
    """
    encoding_cfg = config["preprocessing"]["encoding"].get(column)
    if not encoding_cfg:
        raise ValueError(f"[ERROR] config.yaml içinde '{column}' için encoding ayarı bulunamadı.")

    method = encoding_cfg.get("method")
    output_col = encoding_cfg.get("output_column", column + "_encoded")

    if method == "label_encoder":
        encoder = LabelEncoder()
        df[output_col] = encoder.fit_transform(df[column])
        print(f"[INFO] {column} → {output_col} (LabelEncoder uygulandı)")
        return df

    else:
        raise NotImplementedError(f"[ERROR] Şu anda '{method}' yöntemi desteklenmiyor.")
