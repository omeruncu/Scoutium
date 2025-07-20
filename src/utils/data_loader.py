import pandas as pd
import os


def load_csv(path, separator=";", verbose=True):
    """
    CSV dosyasını okur ve DataFrame döndürür.

    Parameters:
        path (str): Dosya yolu
        verbose (bool): Bilgilendirme çıktısı ister misin?

    Returns:
        pd.DataFrame: Okunan veri seti
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Belirtilen dosya yolu bulunamadı: {path}")

    df = pd.read_csv(path, sep=separator)

    if verbose:
        print(f"[INFO] Dosya yüklendi: {path}")
        print(f"[INFO] Veri seti boyutu: {df.shape}")
        print(f"[INFO] Kolonlar: {list(df.columns)}\n")

    return df


def load_scoutium_data(config):
    """
    YAML config üzerinden scoutium verilerini yükler.

    Parameters:
        config (dict): config.yaml'dan yüklenen parametreler

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: attributes_df, labels_df
    """
    attributes_path = config["paths"]["attributes_raw"]
    labels_path = config["paths"]["labels_raw"]

    separator = config.get("settings", {}).get("csv_separator", ";")

    attributes_df = load_csv(attributes_path, separator=separator)
    labels_df = load_csv(labels_path, separator=separator)

    return attributes_df, labels_df