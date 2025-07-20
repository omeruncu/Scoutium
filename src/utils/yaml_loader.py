import yaml
import os


def load_config(path):
    """
    config.yaml dosyasını yükler ve sözlük olarak döner.

    Parameters:
        path (str): config.yaml dosya yolu

    Returns:
        dict: YAML içeriği
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] YAML dosyası bulunamadı: {path}")

    with open(path, "r", encoding="utf-8") as file:
        try:
            config = yaml.safe_load(file)
            print(f"[INFO] Config yüklendi: {path}")
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"[ERROR] YAML parsing hatası: {e}")