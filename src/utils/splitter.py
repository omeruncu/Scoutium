from sklearn.model_selection import train_test_split


def split_data(X, y, config):
    """
    Config tabanlı veri bölme işlemi.

    Parameters:
        X (pd.DataFrame): Özellikler
        y (pd.Series): Etiketler
        config (dict): YAML yapılandırması

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    split_cfg = config.get("train_test_split", {})
    test_size = split_cfg.get("test_size", 0.2)
    random_state = split_cfg.get("random_state", 42)
    stratify = y if split_cfg.get("stratify", True) else None

    print(
        f"[INFO] Train/Test Split → test_size={test_size}, random_state={random_state}, stratify={stratify is not None}")

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
