def apply_position_filter(df, config):
    """
    Config tabanlı pozisyon filtreleme işlemi yapar (örneğin kaleci).

    Parameters:
        df (pd.DataFrame): Veri seti
        config (dict): config.yaml içeriği

    Returns:
        pd.DataFrame: Filtrelenmiş veri seti
    """
    position_cfg = config["preprocessing"].get("filters", {}).get("drop_positions", {})
    if not position_cfg.get("enabled", False):
        print("[INFO] Pozisyon filtreleme devre dışı.")
        return df

    ids_to_drop = position_cfg.get("position_ids", [])
    initial_shape = df.shape
    df_filtered = df[~df["position_id"].isin(ids_to_drop)].copy()
    removed = initial_shape[0] - df_filtered.shape[0]

    print(f"[INFO] Pozisyon filtreleme uygulandı → {removed} satır silindi")
    return df_filtered


def apply_label_filter(df, config):
    """
    Config tabanlı etiket filtreleme işlemi yapar (örneğin below_average).

    Parameters:
        df (pd.DataFrame): Veri seti
        config (dict): config.yaml içeriği

    Returns:
        pd.DataFrame: Filtrelenmiş veri seti
    """
    label_cfg = config["preprocessing"].get("filters", {}).get("drop_labels", {})
    if not label_cfg.get("enabled", False):
        print("[INFO] Etiket filtreleme devre dışı.")
        return df

    labels_to_drop = label_cfg.get("labels_to_drop", [])
    initial_shape = df.shape
    df_filtered = df[~df["potential_label"].isin(labels_to_drop)].copy()
    removed = initial_shape[0] - df_filtered.shape[0]

    print(f"[INFO] Etiket filtreleme uygulandı → {removed} satır silindi")
    return df_filtered
