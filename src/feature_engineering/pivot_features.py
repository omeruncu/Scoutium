import pandas as pd


def create_player_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Oyuncu bazında özellik matrisini oluşturur.
    Her satırda bir oyuncu olacak şekilde attribute_id skorları sütunlara taşınır.

    Parametre:
        df (pd.DataFrame): Scoutium birleştirilmiş ve filtrelenmiş veri seti

    Dönüş:
        pd.DataFrame: Pivotlanmış, modellemeye hazır oyuncu veri seti
    """

    print("[INFO] Pivot işlemi başlatıldı...")

    # Pivot işlemi: her satırda bir oyuncu olacak şekilde yatay genişlet
    pivot_df = df.pivot_table(
        index=["player_id", "position_id", "potential_label"],
        columns="attribute_id",
        values="attribute_value",
        aggfunc="mean"
    ).reset_index()

    # Sütun adlarını string'e çevir
    pivot_df.columns = [str(col) for col in pivot_df.columns]

    print(f"[INFO] Pivot tablo oluşturuldu → Yeni boyut: {pivot_df.shape}")
    return pivot_df
