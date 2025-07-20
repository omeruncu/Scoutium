import pandas as pd


def merge_scoutium_data(attributes_df, labels_df):
    """
    Scoutium verilerini ortak anahtarlar üzerinden birleştirir.

    Returns:
        pd.DataFrame: Birleştirilmiş veri seti
    """
    merged_df = pd.merge(
        attributes_df,
        labels_df,
        how="inner",
        on=["task_response_id", "match_id", "evaluator_id", "player_id"]
    )

    print(f"[INFO] Merge sonrası boyut: {merged_df.shape}")
    return merged_df
