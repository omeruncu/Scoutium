import pandas as pd

def data_overview(df: pd.DataFrame) -> None:
    """
    Veri setinin genel yapısını özetler.

    Parametre:
    df (pd.DataFrame): Analiz edilecek veri seti

    Çıktı:
    - Gözlem ve özellik sayısı
    - Sütun isimleri
    - Veri tipleri ve bellek kullanımı
    - Temel istatistikler (özelleştirilmiş yüzdeliklerle)
    - Eksik değer özet tablosu
    """
    print("\n🧾 Veri Seti Genel Bilgisi")
    print(f"Gözlem sayısı : {df.shape[0]}")
    print(f"Özellik sayısı: {df.shape[1]}")
    print(f"Sütunlar      : {list(df.columns)}")

    print("\n📊 Veri Tipleri ve Bellek Kullanımı")
    df.info()

    print("\n📈 Temel İstatistikler (özelleştirilmiş yüzdeliklerle)")
    desc = df.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T
    print(desc)

    print("\n🚨 Eksik (NaN) Değerler")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print(missing)
    else:
        print("Eksik değer bulunmuyor.")


def report_target_distribution(y: pd.Series):
    distribution = y.value_counts(normalize=True).sort_index()
    absolute = y.value_counts().sort_index()
    print("\n🎯 Hedef Etiket Dağılımı (potential_label_encoded):")
    for label in distribution.index:
        print(f"  Sınıf {label}: {absolute[label]} gözlem ({distribution[label]*100:.2f}%)")


def get_top_models(best_scores: dict, best_models: dict, best_params_dict: dict, top_n=3):
    """
    En iyi 'top_n' modeli skorlar üzerinden seçer.
    """
    sorted_models = sorted(best_scores.items(), key=lambda item: item[1], reverse=True)[:top_n]

    selected_models = {}
    selected_params = {}

    for name, _ in sorted_models:
        selected_models[name] = best_models[name]
        selected_params[name] = best_params_dict[name]

    return selected_models, selected_params