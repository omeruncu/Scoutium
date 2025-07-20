import warnings

warnings.filterwarnings("ignore")

from src.pipeline.predict_sample import predict_random_sample


def main():
    print("\n🚀 Scoutium Tahmin Pipeline Başlatıldı")

    # Rastgele veri üzerinden modellerle tahmin yap
    predict_random_sample()

    print("\n✅ Tahmin süreci tamamlandı. Raporu kontrol edebilirsiniz.")


if __name__ == "__main__":
    main()
