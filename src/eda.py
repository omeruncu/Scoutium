import pandas as pd

from src.modeling.evulate_model import compile_model_metrics
from src.preprocessing.scaling import scale_features
from src.utils.eda_helpers import report_target_distribution, get_top_models
from src.utils.splitter import split_data
from src.utils.yaml_loader import load_config
from src.utils.data_loader import load_scoutium_data
from src.preprocessing.merge import merge_scoutium_data
from src.feature_engineering.pivot_features import create_player_feature_matrix
from src.preprocessing.filters import apply_label_filter, apply_position_filter
from src.preprocessing.encode_labels import encode_column
from src.utils.column_selector import get_numeric_columns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# Config dosyasını yükle
config = load_config("config/config.yaml")

# Scoutium CSV'lerini yükle
attributes_df, labels_df = load_scoutium_data(config)

# attributes_df.head()
# labels_df.head()

# Merge işlemi
merged_df = merge_scoutium_data(attributes_df, labels_df)

# merged_df.head()

# filtered_df = apply_label_filter(apply_position_filter(merged_df, config), config)
no_goalkeepers_df = apply_position_filter(merged_df, config)
filtered_df = apply_label_filter(no_goalkeepers_df, config)

# filtered_df.head()
# filtered_df.info()

pivot_df = create_player_feature_matrix(filtered_df)

# pivot_df.head()
# pivot_df.info()

# pivot_df["potential_label"].unique()

labeled_df = encode_column(pivot_df, config, column="potential_label")

# labeled_df.head()
# labeled_df.info()

num_cols = get_numeric_columns(labeled_df)
num_cols = [col for col in num_cols if col != "potential_label_encoded"]

X = labeled_df[num_cols]
y = labeled_df["potential_label_encoded"]

# Etiket dağılımını gözlemle
report_target_distribution(y)

# train/Test Split
X_train, X_test, y_train, y_test = split_data(X, y, config)

# Feature scaling
X_train_scaled, X_test_scaled = scale_features(X_train, X_test, num_cols)

from src.modeling.base_models import base_models

base_models_dict, base_scores_dict = base_models(X_train_scaled, y_train)

from src.modeling.hyperparameter_search import hyperparameter_optimization

best_models, best_params_dict, best_scores = hyperparameter_optimization(X_train_scaled, y_train)

for name in best_params_dict:
    print(f"\n{name} Best Score: {best_scores[name]}")
    print(f"Best Params: {best_params_dict[name]}")

from src.modeling.ensemble import soft_voting_classifier
from src.modeling.stacking import stacking_classifier

top_models, top_params = get_top_models(best_scores, best_models, best_params_dict)

voting_clf = soft_voting_classifier(top_models, X_train_scaled, y_train, log_params=True)
stacking_clf = stacking_classifier(top_models, X_train_scaled, y_train, log_params=True)



ensemble_models_dict = {
    "Voting": voting_clf,
    "Stacking": stacking_clf
}

metrics_df = compile_model_metrics(base_models_dict, best_models, ensemble_models_dict, X_test_scaled, y_test)
print("\nTam Model Karşılaştırma Tablosu:")
print(metrics_df)

from src.utils.reports_writer import save_report

# Tam tablo
save_report(metrics_df, report_name="model_comparison")


from src.utils.model_saver import save_models

models_to_save = {
    "Base_RF": base_models_dict["RF"],
    "Best_RF": best_models["RF"],
    "Voting": voting_clf
}

custom_names = {
    "Base_RF": "base_rf_model.pkl",
    "Best_RF": "best_rf_model.pkl",
    "Voting": "voting_classifier.pkl"
}

save_models(models=models_to_save, names=custom_names)

