from sklearn.preprocessing import StandardScaler

def scale_features(X_train, X_test, num_cols):
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols]  = scaler.transform(X_test[num_cols])

    print(f"[INFO] {len(num_cols)} sütun scale edildi (StandardScaler)")
    return X_train_scaled, X_test_scaled
