from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

def stacking_classifier(models: dict, X, y, meta_model=None, cv=3, log_params=False):
    """
    En iyi modellerle stacking classifier kurar.
    """
    if log_params:
        print("\n🧮 Stacking Parametreleri:")
        for name, model in models.items():
            print(f"  {name}: {model.get_params()}")

    base_learners = [(name, model) for name, model in models.items()]
    final_estimator = meta_model if meta_model else LogisticRegression()

    stacking_clf = StackingClassifier(
        estimators=base_learners,
        final_estimator=final_estimator,
        cv=cv
    )
    stacking_clf.fit(X, y)
    return stacking_clf
