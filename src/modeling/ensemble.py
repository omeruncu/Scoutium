from sklearn.ensemble import VotingClassifier

def soft_voting_classifier(models: dict, X, y, voting_type="soft", log_params=False):
    """
    En iyi modellerle soft voting ensemble kurar.
    """
    if log_params:
        print("\n🧮 Ensemble Parametreleri:")
        for name, model in models.items():
            print(f"  {name}: {model.get_params()}")

    estimators = [(name, model) for name, model in models.items()]
    voting_clf = VotingClassifier(estimators=estimators, voting=voting_type)
    voting_clf.fit(X, y)
    return voting_clf

