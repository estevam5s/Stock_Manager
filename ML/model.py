from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline


def get_model(model_type, random_state=42):
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        )
    elif model_type == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=random_state
        )
    else:
        raise ValueError(f"Modelo '{model_type}' não suportado. Use 'random_forest' ou 'gradient_boosting'")


def create_pipeline(preprocessor, model):
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
