from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline


def get_classification_model(model_type, random_state=42):
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=random_state,
            n_jobs=-1
        )
    elif model_type == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            random_state=random_state
        )
    else:
        raise ValueError(f"Modelo '{model_type}' não suportado")


def get_regression_model(model_type, random_state=42):
    if model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=random_state,
            n_jobs=-1
        )
    elif model_type == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            random_state=random_state
        )
    else:
        raise ValueError(f"Modelo '{model_type}' não suportado")


def create_pipeline(preprocessor, model):
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
