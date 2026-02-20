import os


class ModelConfig:
    MODEL_TYPES = ["random_forest", "gradient_boosting"]
    
    DEFAULT_PARAMS = {
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42,
            "n_jobs": -1
        },
        "gradient_boosting": {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "random_state": 42
        }
    }
    
    ADVANCED_PARAMS = {
        "random_forest": [
            ("n_estimators", "int", "Número de árvores", (10, 500)),
            ("max_depth", "int", "Profundidade máxima", (1, 50)),
            ("min_samples_split", "int", "Mín. amostras para dividir", (2, 20)),
            ("min_samples_leaf", "int", "Mín. amostras por folha", (1, 10)),
            ("max_features", "str", "Features por split", ["sqrt", "log2", None]),
        ],
        "gradient_boosting": [
            ("n_estimators", "int", "Número de estágios", (10, 500)),
            ("max_depth", "int", "Profundidade máxima", (1, 20)),
            ("learning_rate", "float", "Taxa de aprendizado", (0.01, 1.0)),
            ("subsample", "float", "Subamostragem", (0.5, 1.0)),
        ]
    }
    
    TASK_TYPES = ["classification", "regression"]
    
    EVALUATION_METRICS = {
        "classification": ["accuracy", "precision", "recall", "f1_score", "confusion_matrix"],
        "regression": ["mae", "mse", "rmse", "r2_score"]
    }
    
    TEST_SIZE_OPTIONS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    
    @classmethod
    def get_model_params(cls, model_type, custom_params=None):
        params = cls.DEFAULT_PARAMS.get(model_type, {}).copy()
        if custom_params:
            params.update(custom_params)
        return params
    
    @classmethod
    def validate_params(cls, model_type, params):
        default = cls.DEFAULT_PARAMS.get(model_type, {})
        validated = {}
        
        for key, value in params.items():
            if key in default:
                if isinstance(default[key], int):
                    validated[key] = int(value) if value else default[key]
                elif isinstance(default[key], float):
                    validated[key] = float(value) if value else default[key]
                else:
                    validated[key] = value
            else:
                validated[key] = value
        
        return validated
    
    @classmethod
    def get_metrics_for_task(cls, task):
        return cls.EVALUATION_METRICS.get(task, [])
