import dill
import os, sys
from sklearn.metrics import r2_score
from src.exception import CustomException


def save_object(file_path, obj):
    """
    Stores the Object (obj) such as Model, Pipelines, etc to the specified file_path.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as f:
            dill.dump(obj, f)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for m in list(models.keys()):
            model = models.get(m)

            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)

            test_score = r2_score(y_test, y_test_pred)

            models[m] = model  # Updating Models Dict with Fitted Model
            report[m] = test_score
        return report, models

    except Exception as e:
        raise CustomException(e, sys)
