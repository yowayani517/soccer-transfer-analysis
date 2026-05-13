import numpy as np
import joblib
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor

MODEL_PATH = "model.pkl"


def train(X, y, n_splits: int = 5):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=7,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.7,
        colsample_bylevel=0.7,
        reg_alpha=0.05,
        reg_lambda=1.5,
        gamma=0.1,
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_r2 = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2", n_jobs=-1)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "cv_r2_mean": float(cv_r2.mean()),
        "cv_r2_std": float(cv_r2.std()),
        "test_r2": float(r2_score(y_test, y_pred)),
        "test_mae_log": float(mean_absolute_error(y_test, y_pred)),
        "test_rmse_eur": float(np.sqrt(mean_squared_error(
            np.expm1(y_test), np.expm1(y_pred)
        ))),
        "test_mae_eur": float(mean_absolute_error(
            np.expm1(y_test), np.expm1(y_pred)
        )),
    }

    joblib.dump(model, MODEL_PATH)
    return model, metrics, (X_test, y_test, y_pred)


def predict_value(model, X_input: np.ndarray) -> float:
    log_pred = float(model.predict(X_input)[0])
    return float(np.expm1(log_pred))


def predict_all(model, X: np.ndarray) -> np.ndarray:
    return np.expm1(model.predict(X))


def feature_importance(model, feature_names: list) -> dict:
    importances = model.feature_importances_
    return dict(sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True,
    ))
