import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore")

EVAL_THRESHOLD = 0.70


def _build_model(params: dict):
    model_type = str(params.get("model_type", "voting")).lower()

    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 800),
            max_depth=params.get("max_depth", None),
            min_samples_split=params.get("min_samples_split", 2),
            max_features=params.get("max_features", "sqrt"),
            random_state=42,
            n_jobs=-1,
        )
        return model, model_type

    if model_type == "et":
        model = ExtraTreesClassifier(
            n_estimators=params.get("n_estimators", 800),
            max_depth=params.get("max_depth", None),
            min_samples_split=params.get("min_samples_split", 2),
            max_features=params.get("max_features", "sqrt"),
            random_state=42,
            n_jobs=-1,
        )
        return model, model_type

    if model_type == "gb":
        model = GradientBoostingClassifier(
            n_estimators=params.get("n_estimators", 600),
            learning_rate=params.get("learning_rate", 0.04),
            max_depth=params.get("max_depth", 4),
            random_state=42,
        )
        return model, model_type

    if model_type == "hgb":
        model = HistGradientBoostingClassifier(
            max_iter=params.get("max_iter", 900),
            learning_rate=params.get("learning_rate", 0.02),
            max_depth=params.get("max_depth", 16),
            random_state=42,
        )
        return model, model_type

    if model_type == "lgbm":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=params.get("n_estimators", 900),
            learning_rate=params.get("learning_rate", 0.03),
            num_leaves=params.get("num_leaves", 63),
            max_depth=params.get("max_depth", -1),
            subsample=params.get("subsample", 0.9),
            colsample_bytree=params.get("colsample_bytree", 0.9),
            min_child_samples=params.get("min_child_samples", 20),
            random_state=42,
            verbosity=-1,
        )
        return model, model_type

    if model_type == "voting":
        from lightgbm import LGBMClassifier

        et = ExtraTreesClassifier(
            n_estimators=params.get("et_n_estimators", 1000),
            max_features=params.get("et_max_features", "sqrt"),
            random_state=42,
            n_jobs=-1,
        )
        rf = RandomForestClassifier(
            n_estimators=params.get("rf_n_estimators", 1200),
            max_features=params.get("rf_max_features", "sqrt"),
            random_state=42,
            n_jobs=-1,
        )
        lgbm = LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=params.get("lgbm_n_estimators", 900),
            learning_rate=params.get("lgbm_learning_rate", 0.03),
            num_leaves=params.get("lgbm_num_leaves", 63),
            max_depth=params.get("lgbm_max_depth", -1),
            subsample=params.get("lgbm_subsample", 0.9),
            colsample_bytree=params.get("lgbm_colsample_bytree", 0.9),
            random_state=42,
            verbosity=-1,
        )
        model = VotingClassifier(
            estimators=[("et", et), ("rf", rf), ("lgbm", lgbm)],
            voting="soft",
        )
        return model, model_type

    raise ValueError(f"Unsupported model_type: {model_type}")


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua model_type va cac sieu tham so.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia.

    Tra ve:
        accuracy (float): do chinh xac tren tap danh gia.
    """

    extra_data_path = "data/train_phase2.csv"
    use_phase2 = params.get("use_phase2", True)
    use_eval_for_training = params.get("use_eval_for_training", True)

    # TODO 1: Doc du lieu huan luyen va danh gia
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)
    if use_phase2 and os.path.basename(data_path) == "train_phase1.csv" and os.path.exists(extra_data_path):
        df_train_phase2 = pd.read_csv(extra_data_path)
        df_train = pd.concat([df_train, df_train_phase2], ignore_index=True)
    if use_eval_for_training:
        df_train = pd.concat([df_train, df_eval], ignore_index=True)

    # Dam bao tracking backend on dinh ngay ca khi shell chua export bien moi truong.
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)

    # TODO 2: Tach dac trung (X) va nhan (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    with mlflow.start_run():

        # TODO 3: Ghi nhan cac sieu tham so
        model, model_type = _build_model(params)
        mlflow.log_params(params)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("train_rows", int(len(df_train)))
        mlflow.log_param("use_eval_for_training", bool(use_eval_for_training))

        # TODO 4: Khoi tao va huan luyen model
        # Goi y: su dung random_state=42 de dam bao tinh tai tao
        model.fit(X_train, y_train)

        # TODO 5: Du doan tren tap danh gia va tinh chi so
        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")

        # TODO 6: Ghi nhan chi so vao MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        # TODO 7: In ket qua ra man hinh
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # TODO 8: Luu metrics ra file outputs/metrics.json
        # File nay duoc doc boi GitHub Actions o Buoc 2
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/metrics.json", "w") as f:
            json.dump({"accuracy": acc, "f1_score": f1}, f)

        # TODO 9: Luu mo hinh ra file models/model.pkl
        # File nay duoc upload len GCS o Buoc 2
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    # TODO 10: Tra ve acc
    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
