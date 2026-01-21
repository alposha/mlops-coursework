import os
import pandas as pd
import numpy as np
import fasttext
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier
import mlflow
import mlflow.sklearn


# Загружаем конфиг
cfg = OmegaConf.load("configs/train.yaml")


# Загрузка данных
data = pd.read_csv(cfg.data.path)


# Обучение fastText
ft_params = cfg.model.fasttext
ft_model = fasttext.train_supervised(
    input=cfg.data.fasttext_file,
    epoch=ft_params.epoch,
    lr=ft_params.lr,
    wordNgrams=ft_params.word_ngrams,
    bucket=ft_params.bucket
)

os.makedirs("models", exist_ok=True)
ft_model.save_model(cfg.train.save_fasttext_model_path)

def get_fasttext_vectors(texts, model):
    return np.array([model.get_sentence_vector(str(t).replace("\n", " ")) for t in texts])


# Подготовка данных
X = get_fasttext_vectors(data["Text"], ft_model)
y = data["Rate"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=cfg.train.test_size, random_state=cfg.train.random_state
)


# MLflow helper
def run_experiment(model, X_train, X_test, y_train, y_test, experiment_name, params):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        for k, v in params.items():
            mlflow.log_param(k, v)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1": f1_score(y_test, y_pred, average="weighted")
        }
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_dict(classification_report(y_test, y_pred, output_dict=True), "classification_report.json")
        mlflow.log_dict(confusion_matrix(y_test, y_pred).tolist(), "confusion_matrix.json")
        print(f"\n{experiment_name}")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")


# RandomForest эксперименты
for exp_name, params in cfg.model.random_forest.items():
    rf_model = RandomForestClassifier(**params)
    run_experiment(
        rf_model, X_train, X_test, y_train, y_test,
        f"RandomForest-{exp_name}",
        params
    )

# Logistic Regression
lr_model = LogisticRegression(
    max_iter=cfg.model.logistic_regression.max_iter,
    random_state=cfg.model.logistic_regression.random_state
)
run_experiment(
    lr_model, X_train, X_test, y_train, y_test,
    "LogisticRegression",
    OmegaConf.to_container(cfg.model.logistic_regression)
)

# CatBoost
cb_params = cfg.model.catboost
cb_model = CatBoostClassifier(
    iterations=cb_params.iterations,
    learning_rate=cb_params.learning_rate,
    depth=cb_params.depth,
    verbose=cb_params.verbose,
    random_state=cb_params.random_state
)
run_experiment(
    cb_model, X_train, X_test, y_train, y_test,
    "CatBoost",
    OmegaConf.to_container(cb_params)
)

print("\n✅ Все эксперименты завершены")
