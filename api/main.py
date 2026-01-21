from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import fasttext
from catboost import CatBoostClassifier
from omegaconf import OmegaConf
import os


infer_cfg = OmegaConf.load("configs/inference.yaml")


app = FastAPI()


class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    prediction: str
    confidence: float
    all_scores: dict = None


# Глобальные переменные
model = None
fasttext_model = None


# Загрузка моделей
@app.on_event("startup")
def startup_event():
    global model, fasttext_model

    # CatBoost
    if os.path.exists(infer_cfg.model.catboost_path):
        model = CatBoostClassifier()
        model.load_model(infer_cfg.model.catboost_path)
    else:
        raise RuntimeError("CatBoost модель не найдена")

    # fastText
    if os.path.exists(infer_cfg.model.fasttext_path):
        fasttext_model = fasttext.load_model(infer_cfg.model.fasttext_path)
    else:
        raise RuntimeError("fastText модель не найдена")


def text_to_vector(text: str):
    text = text.replace("\n", " ")
    return fasttext_model.get_sentence_vector(text)


@app.get("/health")
def health_check():
    if model is not None and fasttext_model is not None:
        return {"status": "healthy", "message": "Service is running correctly"}
    else:
        return {"status": "unhealthy", "message": "Service not initialized"}, 500

@app.post("/predict", response_model=SentimentResponse)
def predict(request: SentimentRequest):
    try:
        X = np.expand_dims(text_to_vector(request.text), axis=0)
        # Преобразуем predict в обычную строку
        prediction = str(model.predict(X)[0])  

        # Вероятности
        try:
            probabilities = model.predict_proba(X)[0]
            confidence = float(np.max(probabilities))
            classes = ["Положительный", "Нейтральный", "Отрицательный"]
            all_scores = {cls: float(prob) for cls, prob in zip(classes, probabilities)}
        except AttributeError:
            confidence = 0.0
            all_scores = None

        return SentimentResponse(prediction=prediction, confidence=confidence, all_scores=all_scores)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

