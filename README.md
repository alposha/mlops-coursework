# NLP Sentiment Analysis (Russian Text)

Проект классифицирует русскоязычные отзывы на три класса: Положительный, Нейтральный, Отрицательный.  
Реализован полный ML-пайплайн с обучением моделей, инференс-сервисом на FastAPI и Streamlit UI.

---

## Pipeline

        +-------------------+
        |   Raw Text Data   |
        +-------------------+
                 |
                 v
        +-------------------+
        |  Data Cleaning    |
        +-------------------+
                 |
                 v
        +-------------------+
        |  FastText Model   |
        +-------------------+
                 |
                 v
        +-------------------+
        | Text -> Vectors   |
        +-------------------+
                 |
                 v
        +-------------------+
        |  ML Model         |
        +-------------------+
                 |
                 v
        +-------------------+
        |  Predictions      |
        +-------------------+
                 |
                 v
        +-------------------+
        | FastAPI Inference |
        +-------------------+

## Быстрый старт

### 1.Клонирование и настройка окружения

```bash
# Клонировать репозиторий
git clone <repository-url>
cd mlops-coursework
```

```bash
# Создать виртуальное окружение
python -m venv venv
```

```bash
# Активировать (Linux/macOS)
source venv/bin/activate
```

```bash
# Активировать (Windows)
venv\Scripts\activate
```

```bash
# Установить зависимости
pip install -r requirements.txt
```

### 2. Эксперименты

```bash
# Запуск экспериментов
python src/experiments.py

```

### MLflow UI

```bash
# Отслеживания экспериментов
mlflow ui --port 5000
```

Открыть: http://127.0.0.1:5000/

### Локальный запуск FastAPI сервиса

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Эндпоинты:

- /health — проверка состояния сервиса
- /predict — предсказание сентимента

### Streamlit UI

```bash
# Для удобного тестирования модели через веб-интерфейс:
streamlit run ui/streamlit_ui.py
```

### Контейнеризация

```bash
docker build -t nlp-sentiment-inference .
docker run -p 8000:8000 nlp-sentiment-inference
```
