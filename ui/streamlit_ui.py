import streamlit as st
import requests


# Конфиг
API_URL = "http://localhost:8000"  # URL вашего FastAPI сервиса

st.set_page_config(page_title="Sentiment Analysis", layout="wide")

st.title("NLP Sentiment Analysis API")
st.write("Интерфейс для тестирования модели анализа сентимента русского текста.")


# Проверка состояния сервиса
if st.button("Проверить состояние сервиса"):
    try:
        r = requests.get(f"{API_URL}/health")
        if r.status_code == 200:
            st.success(f"Сервис здоров: {r.json()['message']}")
        else:
            st.error(f"Сервис недоступен: {r.status_code}")
    except Exception as e:
        st.error(f"Ошибка при подключении: {e}")

st.markdown("---")


# Ввод текста
st.subheader("Введите текст для анализа сентимента")

user_input = st.text_area("Текст отзыва", "")

if st.button("Получить предсказание"):
    if not user_input.strip():
        st.warning("Введите текст!")
    else:
        try:
            payload = {"text": user_input}
            response = requests.post(f"{API_URL}/predict", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                st.success(f"**Предсказание:** {data['prediction']}")
                st.info(f"**Уверенность:** {data['confidence']:.2f}")
                st.write("**Все вероятности:**", data.get("all_scores"))
            else:
                st.error(f"Ошибка сервиса: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Ошибка запроса: {e}")
