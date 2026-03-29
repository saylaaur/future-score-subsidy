import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px

# Настройка страницы
st.set_page_config(page_title="FutureScore PRO", layout="wide", page_icon="🚜")

# --- ФУНКЦИЯ ЗАГРУЗКИ (С отладкой) ---
@st.cache_resource
def load_ai_assets():
    # Стандартные пути (БЕЗ ПРОБЕЛОВ И СКОБОК)
    base_path = "models"
    model_name = "futurescore_model_pro.pkl"
    arts_name = "data_pipeline_artifacts_pro.pkl"
    
    model_path = os.path.join(base_path, model_name)
    arts_path = os.path.join(base_path, arts_name)

    # Проверка существования файлов
    if not os.path.exists(model_path) or not os.path.exists(arts_path):
        return None, None, None, f"Файлы не найдены. Убедитесь, что в /{base_path} лежат {model_name} и {arts_name}"

    try:
        with open(model_path, 'rb') as f:
            m_data = pickle.load(f)
            # Извлекаем из словаря, который мы создавали в ноутбуке
            model = m_data['xgboost_model']
            explainer = m_data['explainer']
            
        with open(arts_path, 'rb') as f:
            artifacts = pickle.load(f)
            
        return model, explainer, artifacts, "OK"
    except Exception as e:
        return None, None, None, f"Ошибка загрузки: {str(e)}"

# --- Инициализация ---
model, explainer, artifacts, status_msg = load_ai_assets()

# --- САЙДБАР ---
with st.sidebar:
    st.title("🚜 FutureScore PRO")
    if model:
        st.success("✅ ИИ-модель загружена")
    else:
        st.error(f"❌ Ошибка: {status_msg}")
        # Помощь в отладке прямо в интерфейсе
        st.write("Содержимое корня:", os.listdir("."))
        if os.path.exists("models"):
            st.write("Файлы в /models:", os.listdir("models"))
    
    st.divider()
    st.info("MVP для оценки эффективности субсидий Минсельхоза РК")

# --- ОСНОВНОЙ КОНТЕНТ ---
st.title("Система скоринга субсидий (Merit-based)")

if not model:
    st.stop() # Останавливаем приложение, если модель не загрузилась

tab1, tab2, tab3 = st.tabs(["📊 Рейтинг", "🔍 Анализ заявки", "📋 Рекомендации"])

# Загрузка данных
@st.cache_data
def load_data():
    path = "data/final_dataset_pro.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        if 'id' not in df.columns:
            df.insert(0, 'id', range(1000, 1000 + len(df)))
        return df
    return None

df = load_data()

with tab1:
    if df is not None:
        st.subheader("📊 Текущий рейтинг надежности хозяйств")
        
        # Расчет скоринга (используем только нужные фичи)
        features = artifacts['features_list']
        X = df[features]
        
        # Получаем вероятности и переводим в баллы 0-100
        df['FutureScore'] = np.round(model.predict_proba(X)[:, 1] * 100, 1)
        df = df.sort_values("FutureScore", ascending=False)
        
        # Вывод таблицы
        view_cols = ['id', 'FutureScore', 'climate_risk', 'amount_to_norm_ratio', 'is_breeding']
        st.dataframe(df[view_cols].head(100), use_container_width=True)
        
        # Визуализация
        fig = px.histogram(df, x="FutureScore", title="Распределение баллов по всем заявкам", color_discrete_sequence=['#2ecc71'])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Файл данных в data/ не найден!")

with tab2:
    st.subheader("🔍 Детальный разбор хозяйства")
    search_id = st.text_input("Введите ID хозяйства для проверки (например, 1050):")
    
    if search_id and df is not None:
        res = df[df['id'].astype(str) == search_id]
        if not res.empty:
            row = res.iloc[0]
            score = row['FutureScore']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Балл FutureScore", f"{score}/100")
                if score >= 80: st.success("🟢 РЕКОМЕНДАЦИЯ: ОДОБРИТЬ")
                elif score >= 60: st.warning("🟡 РЕКОМЕНДАЦИЯ: ПРОВЕРИТЬ КОМИССИЕЙ")
                else: st.error("🔴 РЕКОМЕНДАЦИЯ: ВЫСОКИЙ РИСК")
            
            with col2:
                st.write("**Ключевые показатели:**")
                st.write(f"🌍 Климатический риск: {row['climate_risk']}")
                st.write(f"🧬 Племенное дело: {'Да' if row['is_breeding'] == 1 else 'Нет'}")
            
            st.divider()
            st.markdown("#### 💡 Советы GovTech Advisor:")
            if row['climate_risk'] > 0.7:
                st.info("⚠️ Хозяйство в зоне засухи. Низкий балл может быть обусловлен внешними факторами.")
            if row['is_breeding'] == 0:
                st.info("🐄 Рекомендуется переход на племенное животноводство для повышения эффективности.")
        else:
            st.warning("ID не найден.")

with tab3:
    st.subheader("Методология FutureScore")
    st.write("""
    1. **Прозрачность:** Каждый балл обоснован историческими данными и внешними факторами (Казгидромет).
    2. **Справедливость:** Мы учитываем региональные особенности (Climate Risk), чтобы фермеры в засушливых зонах не дискриминировались.
    3. **Эффективность:** Приоритет отдается хозяйствам с высокой долей инноваций (селекция, племенное дело).
    """)
