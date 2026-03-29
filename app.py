import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# 1. Настройка страницы
st.set_page_config(page_title="FutureScore PRO | AI GovTech", layout="wide", page_icon="🚜")
st.title("🚜 FutureScore PRO: Merit-based скоринг")

# 2. Загрузка Артефактов (БЕЗ ВСЯКИХ УСЛОЖНЕНИЙ)
@st.cache_resource
def load_ai_brains():
    # Просто читаем файлы из папки models
    with open('models/data_pipeline_artifacts_pro.pkl', 'rb') as f:
        data_arts = pickle.load(f)
    with open('models/futurescore_model_pro.pkl', 'rb') as f:
        model_arts = pickle.load(f)
    return data_arts, model_arts

try:
    data_arts, model_arts = load_ai_brains()
    model = model_arts['xgboost_model']
    explainer = model_arts['explainer']
    st.sidebar.success("✅ Модель загружена!")
except Exception as e:
    st.sidebar.error(f"❌ Ошибка загрузки модели. Проверьте файлы в папке models/ \n Ошибка: {e}")
    st.stop()

st.sidebar.header("📁 Входные данные")
uploaded_file = st.sidebar.file_uploader("Загрузите CSV с заявками", type="csv")

if uploaded_file:
    # Читаем загруженный файл
    df = pd.read_csv(uploaded_file)
    
    # Отделяем таргет, если он есть
    if 'target' in df.columns:
        X = df.drop('target', axis=1)
    else:
        X = df.copy()
        
    # Считаем баллы
    probs = model.predict_proba(X)[:, 1]
    df['FutureScore'] = np.round(probs * 100, 1)
    
    # Вывод топа
    st.header("🏆 Шорт-лист кандидатов (Merit-based)")
    top_farmers = df.sort_values(by='FutureScore', ascending=False)
    st.dataframe(top_farmers[['FutureScore', 'climate_risk', 'amount_to_norm_ratio', 'is_breeding']].head(50))
    
    # Интерактивный AI-Ассистент
    st.header("🧠 GovTech Advisor (ИИ-Ассистент Комиссии)")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_idx = st.number_input("Введите индекс заявки в таблице (от 0):", min_value=0, max_value=len(df)-1, value=0)
        score = df['FutureScore'].iloc[selected_idx]
        climate = df['climate_risk'].iloc[selected_idx]
        
        # Светофор
        if score >= 80:
            st.success(f"Балл: {score}/100. СТАТУС: Одобрить.")
        elif score >= 60:
            st.warning(f"Балл: {score}/100. СТАТУС: Требует внимания.")
        else:
            st.error(f"Балл: {score}/100. СТАТУС: Высокий риск.")
            
        st.info(f"🌍 Климатический риск региона: {climate}")
        
    with col2:
        st.write("**Факторы принятия решения (SHAP):**")
        shap_values = explainer.shap_values(X)
        fig, ax = plt.subplots(figsize=(8, 3))
        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value, 
            shap_values[selected_idx], 
            feature_names=X.columns, 
            show=False
        )
        st.pyplot(fig)
else:
    st.info("👈 Загрузите файл `final_dataset_pro.csv` в боковом меню слева, чтобы система начала работу.")
