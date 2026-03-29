import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import os

# --- 1. Настройка страницы ---
st.set_page_config(page_title="FutureScore PRO | AI GovTech", layout="wide", page_icon="🚜")
st.title("🚜 FutureScore PRO: Merit-based скоринг субсидий")

# --- 2. Умная загрузка моделей (Автопоиск файлов) ---
@st.cache_resource
def load_ai_brains():
    # Пути к файлам (проверка корня и папки models)
    arts_name = 'data_pipeline_artifacts_pro.pkl'
    model_name = 'futurescore_model_pro.pkl'
    
    arts_path = arts_name if os.path.exists(arts_name) else os.path.join('models', arts_name)
    model_path = model_name if os.path.exists(model_name) else os.path.join('models', model_name)
    
    with open(arts_path, 'rb') as f:
        data_arts = pickle.load(f)
    with open(model_path, 'rb') as f:
        model_arts = pickle.load(f)
    return data_arts, model_arts

try:
    data_arts, model_arts = load_ai_brains()
    model = model_arts['xgboost_model']
    explainer = model_arts['explainer']
    features_list = data_arts['features_list']
    st.sidebar.success("✅ ИИ-модель активна")
except Exception as e:
    st.error(f"❌ Ошибка загрузки модели. Проверьте наличие .pkl файлов. \n {e}")
    st.stop()

# --- 3. Автозагрузка данных (Zero-Click) ---
@st.cache_data
def load_data():
    data_name = 'final_dataset_pro.csv'
    data_path = data_name if os.path.exists(data_name) else os.path.join('data', data_name)
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        if 'id' not in df.columns:
            df.insert(0, 'id', range(1000, 1000 + len(df)))
        return df
    return None

df_raw = load_data()

if df_raw is None:
    st.warning("⚠️ Файл данных не найден в репозитории.")
    st.stop()

# --- 4. Расчет рейтинга ---
X = df_raw[features_list].copy()
probs = model.predict_proba(X)[:, 1]
df_raw['FutureScore'] = np.round(probs * 100, 1)

st.header("🏆 Шорт-лист кандидатов (Merit-based)")
top_farmers = df_raw.sort_values(by='FutureScore', ascending=False).reset_index(drop=True)

# Колонки для отображения
display_cols = ['id', 'FutureScore', 'climate_risk', 'amount_to_norm_ratio', 'is_breeding']
st.dataframe(top_farmers[display_cols].head(50), use_container_width=True)

# --- 5. GovTech Advisor (Интерфейс анализа) ---
st.divider()
st.header("🧠 GovTech Advisor (Интерпретация ИИ)")

col1, col2 = st.columns([1, 1.5])

with col1:
    selected_idx = st.number_input("Выберите индекс строки (из таблицы выше):", min_value=0, max_value=len(top_farmers)-1, value=0)
    
    farmer = top_farmers.iloc[selected_idx]
    score = farmer['FutureScore']
    climate = farmer.get('climate_risk', 0)
    breeding = farmer.get('is_breeding', 1)
    
    st.subheader(f"Анализ заявки № {farmer['id']}")
    
    if score >= 80:
        st.success(f"🟢 **СТАТУС: Одобрить** ({score}/100)")
    elif score >= 60:
        st.warning(f"🟡 **СТАТУС: Доп. проверка** ({score}/100)")
    else:
        st.error(f"🔴 **СТАТУС: Высокий риск** ({score}/100)")
        
    st.markdown("#### 💡 Рекомендация системы:")
    if climate >= 0.6:
        st.info(f"🌍 **Климат:** Высокий риск в регионе ({climate}). Балл скорректирован с учетом погодных условий.")
    if breeding == 0:
        st.info("🐄 **Совет:** Рекомендуется внедрение племенного дела для повышения балла в будущем.")

with col2:
    st.write("**Математическое обоснование (SHAP):**")
    
    # ФИКС ВИЗУАЛИЗАЦИИ: Настройка под темную тему
    plt.style.use('dark_background') 
    fig, ax = plt.subplots(figsize=(10, 5))
    
    row_for_shap = farmer[features_list]
    shap_values = explainer.shap_values(pd.DataFrame([row_for_shap]))
    
    try:
        # Рисуем график
        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value, 
            shap_values[0], 
            feature_names=features_list, 
            show=False
        )
        
        # Финальные штрихи по цветам текста
        plt.gcf().set_facecolor('#0e1117') # Цвет фона Streamlit
        plt.gca().set_facecolor('#0e1117')
        st.pyplot(fig)
    except Exception as e:
        st.error("Ошибка отрисовки графика.")

# --- Сайдбар (Анонимный) ---
st.sidebar.title("FutureScore System")
st.sidebar.info("Система оценки эффективности сельскохозяйственных субсидий. Версия: MVP 1.0 (Decentrathon 5.0)")
st.sidebar.divider()
st.sidebar.write("Используется алгоритм: XGBoost")
st.sidebar.write("Слой интерпретации: SHAP")
