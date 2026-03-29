import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px

import os
st.sidebar.write("### 🔍 Проверка файлов:")
st.sidebar.write("Текущая папка:", os.getcwd())
if os.path.exists('models'):
    st.sidebar.write("✅ Папка 'models' найдена!")
    st.sidebar.write("Файлы в ней:", os.listdir('models'))
else:
    st.sidebar.error("❌ Папка 'models' НЕ найдена!")

# Настройка страницы
st.set_page_config(page_title="FutureScore PRO", layout="wide", page_icon="🐄")

# --- 1. ЗАГРУЗКА AI-АКТИВОВ ---
@st.cache_resource
def load_ai_assets():
    model_path = 'models/futurescore_model_pro.pkl'
    artifacts_path = 'models/data_pipeline_artifacts_pro (1).pkl' 
    
    if os.path.exists(model_path) and os.path.exists(artifacts_path):
        try:
            # Используем pickle, так как сохраняли через него
            with open(model_path, 'rb') as f:
                model_arts = pickle.load(f)
                model = model_arts['xgboost_model']
                explainer = model_arts['explainer']
                
            with open(artifacts_path, 'rb') as f:
                artifacts = pickle.load(f)
                
            return model, explainer, artifacts
        except Exception as e:
            st.error(f"❌ Ошибка при загрузке .pkl файлов: {e}")
            return None, None, None
    return None, None, None

model, explainer, artifacts = load_ai_assets()

# --- 2. САЙДБАР ---
with st.sidebar:
    st.title("🐄 FutureScore PRO")
    if model is not None:
        st.success("🚀 ИИ-модель (XGBoost) активна!")
        st.info(f"Используется признаков: {len(artifacts['features_list'])}")
    else:
        st.error("⚠️ Файлы модели не найдены в папке /models")
    
    st.divider()
    st.markdown("**MVP для Минсельхоза РК**\n\n*Merit-based скоринг на основе открытых данных*")

# --- 3. ГЛАВНЫЙ ИНТЕРФЕЙС ---
st.title("Система скоринга субсидий (Merit-based)")

tab1, tab2, tab3 = st.tabs(["📊 Рейтинг заявок", "🔍 Детали (SHAP)", "📋 AI-Рекомендации"])

df_final = None

with tab1:
    st.subheader("📊 Текущий рейтинг FutureScore")
    
    # 1. Загрузка данных
    try:
        if os.path.exists('data/final_dataset_pro.csv'):
            df_final = pd.read_csv('data/final_dataset_pro.csv')
            # Если в данных нет колонки ID, создадим искусственную для демо
            if 'id' not in df_final.columns:
                df_final.insert(0, 'id', range(1000, 1000 + len(df_final)))
        else:
            st.error("📁 Файл 'data/final_dataset_pro.csv' не найден!")
    except Exception as e:
        st.error(f"❌ Ошибка при чтении CSV: {e}")

    # 2. Расчет скоринга
    if df_final is not None and model is not None:
        try:
            # БЕРЕМ ТОЛЬКО ТЕ ПРИЗНАКИ, НА КОТОРЫХ УЧИЛАСЬ МОДЕЛЬ! (Исправленный баг)
            features_to_use = artifacts['features_list']
            
            if all(col in df_final.columns for col in features_to_use):
                X_input = df_final[features_to_use]
                
                # Прогноз вероятности (от 0 до 1) переводим в 100-балльную шкалу
                probs = model.predict_proba(X_input)[:, 1]
                df_final['FutureScore'] = np.round(probs * 100, 1)

                # Сортируем от лучших к худшим
                df_final = df_final.sort_values('FutureScore', ascending=False)
                st.success(f"✅ Скоринг успешно рассчитан для {len(df_final)} хозяйств.")
                
                # Вывод красивой таблицы (ID, Score + важные фичи)
                display_cols = ['id', 'FutureScore', 'climate_risk', 'amount_to_norm_ratio', 'is_breeding', 'district_historical_score']
                st.dataframe(df_final[display_cols].head(50), use_container_width=True)
                
                # График распределения
                fig = px.histogram(df_final, x='FutureScore', nbins=50, title="Распределение баллов надежности по региону", color_discrete_sequence=['#2ecc71'])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("❌ В загруженном файле не хватает нужных колонок для скоринга!")
        except Exception as e:
            st.error(f"⚠️ Ошибка предикта модели: {e}")

    # 3. Интерактивный поиск (Фишка для демо)
    st.divider()
    st.subheader("🧠 GovTech Advisor (Проверка конкретной заявки)")
    
    if df_final is not None:
        search_id = st.text_input("Введите ID хозяйства (например, 1030):", "")
        
        if search_id:
            result = df_final[df_final['id'].astype(str) == search_id]
            
            if not result.empty:
                score = result['FutureScore'].values[0]
                climate = result['climate_risk'].values[0]
                breeding = result['is_breeding'].values[0]
                
                st.write(f"### Результат скоринга для ID {search_id}")
                
                # Система "Светофор"
                if score >= 80:
                    st.success(f"🟢 СТАТУС: Высокий потенциал. Балл: {score}/100. Рекомендуется одобрение.")
                elif score >= 60:
                    st.warning(f"🟡 СТАТУС: Средний риск. Балл: {score}/100. Требует внимания комиссии.")
                else:
                    st.error(f"🔴 СТАТУС: Высокий риск неэффективности. Балл: {score}/100.")

                # Умный советник (Твоя бизнес-логика!)
                st.markdown("#### 💡 Анализ ИИ-советника:")
                if climate >= 0.7:
                    st.info(f"🌍 **Сложный климат:** Регион находится в зоне высокого климатического риска (индекс {climate}). Низкие показатели могут быть вызваны засухой. Рекомендуется индивидуальное рассмотрение.")
                if breeding == 0 and score < 80:
                    st.info("🐄 **Точка роста:** Переход на племенное поголовье или селекцию значительно повысит рейтинг надежности (FutureScore) данного хозяйства.")
                    
                st.dataframe(result[display_cols])
            else:
                st.warning("Хозяйство с таким ID не найдено. Проверьте номер.")

with tab2:
    st.subheader("Прозрачность ИИ (Explainable AI)")
    st.info("В полной версии здесь отображаются водопадные графики библиотеки SHAP. Они показывают проверяющему чиновнику математический вклад каждого фактора (положительный или отрицательный) в итоговый балл FutureScore. Это исключает эффект 'черного ящика'.")
    # Подсказка для Участника С: Сюда можно просто вставить st.image('shap_plot.png')

with tab3:
    st.subheader("Системные рекомендации")
    st.markdown("""
    **Как принимаются решения на основе FutureScore:**
    1. **Merit-based приоритет:** Финансирование в первую очередь направляется в зеленую зону (>80 баллов), независимо от времени подачи заявки.
    2. **Учет форс-мажоров:** Система автоматически 'прощает' часть неэффективности хозяйствам в регионах с суровым климатом (коэффициент climate_risk > 0.75).
    3. **Борьба с приписками:** Аномальное соотношение запрошенной суммы к нормативу (amount_to_norm_ratio) жестко пенализируется алгоритмом XGBoost.
    """)
