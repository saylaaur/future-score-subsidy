import streamlit as st
import pandas as pd
import datetime
from logic import FutureScoreLogic

# --- КОНФИГУРАЦИЯ ---
st.set_page_config(page_title="FutureScore РК", page_icon="🌾", layout="wide")

# Инициализация логики
@st.cache_resource
def get_logic():
    return FutureScoreLogic()

logic = get_logic()

# --- СТИЛИЗАЦИЯ ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #334155; }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR: ВЫБОР РОЛИ И РЫЧАГИ ---
st.sidebar.title("🎛 Панель управления")
role = st.sidebar.radio("Войти как:", ["👨‍🌾 Фермер", "🏛 Аудитор МСХ"])

# Рычаги Министра (Участник №2)
st.sidebar.divider()
st.sidebar.subheader("📍 Приоритеты государства")
priority_val = st.sidebar.slider("Приоритет мясного сектора (Коэфф.)", 1.0, 1.5, 1.2, 0.1)
weights = {'priority_multiplier': priority_val}

# --- ЛОГИКА ПРИЛОЖЕНИЯ ---

if role == "👨‍🌾 Фермер":
    st.title("👨‍🌾 Кабинет фермера: FutureScore")
    st.info("Подайте заявку и получите мгновенный прозрачный расчет баллов.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Форма подачи")
        with st.form("farmer_form"):
            region = st.selectbox("Регион", ["область Абай", "Акмолинская область", "Туркестанская область"])
            direction = st.selectbox("Направление", ["Субсидирование в скотоводстве", "Субсидирование мясного скотоводства"])
            district = st.text_input("Район хозяйства", "Жарминский район")
            
            cows = st.number_input("Поголовье (голов)", 1, 5000, 100)
            area = st.number_input("Площадь пастбищ (га)", 0, 50000, 800)
            mortality = st.slider("Уровень падежа (%)", 0.0, 20.0, 1.5)
            requested_sum = st.number_input("Запрашиваемая сумма (₸)", 100000, 100000000, 1500000)
            
            submit = st.form_submit_button("🚀 Рассчитать баллы")

    if submit:
        # Подготовка данных
        farmer_data = {
            "region": region,
            "Направление водства": direction,
            "Район хозяйства": district,
            "cows_count": cows,
            "pasture_area": area,
            "mortality_rate": mortality / 100,
            "Причитающая сумма": requested_sum
        }

        # Расчет
        res = logic.calculate_future_score(farmer_data, weights)
        
        with col2:
            st.subheader("📊 Результат анализа")
            
            # Спидометр / Метрики
            m1, m2 = st.columns(2)
            m1.metric("Итоговый балл", f"{res['final_score']} / 100")
            m2.metric("Статус", res['status'])

            if res['alerts']:
                for a in res['alerts']: st.error(a)
            else:
                st.success("✅ Заявка соответствует всем приказам МСХ РК.")

            # SHAP Визуализация (Участник №1)
            st.divider()
            st.subheader("🔍 Объяснение решения ИИ (SHAP)")
            shap_img = logic.get_shap_visual(res['df_input'])
            st.image(shap_img, use_container_width=True)

    # WHAT-IF СЕКЦИЯ (Участник №3)
    st.divider()
    st.subheader("🚀 Инвестиционное моделирование: Как повысить балл?")
    st.write("Используйте симулятор, чтобы понять, какие изменения в хозяйстве помогут получить субсидию.")
    
    wc1, wc2 = st.columns(2)
    with wc1:
        add_land = st.slider("Докупить/арендовать земли (га)", 0, 2000, 0)
        reduce_mort = st.slider("Снизить падеж до (%)", 0.0, 5.0, 1.5)
    
    with wc2:
        # Сценарий изменений
        sim_changes = {
            'pasture_area': area + add_land,
            'mortality_rate': reduce_mort / 100
        }
        # Исходные данные для симуляции те же, что из формы
        sim_data = {
            "region": region, "Направление водства": direction, "Район хозяйства": district,
            "cows_count": cows, "pasture_area": area, "mortality_rate": mortality / 100,
            "Причитающая сумма": requested_sum
        }
        
        sim_res = logic.get_what_if_analysis(sim_data, sim_changes, weights)
        
        st.write(f"### Новый прогноз: **{sim_res['new_score']}**")
        st.write(f"Изменение: :green[{sim_res['delta_str']}]" if sim_res['delta'] > 0 else f"Изменение: {sim_res['delta_str']}")
        st.info(f"Новый статус: {sim_res['new_status']}")

elif role == "🏛 Аудитор МСХ":
    st.title("🏛 Цифровой аудитор: Министерство сельского хозяйства РК")
    
    # Blind Review Toggle (Участник №2)
    blind_mode = st.toggle("🔒 Включить Режим Blind Review (Антикоррупция)", value=True)
    
    # Имитация базы данных заявок
    mock_data = [
        {"id": "7712", "farm": "КХ 'АгроЛидер'", "region": "область Абай", "score": 88.5, "sum": 4500000},
        {"id": "8291", "farm": "ТОО 'Мясной Мир'", "region": "Туркестанская область", "score": 42.1, "sum": 12000000},
        {"id": "9012", "farm": "ИП 'Степное'", "region": "Акмолинская область", "score": 75.0, "sum": 2300000}
    ]
    
    df_apps = pd.DataFrame(mock_data)
    
    if blind_mode:
        df_apps['farm'] = "ID_" + df_apps['id']
        st.warning("🕵️ Режим Blind Review активен: Названия хозяйств скрыты для исключения предвзятости.")

    st.subheader("📥 Поступившие заявки")
    st.dataframe(df_apps, use_container_width=True)
    
    st.divider()
    st.subheader("📈 Финансовый дашборд")
    k1, k2, k3 = st.columns(3)
    k1.metric("Общий бюджет", "1.2 млрд ₸")
    k2.metric("Одобрено заявок", "485 млн ₸")
    k3.metric("Остаток", "715 млн ₸", delta="-12%")
