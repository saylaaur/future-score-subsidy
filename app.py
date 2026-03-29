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
    1. Прозрачность: Каждый балл обоснован историческими данными и внешними факторами (Казгидромет).
    2. Справедливость: Мы учитываем региональные особенности (Climate Risk), чтобы фермеры в засушливых зонах не дискриминировались.
    3. Эффективность: Приоритет отдается хозяйствам с высокой долей инноваций (селекция, племенное дело).
    """)
