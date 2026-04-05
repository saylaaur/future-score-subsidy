import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import io

class FutureScoreLogic:
    def __init__(self, model_path='xgb_model.pkl', encoders_path='encoders.pkl'):
        """Инициализация ядра: загрузка ML-модели, энкодеров и SHAP."""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(encoders_path, 'rb') as f:
                self.encoders = pickle.load(f)
                
            # Инициализация SHAP Explainer (только для древовидных моделей)
            self.explainer = shap.TreeExplainer(self.model)
        except FileNotFoundError:
            print("⚠️ Ошибка: Файлы xgb_model.pkl и encoders.pkl не найдены в директории!")
            self.model = None
            self.encoders = {}
            self.explainer = None
            
        # Нормативы пастбищ (гектар на 1 голову КРС)
        self.pasture_norms = {
            'Акмолинская область': 8.5,
            'Туркестанская область': 11.0,
            'область Абай': 10.0,
            'по умолчанию': 9.0
        }

    def calculate_future_score(self, data, weights=None):
        """
        ЗАДАЧА 1: Универсальный гибридный скоринг.
        Возвращает итоговый балл, статус, алерты и подготовленный DataFrame для SHAP.
        """
        if weights is None:
            weights = {'priority_multiplier': 1.0}

        # 1. Безопасная подготовка данных (предотвращает падение при новых значениях)
        df_input = self._prepare_dataframe(data)
        
        # 2. ML-вероятность (от 0 до 100)
        if self.model:
            ml_prob = self.model.predict_proba(df_input)[0][1] * 100
        else:
            ml_prob = 50.0  # Фолбэк, если модель не загрузилась

        final_score = ml_prob
        penalties = []

        # 3. LEGAL LAYER: Приказ №2 (Штраф за падеж > 3%)
        if data.get('mortality_rate', 0) > 0.03:
            final_score -= 30
            penalties.append(f"⚠️ Приказ №2: Превышен лимит падежа (>3%). Штраф -30б.")

        # 4. LEGAL LAYER: Приказ №3 (Штраф за нехватку земли)
        region = data.get('region', 'по умолчанию')
        norm = self.pasture_norms.get(region, self.pasture_norms['по умолчанию'])
        required_pasture = data.get('cows_count', 0) * norm
        
        if data.get('pasture_area', 0) < required_pasture:
            final_score -= 20
            penalties.append(f"⚠️ Приказ №3: Дефицит пастбищ (нужно минимум {int(required_pasture)} га). Штраф -20б.")

        # 5. ПРИОРИТЕТЫ (Рычаги Министра)
        priority_coeff = weights.get('priority_multiplier', 1.0)
        if "мясного" in str(data.get('Направление водства', '')).lower():
            final_score *= priority_coeff
            if priority_coeff > 1.0:
                penalties.append(f"⭐ Приоритет сектора: Мясное направление (Умножение x{priority_coeff})")

        # Форматируем результат (от 0 до 100)
        final_score = round(max(0, min(100, final_score)), 1)
        status = "✅ Одобрено" if final_score >= 60 else "❌ Отклонено"

        return {
            "ml_score": round(ml_prob, 1),
            "final_score": final_score,
            "alerts": penalties,
            "status": status,
            "df_input": df_input # Отдаем дальше для SHAP
        }

    def get_what_if_analysis(self, base_data, changes, weights=None):
        """
        ЗАДАЧА 2: Симулятор "А что если?". 
        changes: dict, например {'pasture_area': 1500, 'mortality_rate': 0.01}
        """
        # Считаем текущий балл
        base_result = self.calculate_future_score(base_data, weights)
        base_score = base_result['final_score']
        
        # Симулируем новые данные
        simulated_data = base_data.copy()
        for feature, new_val in changes.items():
            simulated_data[feature] = new_val
            
        # Считаем новый балл
        new_result = self.calculate_future_score(simulated_data, weights)
        new_score = new_result['final_score']
        
        delta = round(new_score - base_score, 1)
        sign = "+" if delta > 0 else ""
        
        return {
            "old_score": base_score,
            "new_score": new_score,
            "delta": delta,
            "delta_str": f"{sign}{delta}",
            "new_alerts": new_result['alerts'],
            "new_status": new_result['status']
        }

    def get_shap_visual(self, df_input):
        """
        ЗАДАЧА 3: Визуализация SHAP (Объяснение ИИ).
        Возвращает буфер с картинкой для Streamlit: st.image(image_buffer)
        """
        if self.explainer is None:
            return None
            
        shap_values = self.explainer(df_input)
        
        # Настройка графика под UI (без лишних белых полей)
        plt.figure(figsize=(9, 4))
        shap.plots.waterfall(shap_values[0], show=False)
        plt.tight_layout()
        
        # Конвертация в картинку для веба
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight', dpi=150)
        plt.close()
        buf.seek(0)
        return buf

    def _prepare_dataframe(self, data):
        """
        Внутренняя функция: Защита от ошибок "Unseen Label" и фиксация порядка фичей.
        """
        input_row = {}
        
        # 1. Численные признаки
        for feat in ['cows_count', 'pasture_area', 'mortality_rate', 'Причитающая сумма']:
            input_row[feat] = data.get(feat, 0)

        # 2. Категориальные признаки (с фолбэком на 0)
        cat_map = {
            'region': 'region_encoded',
            'Направление водства': 'Направление водства_encoded',
            'Район хозяйства': 'Район хозяйства_encoded'
        }

        for orig_col, encoded_col in cat_map.items():
            val = str(data.get(orig_col, 'Неизвестно'))
            if self.encoders and orig_col in self.encoders:
                le = self.encoders[orig_col]
                # Если категория знакома ИИ - кодируем, иначе ставим первую (обычно 0)
                input_row[encoded_col] = le.transform([val])[0] if val in le.classes_ else 0
            else:
                input_row[encoded_col] = 0

        # 3. Фиксация порядка колонок (КРИТИЧНО ДЛЯ XGBOOST!)
        features_order = [
            'cows_count', 'pasture_area', 'mortality_rate', 'Причитающая сумма', 
            'region_encoded', 'Направление водства_encoded', 'Район хозяйства_encoded'
        ]
        
        return pd.DataFrame([input_row])[features_order]