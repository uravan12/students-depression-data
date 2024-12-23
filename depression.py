# импорт библиотек
import pandas as pd
import numpy as np
import streamlit as st
import joblib
from joblib import load
from sklearn.preprocessing import OneHotEncoder

# загрузка модели
model = joblib.load("depression_model.pkl")
data = pd.read_csv('Student Depression Dataset.csv')

for col in data.columns:
    if data[col].isnull().any():
        if data[col].dtype in ['int64', 'float64']:
          median_value = data[col].median() # Замена на медиану для численных признаков
          data[col] = data[col].fillna(median_value)
        else:
            mode_value = data[col].mode()[0]  # Замена на моду для категориальных признаков
            data[col] = data[col].fillna(mode_value)

data = data.drop(['id','City', 'Profession', 'Work Pressure', 'Job Satisfaction', 'Degree'], axis=1)
# Категориальные признаки
categorical_features = ['Gender', 'Sleep Duration', 'Dietary Habits', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']
numerical_features = ['Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction', 'Work/Study Hours', 'Financial Stress']
# Создаем энкодер и обучаем его на всех данных, чтобы не было проблем с новыми значениями
encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
X = data.drop('Depression', axis=1)
encoder.fit(X[categorical_features])
feature_names = encoder.get_feature_names_out(categorical_features)


all_columns = [
    'Gender',
    'Age',
    'Academic Pressure',
    'CGPA',
    'Study Satisfaction',
    'Sleep Duration',
    'Dietary Habits',
    'Have you ever had suicidal thoughts ?',
    'Work/Study Hours',
    'Financial Stress',
    'Family History of Mental Illness'
    ]

# Указание контента сайта
st.title("Прогнозирование возможности появления депрессии у студента")

gender = st.selectbox('Пол', options=data['Gender'].unique())
age = st.number_input(
    "Возраст",
    min_value=18,
    max_value=59,
    value=18,
    step=1
)
academic_pressure = st.number_input(
    "Учебная нагрузка",
    min_value=0,
    max_value=5,
    value=0,
    step=1
)
grade = st.number_input(
    "Средний балл",
    min_value=0.00,
    max_value=10.00,
    value=0.00,
    step=0.01
)
study_satisfaction = st.number_input(
    "Удовлетворенность от учебы",
    min_value=0,
    max_value=5,
    value=0,
    step=1
)
sleep_duration = st.selectbox('Продолжительность сна', options=data['Sleep Duration'].unique())

dietary_habits = st.selectbox('Пищевые привычки', options=data['Dietary Habits'].unique())

suiciadal_thoughts = st.selectbox('Возникали ли у вас суицидальные мысли?', options=data['Have you ever had suicidal thoughts ?'].unique())

work_hours = st.number_input(
    "Рабочие/учебные часы",
    min_value=0,
    max_value=12,
    value=0,
    step=1
)
financial_stress = st.number_input(
    "Финансовые трудности",
    min_value=1,
    max_value=5,
    value=1,
    step=1
)

family = st.selectbox('Были ли у кого то в семье психические заболевания?', options=data['Family History of Mental Illness'].unique())

# Преобразование данных и прогноз
input_data = pd.DataFrame({
    'Gender': [gender],
    'Age': [age],
    'Academic Pressure': [academic_pressure],
    'CGPA': [grade],
    'Study Satisfaction': [study_satisfaction],
    'Sleep Duration': [sleep_duration],
    'Dietary Habits': [dietary_habits],
    'Have you ever had suicidal thoughts ?': [suiciadal_thoughts],
    'Work/Study Hours': [work_hours],
    'Financial Stress': [financial_stress],
    'Family History of Mental Illness': [family]
})
# Отделяем числовые и категориальные признаки
input_data_num = input_data[numerical_features]
input_data_cat = input_data[categorical_features]


# Кодируем категориальные признаки
input_data_encoded = encoder.transform(input_data_cat)
input_data_encoded = pd.DataFrame(input_data_encoded, columns = feature_names)


# Объединяем числовые и закодированные категориальные признаки
input_data_processed = pd.concat([input_data_num, input_data_encoded], axis = 1)

if st.button('Предсказать возможность появления депрессии'):
    prediction = model.predict(input_data_processed)[0]
    st.success(f'У вас может появится депрессия с вероятностью: {prediction * 100:.2f}%')