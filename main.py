import os
import pandas as pd
import numpy as np
import base64
import logging
import random
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    silhouette_score,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.decomposition import PCA

import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

from dash import Dash, dcc, html, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Business Analytics Dashboard"

# Функція для конвертації фігури matplotlib у bytes
def fig_to_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return buf.read()

# Завантаження та перевірка файлів CSV
required_files = ['trans.csv', 'loan.csv', 'account.csv', 'client.csv', 'disp.csv']
missing_files = [file for file in required_files if not os.path.exists(file)]

if missing_files:
    logging.error(f"Наступні файли не знайдено: {missing_files}")
    raise FileNotFoundError(f"Наступні файли не знайдено: {missing_files}")
else:
    logging.info("Усі необхідні файли знайдено.")

# Завантаження CSV з використанням правильного роздільника ';'
try:
    trans = pd.read_csv('trans.csv', sep=';', low_memory=False)
    loan = pd.read_csv('loan.csv', sep=';', low_memory=False)
    account = pd.read_csv('account.csv', sep=';', low_memory=False)
    client = pd.read_csv('client.csv', sep=';', low_memory=False)
    disp = pd.read_csv('disp.csv', sep=';', low_memory=False)  # <-- Ключовий файл для зв'язку
    logging.info("Усі файли CSV успішно завантажено.")
except FileNotFoundError as e:
    logging.error(f"Файл не знайдено: {e.filename}")
    raise
except pd.errors.ParserError as e:
    logging.error(f"Помилка парсингу CSV: {e}")
    raise
except Exception as e:
    logging.error(f"Сталася помилка: {e}")
    raise

# Перевірка, чи файли не порожні
for name, df_temp in [('trans', trans), ('loan', loan), ('account', account), ('client', client), ('disp', disp)]:
    logging.info(f"Файл {name}: {df_temp.shape[0]} рядків, {df_temp.shape[1]} колонок")
    if df_temp.empty:
        logging.error(f"Файл {name} порожній! Переконайтеся, що він містить дані.")
        raise ValueError(f"Файл {name} порожній! Переконайтеся, що він містить дані.")

# Перевірка, що disp.csv містить необхідні колонки
required_disp_columns = {'disp_id', 'client_id', 'account_id', 'type'}
if not required_disp_columns.issubset(disp.columns):
    missing = required_disp_columns - set(disp.columns)
    logging.error(f"disp.csv не містить необхідних стовпців: {missing}")
    raise Exception(f"disp.csv не містить необхідних стовпців: {missing}")
else:
    logging.info("disp.csv містить всі необхідні стовпці.")

# Перед об'єднанням, перейменуємо 'trans.csv' та 'loan.csv' колонки, щоб уникнути конфліктів
trans.rename(columns={'amount': 'trans_amount', 'date': 'trans_date', 'type': 'trans_type'}, inplace=True)
loan.rename(columns={'amount': 'loan_amount', 'date': 'loan_date'}, inplace=True)

# 1) Об'єднуємо disp з client
logging.info("Починаємо злиття disp з client...")
df = disp.merge(client, on='client_id', how='inner')
logging.info(f"Після злиття disp з client: {df.shape[0]} рядків")

# 2) Злиття з account
logging.info("Починаємо злиття з account...")
df = df.merge(account, on='account_id', how='inner', suffixes=('_client', '_account'))
logging.info(f"Після злиття з account: {df.shape[0]} рядків")

# 3) Злиття з trans
logging.info("Починаємо злиття з trans...")
df = df.merge(trans, on='account_id', how='left')
logging.info(f"Після злиття з trans: {df.shape[0]} рядків")

# 4) Злиття з loan
logging.info("Починаємо злиття з loan...")
df = df.merge(loan, on='account_id', how='left')
logging.info(f"Після злиття з loan: {df.shape[0]} рядків")

# Перевірка розміру df після злиття
logging.info(f"Розмір df після всіх злиттів: {df.shape}")
if df.empty:
    logging.error("Після об'єднання даних `df` вийшов порожнім! Перевірте злиття таблиць.")
    raise ValueError("Після об'єднання даних `df` вийшов порожнім! Перевірте злиття таблиць.")

# Замінюємо NaN у 'loan_amount' на 0
if 'loan_amount' in df.columns:
    df['loan_amount'] = df['loan_amount'].fillna(0)
    logging.info("Замінено NaN у 'loan_amount' на 0")
    # Заповнення 'loan_date' з правильним типом
    if df['loan_date'].dtype == object:
        df['loan_date'] = df['loan_date'].fillna('19000101')
    else:
        df['loan_date'] = pd.to_datetime(df['loan_date'], errors='coerce').fillna(pd.Timestamp('1900-01-01'))
    logging.info("'loan_date' оброблено успішно.")

# Замінюємо NaN у 'trans_amount' на 0
if 'trans_amount' in df.columns:
    df['trans_amount'] = df['trans_amount'].fillna(0)
    logging.info("Замінено NaN у 'trans_amount' на 0")
    # Заповнення 'trans_date' з правильним типом
    if df['trans_date'].dtype == object:
        df['trans_date'] = df['trans_date'].fillna('19000101')
    else:
        df['trans_date'] = pd.to_datetime(df['trans_date'], errors='coerce').fillna(pd.Timestamp('1900-01-01'))
    logging.info("'trans_date' оброблено успішно.")

# Якщо balance відсутній — замінимо на 0
if 'balance' in df.columns:
    df['balance'] = df['balance'].fillna(0)
    logging.info("Замінено NaN у 'balance' на 0")

# Обробка birth_number
def extract_gender(birth_number):
    """Якщо birth_number має >= 6 цифр (формат YYMMDD), витягуємо стать з 6-ї або 7-ї цифри."""
    if pd.isna(birth_number):
        return "Unknown"
    try:
        s = str(int(birth_number))
        # Наприклад, якщо s='930705' -> 6 цифр
        if len(s) < 6:
            return "Unknown"
        # Припустимо, беремо 5-й індекс (шіста цифра, s[5]) для статі
        # Або s[6] якщо вважаємо 7-ма цифра = стать
        # Залежить від ваших реальних даних!
        # Припустимо, якщо len=6, беремо s[5]. Якщо len=7 або більше, беремо s[6].
        if len(s) == 6:
            idx_for_gender = 5
        elif len(s) >= 7:
            idx_for_gender = 6
        else:
            return "Unknown"
        if idx_for_gender >= len(s):
            return "Unknown"
        digit = int(s[idx_for_gender])
        return "Male" if digit % 2 == 1 else "Female"
    except:
        return "Unknown"

def calculate_age(birth_number, reference_year=2025):
    """Для 6- або 7-цифрового birth_number: YYMMDD.
       Якщо YY<=25 => 2000+YY, інакше => 1900+YY."""
    if pd.isna(birth_number):
        return 0
    try:
        s = str(int(birth_number))
        if len(s) < 2:
            return 0
        yy = int(s[:2])
        if yy <= 25:
            year = 2000 + yy
        else:
            year = 1900 + yy
        return reference_year - year
    except:
        return 0

if 'birth_number' in df.columns:
    logging.info("Починаємо обробку 'birth_number'...")
    df['gender'] = df['birth_number'].apply(extract_gender)
    df['age_raw'] = df['birth_number'].apply(calculate_age)
    logging.info("Обробка 'birth_number' завершена.")
    # Не видаляємо рядки, залишаємо 'Unknown' та '0'
else:
    logging.warning("'birth_number' відсутній у даних.")

logging.info(f"Після обробки birth_number, розмір df: {df.shape} рядків")

# Перевірка наявності числових стовпців
required_num_cols = ['trans_amount', 'loan_amount', 'balance', 'age_raw']
missing_num_cols = [c for c in required_num_cols if c not in df.columns]
if missing_num_cols:
    logging.error(f"Відсутні необхідні числові стовпці: {missing_num_cols}")
    raise Exception(f"Відсутні необхідні числові стовпці: {missing_num_cols}")
else:
    logging.info("Усі необхідні числові стовпці присутні.")

# Перекодування категоріальних змінних
if 'gender' in df.columns:
    logging.info("Перекодовування 'gender'...")
    le_gender = LabelEncoder()
    df['gender_encoded'] = le_gender.fit_transform(df['gender'])
else:
    logging.warning("'gender' відсутній у даних. Призначаємо 0.")
    df['gender_encoded'] = 0

if 'trans_type' in df.columns:
    logging.info("Перекодовування 'trans_type'...")
    le_trans = LabelEncoder()
    df['trans_type_encoded'] = le_trans.fit_transform(df['trans_type'].fillna('Unknown'))
else:
    logging.warning("'trans_type' відсутній у даних. Призначаємо 0.")
    df['trans_type_encoded'] = 0

if 'loan_type' in df.columns:
    logging.info("Перекодовування 'loan_type'...")
    le_loan_type = LabelEncoder()
    df['loan_type_encoded'] = le_loan_type.fit_transform(df['loan_type'].fillna('NoLoanType'))
else:
    logging.warning("'loan_type' відсутній у даних. Призначаємо 0.")
    df['loan_type_encoded'] = 0

# Масштабування числових даних (виключаючи 'age_raw')
logging.info("Починаємо масштабування числових даних...")
scaler = StandardScaler()
num_features = ['trans_amount', 'loan_amount', 'balance']

# Перевірка, що df не порожній
if df.empty:
    logging.error("Після всіх об'єднань та обробок df порожній. Перевірте формат birth_number та інші дані.")
    raise Exception("Після всіх об'єднань та обробок df порожній. Перевірте формат birth_number та інші дані.")

if df[num_features].empty or df[num_features].shape[0] == 0:
    logging.error("Немає жодного рядка з валідними числовими даними!")
    raise Exception("Немає жодного рядка з валідними числовими даними!")

# Оптимізація типів даних
df[num_features] = df[num_features].astype('float32')

# Масштабування
df[num_features] = scaler.fit_transform(df[num_features])

# Створення 'age_scaled'
df['age_scaled'] = (df['age_raw'] - df['age_raw'].mean()) / df['age_raw'].std()

logging.info(f"Масштабування числових даних виконано успішно. Розмір df: {df.shape} рядків, {df.shape[1]} стовпців.")


#################### ЗАДАЧІ DATA MINING (класифікація, регресія тощо) ####################

# Функція для генерації SVG спарклайнів
def generate_svg_sparkline(series, width=200, height=50, color='blue'):
    if not len(series):
        series = [0]
    max_val = max(series) if max(series) != 0 else 1
    points = " ".join(
        [f"{i * (width / len(series))},{height - (val / max_val * height)}" for i, val in enumerate(series)]
    )
    svg = f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg"><polyline points="{points}" style="fill:none;stroke:{color};stroke-width:2" /></svg>'
    return f"data:image/svg+xml;base64,{base64.b64encode(svg.encode('utf-8')).decode('utf-8')}"

# Завдання класифікації дефолту
def classification_task_extended(data):
    logging.info("Розпочато завдання класифікації дефолту...")
    data['default'] = np.where(data['loan_amount'] > 0, 1, 0)
    X = data[['trans_amount', 'loan_amount', 'balance', 'age_scaled', 'gender_encoded', 'trans_type_encoded',
              'loan_type_encoded']]
    y = data['default']

    if X.empty:
        logging.warning("Немає даних для класифікації.")
        return pd.DataFrame([{"Model": "NoData", "Accuracy": 0, "F1 Score": 0}]), None, None

    if y.nunique() < 2:
        logging.warning("Недостатньо класів для класифікації.")
        return pd.DataFrame([{"Model": "NoData", "Accuracy": 0, "F1 Score": 0}]), None, None

    logging.info("Розділення даних на тренувальні та тестові...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    logging.info("Навчання Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, n_jobs=-1)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    f1_lr = f1_score(y_test, y_pred_lr)

    logging.info("Навчання Random Forest Classifier...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf)

    logging.info("Класифікація дефолту завершена.")

    results = {
        'Model': ['Logistic Regression', 'Random Forest Classifier'],
        'Accuracy': [round(accuracy_lr, 4), round(accuracy_rf, 4)],
        'F1 Score': [round(f1_lr, 4), round(f1_rf, 4)]
    }

    df_results = pd.DataFrame(results)
    return df_results, y_test, y_pred_rf

# Завдання регресії транзакцій
def regression_task_extended(data):
    logging.info("Розпочато завдання регресії транзакцій...")
    X = data[['loan_amount', 'balance', 'age_scaled', 'gender_encoded', 'trans_type_encoded', 'loan_type_encoded']]
    y = data['trans_amount']

    if X.empty or y.empty:
        logging.warning("Немає даних для регресії.")
        return pd.DataFrame([{"Model": "NoData", "MSE": 0}]), None, None

    logging.info("Розділення даних на тренувальні та тестові...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    logging.info("Навчання Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    mse_lr = mean_squared_error(y_test, y_pred_lr)

    logging.info("Навчання Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)

    logging.info("Регресія транзакцій завершена.")

    results = {
        'Model': ['Linear Regression', 'Random Forest Regressor'],
        'MSE': [round(mse_lr, 4), round(mse_rf, 4)]
    }

    df_results = pd.DataFrame(results)
    return df_results, y_test, y_pred_rf

# Завдання кластеризації клієнтів
def clustering_task_extended(data):
    logging.info("Розпочато завдання кластеризації клієнтів...")
    X = data[['age_scaled', 'gender_encoded']]

    if X.empty:
        logging.warning("Немає даних для кластеризації.")
        return pd.DataFrame([{"Model": "NoData", "Silhouette Score": 0}]), None, None

    logging.info("Навчання MiniBatchKMeans кластеризатора...")
    kmeans = MiniBatchKMeans(n_clusters=3, random_state=42, batch_size=10000)
    labels_km = kmeans.fit_predict(X)
    logging.info("Обчислення Silhouette Score для MiniBatchKMeans...")

    # Використовуємо вибірку для обчислення silhouette_score
    sample_size = 10000
    if len(X) > sample_size:
        sample_indices = random.sample(range(len(X)), sample_size)
        silhouette_km = silhouette_score(X.iloc[sample_indices], labels_km[sample_indices])
    else:
        silhouette_km = silhouette_score(X, labels_km)

    logging.info("Навчання Random Forest Classifier для кластеризації...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, labels_km)
    labels_rf = rf.predict(X)
    logging.info("Обчислення Silhouette Score для Random Forest Classification...")

    if len(X) > sample_size:
        silhouette_rf = silhouette_score(X.iloc[sample_indices], labels_rf[sample_indices])
    else:
        silhouette_rf = silhouette_score(X, labels_rf)

    logging.info("Кластеризація клієнтів завершена.")

    results = {
        'Model': ['MiniBatchKMeans', 'Random Forest Classification (Imitated Clusters)'],
        'Silhouette Score': [round(silhouette_km, 4), round(silhouette_rf, 4)]
    }

    df_results = pd.DataFrame(results)
    return df_results, labels_km, labels_rf

# Завдання виявлення залежностей
def dependency_task_extended(data):
    logging.info("Розпочато завдання виявлення залежностей...")
    data['active_credit'] = np.where(data['loan_amount'] > 0, 1, 0)
    X = data[['trans_amount', 'loan_amount', 'balance', 'age_scaled', 'gender_encoded', 'trans_type_encoded',
              'loan_type_encoded']]
    y = data['active_credit']

    if X.empty:
        logging.warning("Немає даних для виявлення залежностей.")
        return pd.DataFrame([{"Model": "NoData", "Accuracy": 0, "F1 Score": 0}]), None, None

    if y.nunique() < 2:
        logging.warning("Недостатньо класів для виявлення залежностей.")
        return pd.DataFrame([{"Model": "NoData", "Accuracy": 0, "F1 Score": 0}]), None, None

    logging.info("Розділення даних на тренувальні та тестові...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    logging.info("Навчання Support Vector Machine...")
    svc = SVC(kernel='linear', random_state=42)
    svc.fit(X_train, y_train)
    y_pred_svc = svc.predict(X_test)
    accuracy_svc = accuracy_score(y_test, y_pred_svc)
    f1_svc = f1_score(y_test, y_pred_svc)

    logging.info("Навчання Random Forest Classifier...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf)

    logging.info("Виявлення залежностей завершено.")

    results = {
        'Model': ['Support Vector Machine', 'Random Forest Classifier'],
        'Accuracy': [round(accuracy_svc, 4), round(accuracy_rf, 4)],
        'F1 Score': [round(f1_svc, 4), round(f1_rf, 4)]
    }

    df_results = pd.DataFrame(results)
    return df_results, y_test, y_pred_rf

# Функція для генерації таблиці зі спарклайнами
def generate_sparkline_table(data):
    logging.info("Генерація спарклайнів для топ-10 банків...")
    if 'bank' not in data.columns:
        logging.warning("'bank' відсутній у даних.")
        return pd.DataFrame(columns=['bank', 'trans_amount', 'sparkline'])

    df_region = data.groupby('bank')['trans_amount'].sum().reset_index()
    top10 = df_region.nlargest(10, 'trans_amount')
    data['year'] = pd.to_datetime(data['trans_date'], errors='coerce').dt.year
    rows = []
    for _, row in top10.iterrows():
        bank_name = row['bank']
        sub = data[data['bank'] == bank_name].sort_values('year')['trans_amount']
        spark = generate_svg_sparkline(sub.tolist())
        rows.append({
            'bank': bank_name,
            'trans_amount': f"{row['trans_amount']:.2f}",
            'sparkline': f"<img src='{spark}' style='width:200px; height:50px;'/>"
        })
    logging.info("Генерація спарклайнів завершена.")
    return pd.DataFrame(rows)

# Генерація спарклайн таблиці
df_spark_example = generate_sparkline_table(df)

# Визначення функцій для додаткових графіків
def plot_feature_importance(model, feature_names, title):
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)
    fig = go.Figure(go.Bar(
        x=importance[sorted_idx],
        y=np.array(feature_names)[sorted_idx],
        orientation='h'
    ))
    fig.update_layout(title=title, xaxis_title='Важливість', yaxis_title='Ознаки')
    return fig

def plot_confusion_matrix_image(y_true, y_pred, title):
    if y_true is None or y_pred is None:
        logging.error("y_true або y_pred є None. Неможливо побудувати матрицю плутанини.")
        return ""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.title(title)
    plt.xlabel('Прогнозований клас')
    plt.ylabel('Фактичний клас')
    img = base64.b64encode(fig_to_image(fig)).decode('utf-8')
    plt.close(fig)
    return f'data:image/png;base64,{img}'

def plot_actual_vs_predicted(y_true, y_pred, title):
    if y_true is None or y_pred is None:
        logging.error("y_true або y_pred є None. Неможливо побудувати графік Actual vs Predicted.")
        return go.Figure()
    fig = px.scatter(x=y_true, y=y_pred, labels={'x': 'Фактичні Значення', 'y': 'Прогнозовані Значення'}, title=title)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_shape(type='line', x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                  line=dict(color='Red', dash='dash'))
    return fig

def plot_clusters(data, labels, title):
    if data.empty or labels is None:
        logging.warning("Немає даних для побудови кластерів.")
        return go.Figure()
    pca = PCA(n_components=2)
    components = pca.fit_transform(data)
    df_pca = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
    df_pca['Cluster'] = labels
    fig = px.scatter(df_pca, x='PC1', y='PC2', color='Cluster',
                     title=title, labels={'PC1': 'Компонента 1', 'PC2': 'Компонента 2'})
    return fig

def plot_correlation_matrix(data, title):
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    plt.title(title)
    img = base64.b64encode(fig_to_image(fig)).decode('utf-8')
    plt.close(fig)
    return f'data:image/png;base64,{img}'

def generate_summary_table(data):
    summary = data.describe().reset_index()
    summary.rename(columns={'index': 'Statistic'}, inplace=True)
    return summary

# Виклик завдань
df_classification, y_test_class, y_pred_rf_class = classification_task_extended(df)
df_regression, y_test_reg, y_pred_rf_reg = regression_task_extended(df)
df_clustering, labels_km, labels_rf = clustering_task_extended(df)
df_dependency, y_test_dep, y_pred_rf_dep = dependency_task_extended(df)

# Генерація таблиці зі статистичним описом
df_summary = generate_summary_table(df[['trans_amount', 'loan_amount', 'balance', 'age_raw']])

# Генерація кореляційної матриці
correlation_image = plot_correlation_matrix(df[['trans_amount', 'loan_amount', 'balance', 'age_raw']],
                                            "Кореляція між числовими ознаками")

# Отримання моделей для важливості ознак
def get_trained_models(data):
    # Класифікація дефолту
    data['default'] = np.where(data['loan_amount'] > 0, 1, 0)
    X_class = data[['trans_amount', 'loan_amount', 'balance', 'age_scaled', 'gender_encoded', 'trans_type_encoded',
                    'loan_type_encoded']]
    y_class = data['default']
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
        X_class, y_class, test_size=0.3, random_state=42, stratify=y_class
    )
    rf_class = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_class.fit(X_train_class, y_train_class)

    # Регресія транзакцій
    X_reg = data[['loan_amount', 'balance', 'age_scaled', 'gender_encoded', 'trans_type_encoded', 'loan_type_encoded']]
    y_reg = data['trans_amount']
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.3, random_state=42
    )
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_reg.fit(X_train_reg, y_train_reg)

    return rf_class, rf_reg, X_class.columns.tolist(), X_reg.columns.tolist()

rf_class_model, rf_reg_model, features_class, features_reg = get_trained_models(df)

# Візуалізація важливості ознак
feature_importance_class = plot_feature_importance(rf_class_model, features_class,
                                                   "Важливість Ознак - Класифікація Дефолту")
feature_importance_reg = plot_feature_importance(rf_reg_model, features_reg, "Важливість Ознак - Регресія Транзакцій")

# Генерація таблиці зі спарклайнами
df_spark_example = generate_sparkline_table(df)

# Генерація таблиці зі статистичним описом
df_summary = generate_summary_table(df[['trans_amount', 'loan_amount', 'balance', 'age_raw']])

# Генерація кореляційної матриці
correlation_image = plot_correlation_matrix(df[['trans_amount', 'loan_amount', 'balance', 'age_raw']],
                                            "Кореляція між числовими ознаками")

# Визначення макету Dash
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Business Analytics Dashboard"), className="text-center mb-4")
    ]),

    # Інтерактивні фільтри
    dbc.Row([
        dbc.Col([
            html.Label("Виберіть Стать:"),
            dcc.Dropdown(
                id='gender-filter',
                options=[
                    {'label': 'Male', 'value': 'Male'},
                    {'label': 'Female', 'value': 'Female'},
                    {'label': 'Unknown', 'value': 'Unknown'}
                ],
                value=['Male', 'Female', 'Unknown'],
                multi=True
            )
        ], width=4),
        dbc.Col([
            html.Label("Виберіть Вік:"),
            dcc.RangeSlider(
                id='age-filter',
                min=df['age_raw'].min(),
                max=df['age_raw'].max(),
                step=1,
                value=[20, 60],
                marks={i: str(i) for i in range(int(df['age_raw'].min()), int(df['age_raw'].max()) + 1, 10)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], width=8)
    ], className="mb-4"),

    # Task 1: Classification of Loan Default
    dbc.Card([
        dbc.CardHeader("Task 1: Classification of Loan Default"),
        dbc.CardBody([
            dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in df_classification.columns],
                data=df_classification.to_dict('records'),
                style_cell={'textAlign': 'center'},
                style_header={'backgroundColor': '#f1f1f1', 'fontWeight': 'bold'},
                page_size=5
            ),
            dcc.Graph(id='f1-score-classification'),
            html.Img(id='confusion-matrix-classification', style={'width': '50%', 'margin': 'auto', 'display': 'block'})
        ])
    ], className="mb-4"),

    # Task 2: Prediction of Transaction Amount
    dbc.Card([
        dbc.CardHeader("Task 2: Prediction of Transaction Amount"),
        dbc.CardBody([
            dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in df_regression.columns],
                data=df_regression.to_dict('records'),
                style_cell={'textAlign': 'center'},
                style_header={'backgroundColor': '#f1f1f1', 'fontWeight': 'bold'},
                page_size=5
            ),
            dcc.Graph(id='mse-regression'),
            dcc.Graph(id='actual-vs-predicted-regression')
        ])
    ], className="mb-4"),

    # Task 3: Customer Clustering
    dbc.Card([
        dbc.CardHeader("Task 3: Customer Clustering"),
        dbc.CardBody([
            dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in df_clustering.columns],
                data=df_clustering.to_dict('records'),
                style_cell={'textAlign': 'center'},
                style_header={'backgroundColor': '#f1f1f1', 'fontWeight': 'bold'},
                page_size=5
            ),
            dcc.Graph(id='silhouette-score-clustering'),
            dcc.Graph(id='clusters-kmeans'),
            dcc.Graph(id='clusters-rf')
        ])
    ], className="mb-4"),

    # Task 4: Dependency Between Transaction Types and Credit Activity
    dbc.Card([
        dbc.CardHeader("Task 4: Dependency Between Transaction Types and Credit Activity"),
        dbc.CardBody([
            dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in df_dependency.columns],
                data=df_dependency.to_dict('records'),
                style_cell={'textAlign': 'center'},
                style_header={'backgroundColor': '#f1f1f1', 'fontWeight': 'bold'},
                page_size=5
            ),
            dcc.Graph(
                figure=px.bar(df_dependency, x='Model', y='F1 Score', title="F1 Score Comparison", text='F1 Score')
            )
        ])
    ], className="mb-4"),

    # Sparkline Table: Transaction Trends by Bank
    dbc.Card([
        dbc.CardHeader("Sparkline Table: Transaction Trends by Bank"),
        dbc.CardBody([
            dash_table.DataTable(
                id='sparkline-table',
                columns=[
                    {"name": "Bank", "id": "bank"},
                    {"name": "Transaction Sum", "id": "trans_amount"},
                    {"name": "Sparkline", "id": "sparkline", "presentation": "markdown"}
                ],
                data=df_spark_example.to_dict('records'),
                style_cell={'textAlign': 'center'},
                style_header={'backgroundColor': '#f1f1f1', 'fontWeight': 'bold'},
                markdown_options={"html": True},
                page_size=10
            )
        ])
    ], className="mb-4"),

    # Correlation Matrix
    dbc.Card([
        dbc.CardHeader("Кореляційна Матриця"),
        dbc.CardBody([
            html.Img(src=correlation_image, style={'width': '100%', 'height': 'auto'})
        ])
    ], className="mb-4"),

    # Statistical Summary
    dbc.Card([
        dbc.CardHeader("Статистичний Опис Даних"),
        dbc.CardBody([
            dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in df_summary.columns],
                data=df_summary.to_dict('records'),
                style_cell={'textAlign': 'center'},
                style_header={'backgroundColor': '#f1f1f1', 'fontWeight': 'bold'},
                page_size=10
            )
        ])
    ], className="mb-4"),

    # Feature Importance - Classification
    dbc.Card([
        dbc.CardHeader("Feature Importance - Classification"),
        dbc.CardBody([
            dcc.Graph(figure=feature_importance_class)
        ])
    ], className="mb-4"),

    # Feature Importance - Regression
    dbc.Card([
        dbc.CardHeader("Feature Importance - Regression"),
        dbc.CardBody([
            dcc.Graph(figure=feature_importance_reg)
        ])
    ], className="mb-4"),

], fluid=True)


#################### CALLBACKS ####################

# Callback для оновлення класифікації дефолту
@app.callback(
    Output('f1-score-classification', 'figure'),
    Output('confusion-matrix-classification', 'src'),
    Input('gender-filter', 'value'),
    Input('age-filter', 'value')
)
def update_classification(genders, age_range):
    logging.info("Оновлення класифікації дефолту з фільтрами: %s, %s", genders, age_range)
    filtered_df = df[
        (df['gender'].isin(genders)) &
        (df['age_raw'] >= age_range[0]) &
        (df['age_raw'] <= age_range[1])
    ]

    if filtered_df.empty:
        logging.warning("Фільтрований датафрейм порожній після фільтрів!")
        logging.warning(f"У вихідному df {df.shape[0]} рядків перед фільтрацією.")
        logging.warning(f"Кількість унікальних значень статі: {df['gender'].unique()}")
        logging.warning(f"Мінімальний та максимальний вік у даних: {df['age_raw'].min()}, {df['age_raw'].max()}")
        # Повертаємо порожній графік та порожнє зображення
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Недостатньо даних для класифікації дефолту")
        return empty_fig, ""

    df_classification, y_test_class, y_pred_rf_class = classification_task_extended(filtered_df)

    if y_test_class is None or y_pred_rf_class is None:
        logging.warning("Недостатньо класів для побудови матриці плутанини.")
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Недостатньо даних для класифікації дефолту")
        return empty_fig, ""

    fig_f1 = px.bar(
        df_classification,
        x='Model',
        y='F1 Score',
        title="F1 Score Comparison",
        text='F1 Score'
    )
    fig_f1.update_traces(textposition='outside')
    fig_f1.update_layout(yaxis=dict(range=[0, 1.1]))

    confusion_img = plot_confusion_matrix_image(y_test_class, y_pred_rf_class, "Матриця Плутанини - Random Forest")

    return fig_f1, confusion_img

# Callback для оновлення регресії
@app.callback(
    Output('mse-regression', 'figure'),
    Output('actual-vs-predicted-regression', 'figure'),
    Input('gender-filter', 'value'),
    Input('age-filter', 'value')
)
def update_regression(genders, age_range):
    logging.info("Оновлення регресії з фільтрами: %s, %s", genders, age_range)
    filtered_df = df[
        (df['gender'].isin(genders)) &
        (df['age_raw'] >= age_range[0]) &
        (df['age_raw'] <= age_range[1])
    ]

    if filtered_df.empty:
        logging.warning("Фільтрований датафрейм порожній після фільтрів!")
        logging.warning(f"У вихідному df {df.shape[0]} рядків перед фільтрацією.")
        logging.warning(f"Кількість унікальних значень статі: {df['gender'].unique()}")
        logging.warning(f"Мінімальний та максимальний вік у даних: {df['age_raw'].min()}, {df['age_raw'].max()}")
        # Повертаємо порожні графіки
        empty_fig_mse = go.Figure()
        empty_fig_mse.update_layout(title="Недостатньо даних для регресії транзакцій")
        empty_fig_actual_pred = go.Figure()
        empty_fig_actual_pred.update_layout(title="Недостатньо даних для регресії транзакцій")
        return empty_fig_mse, empty_fig_actual_pred

    df_regression, y_test_reg, y_pred_rf_reg = regression_task_extended(filtered_df)

    if y_test_reg is None or y_pred_rf_reg is None:
        logging.warning("Недостатньо даних для побудови графіка регресії.")
        empty_fig_mse = go.Figure()
        empty_fig_mse.update_layout(title="Недостатньо даних для регресії транзакцій")
        empty_fig_actual_pred = go.Figure()
        empty_fig_actual_pred.update_layout(title="Недостатньо даних для регресії транзакцій")
        return empty_fig_mse, empty_fig_actual_pred

    fig_mse = px.bar(
        df_regression,
        x='Model',
        y='MSE',
        title="MSE Comparison",
        text='MSE'
    )
    fig_mse.update_traces(textposition='outside')
    fig_mse.update_layout(yaxis=dict(range=[0, df_regression['MSE'].max() * 1.1]))

    fig_actual_pred = plot_actual_vs_predicted(y_test_reg, y_pred_rf_reg, "Actual vs Predicted - Random Forest Regressor")

    return fig_mse, fig_actual_pred

# Callback для оновлення кластеризації
@app.callback(
    Output('silhouette-score-clustering', 'figure'),
    Output('clusters-kmeans', 'figure'),
    Output('clusters-rf', 'figure'),
    Input('gender-filter', 'value'),
    Input('age-filter', 'value')
)
def update_clustering(genders, age_range):
    logging.info("Оновлення кластеризації клієнтів з фільтрами: %s, %s", genders, age_range)
    filtered_df = df[
        (df['gender'].isin(genders)) &
        (df['age_raw'] >= age_range[0]) &
        (df['age_raw'] <= age_range[1])
    ]

    if filtered_df.empty:
        logging.warning("Фільтрований датафрейм порожній після фільтрів!")
        logging.warning(f"У вихідному df {df.shape[0]} рядків перед фільтрацією.")
        logging.warning(f"Кількість унікальних значень статі: {df['gender'].unique()}")
        logging.warning(f"Мінімальний та максимальний вік у даних: {df['age_raw'].min()}, {df['age_raw'].max()}")
        # Повертаємо порожні графіки
        empty_fig_silhouette = go.Figure()
        empty_fig_silhouette.update_layout(title="Недостатньо даних для кластеризації клієнтів")
        empty_fig_km = go.Figure()
        empty_fig_km.update_layout(title="Недостатньо даних для кластеризації клієнтів")
        empty_fig_rf = go.Figure()
        empty_fig_rf.update_layout(title="Недостатньо даних для кластеризації клієнтів")
        return empty_fig_silhouette, empty_fig_km, empty_fig_rf

    df_clustering, labels_km, labels_rf = clustering_task_extended(filtered_df)

    if df_clustering.empty:
        logging.warning("Недостатньо даних для побудови кластерів.")
        empty_fig_silhouette = go.Figure()
        empty_fig_silhouette.update_layout(title="Недостатньо даних для кластеризації клієнтів")
        empty_fig_km = go.Figure()
        empty_fig_km.update_layout(title="Недостатньо даних для кластеризації клієнтів")
        empty_fig_rf = go.Figure()
        empty_fig_rf.update_layout(title="Недостатньо даних для кластеризації клієнтів")
        return empty_fig_silhouette, empty_fig_km, empty_fig_rf

    fig_silhouette = px.bar(
        df_clustering,
        x='Model',
        y='Silhouette Score',
        title="Silhouette Score Comparison",
        text='Silhouette Score'
    )
    fig_silhouette.update_traces(textposition='outside')
    fig_silhouette.update_layout(yaxis=dict(range=[0, 1]))

    if labels_km is not None:
        fig_km = plot_clusters(filtered_df[['age_scaled', 'gender_encoded']], labels_km, "Кластери - MiniBatchKMeans")
    else:
        fig_km = go.Figure()
        fig_km.update_layout(title="Недостатньо даних для MiniBatchKMeans кластеризації")

    if labels_rf is not None:
        fig_rf = plot_clusters(filtered_df[['age_scaled', 'gender_encoded']], labels_rf,
                               "Кластери - Random Forest Classification")
    else:
        fig_rf = go.Figure()
        fig_rf.update_layout(title="Недостатньо даних для Random Forest кластеризації")

    return fig_silhouette, fig_km, fig_rf


#################### ЗАДАЧІ ДЛЯ ВИЗУАЛІЗАЦІЙ ####################

#################### ЗАДАЧІ ДЛЯ ВИЗУАЛІЗАЦІЙ ####################

# Візуалізація важливості ознак
def plot_feature_importance(model, feature_names, title):
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)
    fig = go.Figure(go.Bar(
        x=importance[sorted_idx],
        y=np.array(feature_names)[sorted_idx],
        orientation='h'
    ))
    fig.update_layout(title=title, xaxis_title='Важливість', yaxis_title='Ознаки')
    return fig

# Візуалізація матриці плутанини та Actual vs Predicted
def plot_confusion_matrix_image(y_true, y_pred, title):
    if y_true is None or y_pred is None:
        logging.error("y_true або y_pred є None. Неможливо побудувати матрицю плутанини.")
        return ""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.title(title)
    plt.xlabel('Прогнозований клас')
    plt.ylabel('Фактичний клас')
    img = base64.b64encode(fig_to_image(fig)).decode('utf-8')
    plt.close(fig)
    return f'data:image/png;base64,{img}'

def plot_actual_vs_predicted(y_true, y_pred, title):
    if y_true is None or y_pred is None:
        logging.error("y_true або y_pred є None. Неможливо побудувати графік Actual vs Predicted.")
        return go.Figure()
    fig = px.scatter(x=y_true, y=y_pred, labels={'x': 'Фактичні Значення', 'y': 'Прогнозовані Значення'}, title=title)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_shape(type='line', x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                  line=dict(color='Red', dash='dash'))
    return fig

# Візуалізація кластерів відбувається
def plot_clusters(data, labels, title):
    if data.empty or labels is None:
        logging.warning("Немає даних для побудови кластерів.")
        return go.Figure()
    pca = PCA(n_components=2)
    components = pca.fit_transform(data)
    df_pca = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
    df_pca['Cluster'] = labels
    fig = px.scatter(df_pca, x='PC1', y='PC2', color='Cluster',
                     title=title, labels={'PC1': 'Компонента 1', 'PC2': 'Компонента 2'})
    return fig


#################### Додаткові Перевірки ####################

# Перевірка, чи df містить всі необхідні колонки перед моделлю
expected_columns = ['trans_amount', 'loan_amount', 'balance', 'age_raw', 'gender_encoded', 'trans_type_encoded',
                    'loan_type_encoded', 'age_scaled']
missing_columns = [col for col in expected_columns if col not in df.columns]
if missing_columns:
    logging.error(f"Відсутні колонки у df перед моделюванням: {missing_columns}")
    raise Exception(f"Відсутні колонки у df перед моделюванням: {missing_columns}")
else:
    logging.info("Усі необхідні колонки присутні у df перед моделюванням.")

#################### ЗАПУСК ДОДАТКУ ####################
if __name__ == '__main__':
    logging.info("Запуск Dash додатку...")
    app.run_server(debug=True, port=8050)
