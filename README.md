# Business Analytics Dashboard

This repository contains a Dash application that performs four Data Mining tasks on the Chech Bank dataset. The tasks include:

1. **Loan Default Classification**  
   - **Description:** Classify client defaults based on transaction and loan data. The target variable “default” is set to 1 (if a loan was taken) and 0 (if not).  
   - **Models Used:** Logistic Regression and Random Forest Classifier  
   - **Performance:** Evaluated using Accuracy and F1-Score. The Random Forest model showed superior performance.

2. **Transaction Amount Forecasting**  
   - **Description:** Forecast the transaction amount (trans_amount) using loan, balance, and other features.  
   - **Models Used:** Linear Regression and Random Forest Regressor  
   - **Performance:** Evaluated using Mean Squared Error (MSE). Random Forest Regressor produced a lower MSE compared to Linear Regression.

3. **Customer Clustering**  
   - **Description:** Cluster customers based on features such as age (age_scaled) and gender (gender_encoded) to identify groups with similar characteristics.  
   - **Methods Used:** MiniBatchKMeans and an emulated clustering approach using Random Forest Classification  
   - **Performance:** Evaluated using the Silhouette Score. Both methods yielded similar results, with room for further optimization of the number of clusters.

4. **Dependency Discovery**  
   - **Description:** Discover dependencies between transaction types and credit activity to determine if certain transaction types affect the likelihood of obtaining a loan.  
   - **Models Used:** Support Vector Machine (SVM) with a linear kernel and Random Forest Classifier  
   - **Performance:** Evaluated using Accuracy and F1-Score. Both models produced comparable results, though Random Forest may be more robust to noise and non-linear relationships.

---

## Загальний висновок / Overall Conclusion

**UA:**  
У ході виконання завдань було розроблено чотири різні підходи для аналізу даних. Використання різних моделей дозволило глибше проаналізувати дані та порівняти їхню продуктивність:
- **Класифікація дефолту:** Random Forest показав кращі результати порівняно з Logistic Regression.
- **Прогнозування транзакцій:** Random Forest Regressor продемонстрував нижчу середньоквадратичну помилку (MSE) порівняно з Linear Regression.
- **Кластеризація клієнтів:** Обидва методи дали схожі результати, проте подальше налаштування може покращити сегментацію.
- **Виявлення залежностей:** Моделі SVM і Random Forest ефективно виявляють залежності, з перевагою Random Forest щодо стійкості до шуму.

**EN:**  
Throughout this project, four distinct data mining approaches were developed. The use of various models allowed for a deeper analysis of the data and comparison of their performance:
- **Loan Default Classification:** Random Forest outperformed Logistic Regression.
- **Transaction Amount Forecasting:** Random Forest Regressor achieved a lower MSE compared to Linear Regression.
- **Customer Clustering:** Both methods produced similar results, though further tuning could enhance segmentation.
- **Dependency Discovery:** Both SVM and Random Forest effectively detected relationships, with Random Forest showing higher robustness to noise.

---

## How to Run the Application

### Prerequisites

1. **Python 3.12** (or higher)
2. **Virtual Environment (Recommended)**

### Installation Steps

1. **Clone the Repository**

   Open your terminal or command prompt and run:
   ```bash
   git clone https://github.com/xaxinotf/Bodya_BI_lab_3.git
   cd Bodya_BI_lab_3


```markdown

2. **Create and Activate a Virtual Environment**

   For Windows:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
   For macOS/Linux:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**

   Install the required Python libraries using pip:
   ```bash
   pip install -r requirements.txt
   ```
   *(If a `requirements.txt` file is not provided, ensure you have installed the following packages: `pandas`, `numpy`, `scikit-learn`, `plotly`, `dash`, `dash-bootstrap-components`, `seaborn`, `matplotlib`.)*

4. **Run the Application**

   Start the Dash app by running:
   ```bash
   python main.py
   ```
   You should see log messages indicating that the app has started, and it will be accessible at:
   ```
   http://127.0.0.1:8050/
   ```

5. **Access the Dashboard**

   Open your web browser and navigate to [http://127.0.0.1:8050/](http://127.0.0.1:8050/). The dashboard should load, displaying interactive visualizations for the four Data Mining tasks.

---


## Repository Structure

- `main.py` – Main Dash application code.
- `account.csv`, `client.csv`, `disp.csv`, `loan.csv`, `trans.csv` – Data files.
- `app.log` – Log file for application events.
- Additional CSV files (e.g., `dim_account.csv`, `dim_client.csv`, etc.) may be included for extended analysis.

---





### Міні репорт / Mini Report

---

#### Завдання 1: Класифікація дефолту по кредитах  
**Опис задачі (UA):**  
Було поставлено задачу класифікації дефолту клієнтів на основі даних транзакцій і кредитів. Цільова змінна «default» визначається як 1 (якщо кредит був оформлений) або 0 (якщо кредит не був оформлений).  

**Використані моделі (UA):**  
- Logistic Regression  
- Random Forest Classifier  

**Оцінка продуктивності (UA):**  
Результати оцінювалися за показниками точності (Accuracy) та F1-Score. Порівняльний аналіз показав, що модель Random Forest має вищі значення цих метрик, що свідчить про її кращу здатність розпізнавати дефолтні випадки.  

**Висновок (UA):**  
Random Forest є більш ефективним підходом для цієї задачі; проте подальша оптимізація параметрів може додатково покращити результати.

---

**Task 1: Loan Default Classification**  
**Task Description (EN):**  
The goal was to classify client defaults based on transaction and loan data. The target variable "default" is defined as 1 (if a loan was taken) or 0 (if no loan was taken).  

**Models Used (EN):**  
- Logistic Regression  
- Random Forest Classifier  

**Performance Evaluation (EN):**  
The models were evaluated using Accuracy and F1-Score. Comparative analysis showed that the Random Forest model achieved higher values in these metrics, indicating its superior ability to detect default cases.  

**Conclusion (EN):**  
Random Forest is a more effective approach for this task, though further parameter tuning could further improve performance.

---

#### Завдання 2: Прогнозування суми транзакцій  
**Опис задачі (UA):**  
Метою було побудувати модель прогнозування суми транзакцій (trans_amount) з використанням даних про кредити, баланси та інших ознак.  

**Використані моделі (UA):**  
- Linear Regression  
- Random Forest Regressor  

**Оцінка продуктивності (UA):**  
Модель оцінювали за середньоквадратичною помилкою (MSE). Random Forest Regressor показав нижчу MSE, що свідчить про кращу здатність враховувати нелінійні залежності у даних.  

**Висновок (UA):**  
Для прогнозування суми транзакцій Random Forest Regressor демонструє вищу продуктивність порівняно з лінійною регресією.

---

**Task 2: Transaction Amount Forecasting**  
**Task Description (EN):**  
The objective was to build a model to forecast the transaction amount (trans_amount) using loan, balance, and other feature data.  

**Models Used (EN):**  
- Linear Regression  
- Random Forest Regressor  

**Performance Evaluation (EN):**  
Models were evaluated using Mean Squared Error (MSE). The Random Forest Regressor achieved a lower MSE, indicating its superior capability to capture non-linear relationships in the data.  

**Conclusion (EN):**  
For forecasting transaction amounts, the Random Forest Regressor demonstrates better performance compared to linear regression.

---

#### Завдання 3: Кластеризація клієнтів  
**Опис задачі (UA):**  
Завдання полягало у розподілі клієнтів на кластери за ознаками, такими як вік (age_scaled) та стать (gender_encoded). Метою було виявити групи клієнтів із схожими характеристиками.  

**Використані методи (UA):**  
- MiniBatchKMeans  
- Імітація кластеризації за допомогою Random Forest Classification  

**Оцінка продуктивності (UA):**  
Якість кластеризації оцінювалася за допомогою показника Silhouette Score. Обидва підходи дали подібні результати, хоча кожен з них має свої переваги та недоліки.  

**Висновок (UA):**  
Обидва методи можуть бути застосовані для кластеризації, проте подальше налаштування та оптимізація кількості кластерів може покращити результати сегментації.

---

**Task 3: Customer Clustering**  
**Task Description (EN):**  
The task was to cluster customers based on features such as age (age_scaled) and gender (gender_encoded) to identify groups with similar characteristics.  

**Methods Used (EN):**  
- MiniBatchKMeans  
- Emulated clustering using Random Forest Classification  

**Performance Evaluation (EN):**  
Clustering quality was measured using the Silhouette Score. Both approaches produced similar results, although each method has its own strengths and weaknesses.  

**Conclusion (EN):**  
Both methods can be used for clustering; however, further tuning and selection of the optimal number of clusters may enhance the segmentation quality.

---

#### Завдання 4: Виявлення залежностей у даних  
**Опис задачі (UA):**  
Було поставлено завдання виявлення залежностей між типом транзакцій та кредитною активністю клієнтів. Мета – визначити, чи впливає певний тип транзакції на ймовірність оформлення кредиту.  

**Використані моделі (UA):**  
- Support Vector Machine (SVM) з лінійним ядром  
- Random Forest Classifier  

**Оцінка продуктивності (UA):**  
Моделі оцінювалися за показниками точності (Accuracy) та F1-Score. Обидві моделі показали схожі результати, що свідчить про їх здатність виявляти залежності, хоча налаштування можуть вплинути на деякі нюанси.  

**Висновок (UA):**  
Обидва підходи ефективно виявляють залежності у даних, проте Random Forest Classifier може бути більш стійким до шуму та нелінійних залежностей.

---

**Task 4: Discovery of Dependencies in Data**  
**Task Description (EN):**  
The objective was to discover dependencies between the type of transactions and client credit activity, aiming to determine whether a certain type of transaction affects the likelihood of obtaining a loan.  

**Models Used (EN):**  
- Support Vector Machine (SVM) with a linear kernel  
- Random Forest Classifier  

**Performance Evaluation (EN):**  
Models were evaluated using Accuracy and F1-Score. Both models produced similar results, indicating their ability to detect dependencies, though fine-tuning may address some nuances.  

**Conclusion (EN):**  
Both approaches effectively detect relationships in the data. However, the Random Forest Classifier may be more robust against noise and non-linear dependencies.

---

### Загальний висновок / Overall Conclusion

**UA:**  
У ході виконання завдань було розроблено чотири різні підходи для аналізу даних. Використання різних моделей для кожного завдання дозволило глибше проаналізувати дані та порівняти їхню продуктивність:
- **Класифікація дефолту:** Logistic Regression та Random Forest Classifier показали, що Random Forest є ефективнішим.
- **Прогнозування транзакцій:** Random Forest Regressor показав кращу продуктивність порівняно з Linear Regression за показником MSE.
- **Кластеризація клієнтів:** MiniBatchKMeans та імітація кластеризації за допомогою Random Forest дозволили сегментувати клієнтів, що може бути покращено подальшою оптимізацією.
- **Виявлення залежностей:** Моделі SVM та Random Forest Classifier ефективно виявляють залежності, хоча Random Forest може бути більш стійким до шуму.

**EN:**  
During the project, four distinct data mining approaches were developed. The use of various models for each task allowed for a deeper analysis of the data and comparison of their performance:
- **Loan Default Classification:** Logistic Regression and Random Forest Classifier were used, with Random Forest proving to be more effective.
- **Transaction Amount Forecasting:** Random Forest Regressor demonstrated superior performance over Linear Regression, as evidenced by lower MSE.
- **Customer Clustering:** Both MiniBatchKMeans and an emulated clustering approach using Random Forest were applied to segment customers; further tuning may improve segmentation quality.
- **Discovery of Dependencies:** Both SVM and Random Forest Classifier effectively uncovered relationships in the data, with Random Forest showing enhanced robustness to noise and non-linear relationships.

Overall, employing different models for each task allowed for a comprehensive analysis of the data and successfully met the project requirements.

