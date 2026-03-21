# Customer Churn Prediction using Artificial Neural Networks

Predicts whether a bank customer will churn (leave) based on their account and demographic details. Built using a TensorFlow ANN with a Streamlit web interface for live predictions. Also includes a bonus regression model that predicts estimated salary using the same dataset.

---

## How it works

The dataset (`Churn_Modelling.csv`) contains 10,000 bank customers with features like credit score, geography, gender, age, tenure, balance, number of products, and activity status. The target is `Exited` — whether the customer churned.

**Preprocessing:**
- Dropped non-informative columns: `RowNumber`, `CustomerId`, `Surname`
- Label encoded `Gender` (Male/Female → 0/1)
- One-hot encoded `Geography` (France, Germany, Spain)
- Standard scaled all numerical features
- Encoders and scaler saved as `.pkl` files for reuse in the app

**ANN Architecture (Classification):**
- Input layer → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Sigmoid)
- Optimizer: Adam (lr=0.01), Loss: Binary Crossentropy
- Callbacks: EarlyStopping (patience=10, restore best weights) + TensorBoard

**ANN Architecture (Regression — salary prediction):**
- Input layer → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, linear)
- Optimizer: Adam, Loss: Mean Absolute Error
- Same callbacks setup, logs saved under `regressionlogs/`

---

## Hyperparameter Tuning

`hyperparametertuningann.ipynb` runs a GridSearchCV over the ANN using `scikeras.wrappers.KerasClassifier` to find the best combination of:

| Parameter | Values searched |
|---|---|
| neurons | 16, 32, 64, 128 |
| layers | 1, 2 |
| epochs | 50, 100 |

3-fold cross-validation, all combinations evaluated in parallel (`n_jobs=-1`).

---

## Streamlit App

`app.py` loads the saved model and encoders and runs a live prediction interface.

```bash
streamlit run app.py
```

Input fields: Geography, Gender, Age, Balance, Credit Score, Estimated Salary, Tenure, Number of Products, Has Credit Card, Is Active Member

Output: Churn probability score + a plain-English verdict (likely to churn / not likely to churn)

---

## Stack

Python · TensorFlow 2.20 · Keras · scikit-learn · scikeras · pandas · NumPy · Streamlit · pickle

---

## Project Layout

```
├── app.py                          # Streamlit web app
├── experiments.ipynb               # Data preprocessing, ANN training, TensorBoard
├── hyperparametertuningann.ipynb   # GridSearchCV over neurons, layers, epochs
├── prediction.ipynb                # Single-sample prediction walkthrough
├── salaryregression.ipynb          # ANN regression to predict estimated salary
│
├── Churn_Modelling.csv             # Dataset (10,000 customers)
│
├── model.keras                     # Saved classification model
├── regression_model.keras          # Saved regression model
├── model.h5                        # Alternative saved format
│
├── label_encoder_gender.pkl        # Saved LabelEncoder for Gender
├── onehot_encoder_geo.pkl          # Saved OneHotEncoder for Geography
├── scaler.pkl                      # Saved StandardScaler
│
├── logs/                           # TensorBoard logs (classification)
├── regressionlogs/                 # TensorBoard logs (regression)
└── requirements.txt
```

---

## Run it locally

```bash
git clone https://github.com/JainulTrivedi55555/Customer-Churn-Prediction-using-Artificial-Neural-Networks.git
cd Customer-Churn-Prediction-using-Artificial-Neural-Networks
pip install -r requirements.txt
streamlit run app.py
```
