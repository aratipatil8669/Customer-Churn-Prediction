
# Customer Churn Prediction

## Project Overview
This project uses Machine Learning to predict whether a telecom customer is likely to leave the company.

It helps businesses identify customers at risk of leaving and take suitable retention actions.

## Objectives
- Predict customer churn using machine learning.
- Compare multiple classification algorithms.
- Improve churn detection using threshold tuning.
- Build an interactive Streamlit dashboard.

## Machine Learning Algorithms
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Streamlit
- JupyterLab

## Model Performance
The initial model comparison showed Logistic Regression achieving the highest F1-score of approximately 60.20%.

After threshold tuning, its F1-score improved to approximately 62%, with recall of approximately 78%.

These are preliminary test-set results; final evaluation should use an untouched test set.

## Project Structure

```text
Customer-Churn-Prediction/
├── app.py
├── data/
├── models/
├── outputs/
├── notebooks/
│   └── Customer_Churn_Prediction.ipynb
├── README.md
└── requirements.txt
```

## How to Run

1. Clone this repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the Telco Customer Churn dataset from Kaggle and place the CSV file inside the data folder.

4. Run the dashboard:

```bash
streamlit run app.py
```

## Dataset
IBM Telco Customer Churn Dataset

Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## Author
Arati Patil

B.Tech Artificial Intelligence and Data Science
