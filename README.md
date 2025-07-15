# 🏦 Loan Approval Prediction Web App

A **Machine Learning** and **Streamlit** powered web application that predicts whether a loan will be approved or not based on applicant information.

This project uses the dataset from Kaggle:  
🔗 [Loan Approval Prediction Data – by Armanjit Singh](https://www.kaggle.com/datasets/armanjitsingh/loan-approval-prediction-data)


---  

## 🚀 Features

- ✅ Predicts loan approval with **Decision Tree Classifier**
- 🔍 Performs data cleaning and log transformation
- 🎯 Optimized using **GridSearchCV** and **Cross-Validation**
- 📊 Visual insights via:
  - **Pie chart**: Loan approval distribution by gender
  - **Bar chart**: Applicant income groups vs. loan approval
  - **Tree Diagram**: Top 3 levels of decision tree structure
- 🧾 Displays model accuracy and evaluation
- 📝 Accepts real-time **user input** for prediction via Streamlit interface

---

## 📁 Dataset Overview

| Column             | Description                                |
|-------------------|--------------------------------------------|
| Gender            | Male / Female                              |
| Married           | Marital status                             |
| Dependents        | Number of dependents                       |
| Education         | Graduate / Not Graduate                    |
| Self_Employed     | Employment type                            |
| ApplicantIncome   | Income of applicant                        |
| CoapplicantIncome | Income of co-applicant                     |
| LoanAmount        | Loan amount (in thousands)                 |
| Loan_Amount_Term  | Loan term (in months)                      |
| Credit_History    | 1 = Good credit history, 0 = Poor history  |
| Property_Area     | Rural / Semiurban / Urban                  |
| Loan_Status       | Target label (Y = Approved, N = Not Approved) |

---

## 📊 Visualizations

1. **Pie Chart** – Gender vs. Loan Approval Status  
2. **Bar Chart** – Applicant Income Group vs. Loan Status  
3. **Decision Tree** – Top 3 levels plotted using `sklearn.tree.plot_tree`

---

Loan-Approval-ML-Model/
├── app.py                     # Main Streamlit app
├── processeddataforloan.csv   # Cleaned Kaggle dataset
├── requirements.txt           # Required libraries
└── README.md                  # Project documentation



## 🧠 Model Details

- Algorithm: `DecisionTreeClassifier`
- Optimizations:
  - Hyperparameter tuning with `GridSearchCV`
  - `cross_val_score` for K-fold accuracy
- Accuracy: **~75–80%**


  Deployed Model Link - https://loan-approval-model-sharish.streamlit.app/


