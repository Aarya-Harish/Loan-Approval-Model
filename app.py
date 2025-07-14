import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Loan Approval App", layout="wide")
st.title("🏦 Loan Approval Prediction")

# 1. Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("data/processedloan.csv")
    return df

data = load_data()
st.subheader("📄 Data Preview")
st.dataframe(data.head())

# 2. Exploratory Charts
# st.subheader("📊 Visual Insights")

# # Gender-wise Loan Approval
# gender_status = data.groupby(['Gender', 'Loan_Status']).size().unstack()
# fig1, ax1 = plt.subplots()
# gender_status.plot(kind='pie', subplots=True, autopct='%1.1f%%', figsize=(10, 5), ax=ax1)
# st.pyplot(fig1)

# Applicant Income Group vs Loan_Status
st.markdown("#### Applicant Income vs Loan Approval")
data['IncomeGroup'] = pd.cut(data['ApplicantIncome'], bins=[0, 2500, 4000, 6000, 10000, 20000, 50000],
                             labels=["0-2.5k", "2.5k-4k", "4k-6k", "6k-10k", "10k-20k", "20k+"])
income_vs_status = pd.crosstab(data['IncomeGroup'], data['Loan_Status'])
fig2 = income_vs_status.plot(kind='bar', stacked=True, colormap='Set2', figsize=(10, 5))
plt.xlabel("Applicant Income Group")
plt.ylabel("Count")
plt.title("Loan Status by Applicant Income Group")
st.pyplot(fig2.get_figure())

# 3. Feature Engineering
data['LoanAmount_log'] = np.log1p(data['LoanAmount'])
data['ApplicantIncome_log'] = np.log1p(data['ApplicantIncome'])
data.drop(['LoanAmount', 'ApplicantIncome', 'IncomeGroup'], axis=1, inplace=True)

# Encode categorical
X = pd.get_dummies(data.drop(columns='Loan_Status'), drop_first=True)
y = data['Loan_Status'].map({'Y': 1, 'N': 0})

# 4. Model Training with Tuning
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params = {
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid = GridSearchCV(DecisionTreeClassifier(random_state=1), params, cv=5, scoring='accuracy')
grid.fit(x_train, y_train)
best_tree = grid.best_estimator_

# 5. Model Evaluation
y_pred = best_tree.predict(x_test)
acc = accuracy_score(y_test, y_pred)
cv_acc = cross_val_score(best_tree, X, y, cv=5).mean()

st.subheader("📈 Model Performance")
st.success(f"✅ Decision Tree Accuracy: {acc*100}")
# st.info(f"🔁 Cross-Validation Accuracy: {cv_acc:.4f}")
# st.write(f"Best Params: {grid.best_params_}")

# 6. Tree Plot 
st.subheader("🌳 Top Levels of Decision Tree")
fig3, ax3 = plt.subplots(figsize=(14, 6))
plot_tree(best_tree, feature_names=X.columns, class_names=["No", "Yes"],
          filled=True, max_depth=3, fontsize=10, ax=ax3)
st.pyplot(fig3)

# 7. Prediction UI
st.subheader("🧮 Predict Loan Approval")
with st.form("predict_form"):
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Married = st.selectbox("Married", ["Yes", "No"])
    Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
    ApplicantIncome = st.number_input("Applicant Income", min_value=0)
    LoanAmount = st.number_input("Loan Amount (in thousands)", min_value=0)
    Loan_Amount_Term = st.number_input("Loan Amount Term (months)", min_value=0)
    Credit_History = st.selectbox("Credit History", [1.0, 0.0])
    Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    submit = st.form_submit_button("Predict")

if submit:
    test_df = {
        'Loan_Amount_Term': Loan_Amount_Term,
        'Credit_History': Credit_History,
        'LoanAmount_log': np.log1p(LoanAmount),
        'ApplicantIncome_log': np.log1p(ApplicantIncome),
        'Gender_Male': 1 if Gender == "Male" else 0,
        'Married_Yes': 1 if Married == "Yes" else 0,
        'Education_Not Graduate': 1 if Education == "Not Graduate" else 0,
        'Self_Employed_Yes': 1 if Self_Employed == "Yes" else 0,
        'Property_Area_Semiurban': 1 if Property_Area == "Semiurban" else 0,
        'Property_Area_Urban': 1 if Property_Area == "Urban" else 0,
    }

    test_df = pd.DataFrame([test_df])
    for col in X.columns:
        if col not in test_df.columns:
            test_df[col] = 0
    test_df = test_df[X.columns]  # maintain order

    result = best_tree.predict(test_df)[0]
    st.success("🎉 Loan Approved!" if result == 1 else "❌ Loan Not Approved.")
