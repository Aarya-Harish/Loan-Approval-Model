# decision_tree_classifier.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

print("🚀 Decision Tree Classifier Started...")

# 1. Load the dataset
data = pd.read_csv("processeddataforloan.csv")

# 2. Split features (X) and target (y)
X = data.drop(columns='Loan_Status')
y = data['Loan_Status']

# 3. Encode categorical features in X
X = pd.get_dummies(X, drop_first=True)

# 4. Encode target if it's 'Y'/'N'
if y.dtype == 'object':
    y = y.map({'Y': 1, 'N': 0})

# 5. Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# 6. Train the model
clf = DecisionTreeClassifier(random_state=1)
clf.fit(x_train, y_train)

# 7. Predict and evaluate
y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy score of Decision Tree is: {accuracy:.4f}")

# 8. Display the top 3 levels of the decision tree
plt.figure(figsize=(14, 8))  # Size for readability
plot_tree( 
    clf,
    feature_names=X.columns,
    class_names=["Not Approved", "Approved"],
    filled=True,
    rounded=True,
    max_depth=3,        # 🔁 Only show top 3 tiers
    fontsize=10
)
plt.title("📊 Top 3 Levels of Decision Tree", fontsize=16)
plt.tight_layout()
plt.show()  # 👈 Shows the plot window