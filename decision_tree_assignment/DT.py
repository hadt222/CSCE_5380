import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load dataset (assuming "student-mat.csv" is downloaded from UCI)
df = pd.read_csv("student-mat.csv", sep=";")

# Create Pass/Fail target
df["pass"] = df["G3"] >= 10
y = df["pass"]
# Drop grades (G1, G2, G3) to avoid leakage
X = df.drop(["G1", "G2", "G3", "pass"], axis=1)


# Convert categorical to numeric (one-hot encoding)
X = pd.get_dummies(X, drop_first=True)

# Train decision tree
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = DecisionTreeClassifier(max_depth=4, criterion="entropy", random_state=42)
clf.fit(X_train, y_train)

# Visualize tree
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=X.columns, class_names=["Fail","Pass"], filled=True)
plt.show()

# Accuracy
print("Accuracy:", clf.score(X_test, y_test))
