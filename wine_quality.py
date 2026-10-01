import pandas as pd
df = pd.read_csv("/Users/vikramfakke/Downloads/TY3/ML mini project Final/wine_data.csv")
print(df.head())
print(df.shape)
print(df.columns)
print(df.isnull().sum())
print("Duplicates:", df.duplicated().sum())
df = df.drop_duplicates()

print("New shape:", df.shape)
df = df.fillna(
    df.median(numeric_only=True)
)

print(df.isnull().sum())
df["quality_class"] = df["quality"].apply(
    lambda x: 1 if x >= 7 else 0
)

print(df[["quality", "quality_class"]].head())
print(df["quality_class"].value_counts())
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(
    x="quality",
    data=df
)

plt.title("Wine Quality Distribution")
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True
)

plt.title("Correlation Heatmap")
plt.show()
X = df.drop(
    ["quality", "quality_class"],
    axis=1
)

y = df["quality_class"]

print("Features:")
print(X.columns)

print("\nTarget:")
print(y.head())
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

select = SelectKBest(
    f_classif,
    k=8
)

X_train = select.fit_transform(
    X_train,
    y_train
)

X_test = select.transform(
    X_test
)

print("Selected features:", X_train.shape[1])
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_test = scaler.transform(
    X_test
)

print("Scaling completed")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model1 = LogisticRegression(
    max_iter=1000
)

model1.fit(
    X_train,
    y_train
)

p1 = model1.predict(
    X_test
)

a1 = accuracy_score(
    y_test,
    p1
)

print("Logistic Regression:", a1 * 100, "%")
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

model2 = KNeighborsClassifier(
    n_neighbors=5
)

model2.fit(
    X_train,
    y_train
)

p2 = model2.predict(
    X_test
)

a2 = accuracy_score(
    y_test,
    p2
)

print("KNN:", a2 * 100, "%")
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

model2 = KNeighborsClassifier(
    n_neighbors=5
)

model2.fit(
    X_train,
    y_train
)

p2 = model2.predict(
    X_test
)

a2 = accuracy_score(
    y_test,
    p2
)

print("KNN:", a2 * 100, "%")
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

model3 = DecisionTreeClassifier(
    random_state=42
)

model3.fit(
    X_train,
    y_train
)

p3 = model3.predict(
    X_test
)

a3 = accuracy_score(
    y_test,
    p3
)

print("Decision Tree:", a3 * 100, "%")
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model4 = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model4.fit(
    X_train,
    y_train
)

p4 = model4.predict(
    X_test
)

a4 = accuracy_score(
    y_test,
    p4
)

print("Random Forest:", a4 * 100, "%")
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

model5 = GaussianNB()

model5.fit(
    X_train,
    y_train
)

p5 = model5.predict(
    X_test
)

a5 = accuracy_score(
    y_test,
    p5
)

print("Naive Bayes:", a5 * 100, "%")
import pandas as pd

results = pd.DataFrame({
    "Algorithm": [
        "Logistic Regression",
        "KNN",
        "Decision Tree",
        "Random Forest",
        "Naive Bayes"
    ],

    "Accuracy": [
        a1 * 100,
        a2 * 100,
        a3 * 100,
        a4 * 100,
        a5 * 100
    ]
})
best = results.loc[
    results["Accuracy"].idxmax()
]

print("Best Algorithm:", best["Algorithm"])
print("Best Accuracy:", best["Accuracy"], "%")

models = {
    "Logistic Regression": model1,
    "KNN": model2,
    "Decision Tree": model3,
    "Random Forest": model4,
    "Naive Bayes": model5
}

best_model = models[
    best["Algorithm"]
]

prediction = best_model.predict(X_test)

print("Final Model:", best["Algorithm"])

print("Best Algorithm:", best["Algorithm"])
print("Best Accuracy:", best["Accuracy"], "%")
import matplotlib.pyplot as plt
import seaborn as sns

sns.barplot(
    x="Algorithm",
    y="Accuracy",
    data=results
)

plt.title("Algorithm Accuracy Comparison")
plt.xlabel("Algorithm")
plt.ylabel("Accuracy (%)")

plt.xticks(rotation=20)

plt.show()

best_model = models[
    best["Algorithm"]
]

prediction = best_model.predict(
    X_test
)

print("Final Model:", best["Algorithm"])
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

accuracy = accuracy_score(
    y_test,
    prediction
)

precision = precision_score(
    y_test,
    prediction
)

recall = recall_score(
    y_test,
    prediction
)

f1 = f1_score(
    y_test,
    prediction
)

print("Accuracy:", accuracy * 100, "%")
print("Precision:", precision * 100, "%")
print("Recall:", recall * 100, "%")
print("F1 Score:", f1 * 100, "%")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    y_test,
    prediction
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()
