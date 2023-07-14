import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

path = "cleaned/train.csv"
df = pd.read_csv(path)

df.describe()

# Remove Columns that will not be used for classification
d_col = ["ID", "Customer_ID", "Month", "Name", "SSN", "Monthly_Inhand_Salary"]

for _ in d_col:
    if _ in df.columns:
        df = df.drop(_, axis=1)

df.info()


# See Nominal values
for col in df:
    if df[col].dtypes == object:
        print(col)
        print("**" * 20)
        print(df[col].value_counts(dropna=False))
        print("**" * 20)


df["Credit_Score"]


# Conversion of Nominal data into Numeric
y_, label = pd.factorize(df["Credit_Score"])
df[df.select_dtypes(["object"]).columns] = df[
    df.select_dtypes(["object"]).columns
].apply(lambda x: pd.factorize(x)[0])

df.describe()

# finding Columns with Outliers using IQR method
def find_outliers(df, threshold=1.5):
    cols = []

    for _ in df.columns:
        q1 = np.percentile(df[_], 25)
        q3 = np.percentile(df[_], 75)
        iqr = q3 - q1
        lower_limit = q1 - threshold * iqr
        upper_limit = q3 + threshold * iqr

        if any((df[_] < lower_limit) | (df[_] > upper_limit)):
            cols.append(_)
    return cols


outlier_columns = find_outliers(df)
print(outlier_columns)

import matplotlib.pyplot as plt
import seaborn as sns

# Generate a color palette with a unique color for each box plot
num_plots = len(outlier_columns)
palette = sns.color_palette("PiYG", num_plots)

fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(10, 2 * num_plots))

for i, column in enumerate(outlier_columns):
    ax = axes[i]
    sns.boxplot(x=df[column], ax=ax, color=palette[i])
    ax.set_title(f"Box plot of {column}", fontsize=12)
    ax.set_ylabel("")
    ax.grid(True, axis="y")
plt.text(
    0.9,
    0.1,
    "Roll: 18, 25",
    ha="right",
    va="bottom",
    transform=plt.gca().transAxes,
    color="red",
    fontsize=24,
)
plt.tight_layout()
plt.savefig("outlier_box.png", dpi=300)
plt.show()


# Limit the Outliers to Upper limit and Lower Limit
threshold = 1.5
df2 = df.copy()
for col in outlier_columns:
    q1 = np.percentile(df[col], 25)
    q3 = np.percentile(df[col], 75)
    iqr = q3 - q1
    lower_limit = q1 - threshold * iqr
    upper_limit = q3 + threshold * iqr

    df2[col] = np.where(
        df[col] > upper_limit,
        upper_limit,
        np.where(df[col] < lower_limit, lower_limit, df[col]),
    )

"""for _ in outlier_columns:
    Q1 = df[_].quantile(0.25)
    Q3 = df[_].quantile(0.75)
    IQR = Q3 - Q1
    df = df.drop(df.loc[df[_] > (Q3 + 1.5 * IQR)].index)
    df = df.drop(df.loc[df[_] < (Q1 - 1.5 * IQR)].index)
df.info()"""

df["Annual_Income"]

# Box plot after handeling outliers
fig, axes = plt.subplots(
    nrows=len(outlier_columns), ncols=1, figsize=(10, 2.5 * len(outlier_columns))
)

for i, column in enumerate(outlier_columns):
    ax = axes[i]
    sns.boxplot(x=df2[column], ax=ax)
    ax.set_xlabel("Index", fontsize=12)
    ax.set_ylabel(column, fontsize=12)
    ax.set_title(f"Box plot of {column}", fontsize=14)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Set the color palette
palette = sns.color_palette("PiYG", 14)

fig, axes = plt.subplots(2, 1, figsize=(6, 10))

# Plot "Before" distribution
sns.histplot(df["Annual_Income"], kde=True, ax=axes[0], color=palette[0], alpha=0.5)
axes[0].set_title("Before")

# Plot "After" distribution
sns.histplot(df2["Annual_Income"], kde=True, ax=axes[1], color=palette[0], alpha=0.5)
axes[1].set_title("After")

# Adjust alpha value for plot elements
for ax in axes:
    ax.set_facecolor((1, 1, 1, 1))  # Set background alpha value
    ax.grid(alpha=0.2)  # Adjust gridlines alpha value
plt.text(
    0.9,
    0.1,
    "Roll: 18, 25",
    ha="right",
    va="bottom",
    transform=plt.gca().transAxes,
    color="red",
    fontsize=14,
)
plt.tight_layout()
plt.savefig("limit.png", dpi=300)
plt.show()

corr = df.corr()

plt.figure(figsize=(20, 20))
matrix = np.triu(corr)
sns.heatmap(corr, cmap="PiYG", annot=True, mask=matrix)
plt.tight_layout()
plt.text(
    0.9,
    0.9,
    "Roll: 18, 25",
    ha="right",
    va="top",
    transform=plt.gca().transAxes,
    color="red",
    fontsize=34,
)
plt.savefig("matrix.png", dpi=300)
plt.show()


# Training Data
y = df["Credit_Score"]
X = df.drop("Credit_Score", axis=1)

from sklearn import tree

clf = tree.DecisionTreeClassifier(criterion="entropy")


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=100
)


clf = clf.fit(X_train, y_train)

predicted = clf.predict(X_test)
pred_label = label[predicted]
y_label = label[y_test]

print(pred_label)


print(y_label)

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    classification_report,
)

conf_mat = confusion_matrix(y_label, pred_label)
C = conf_mat / conf_mat.astype(np.float).sum(axis=1)
disp = ConfusionMatrixDisplay(confusion_matrix=C, display_labels=label)
fig, ax = plt.subplots(figsize=(8, 6))

# Use only the green color from the "PiYG" palette
cmap = plt.cm.get_cmap("PiYG")
cmap = cmap(np.linspace(0.5, 1, cmap.N))
cmap = cmap[:, 1:2]
cmap = plt.cm.colors.ListedColormap(cmap)

disp.plot(ax=ax, cmap="Greens", xticks_rotation="vertical")

plt.title("Confusion Matrix")
plt.tight_layout()
plt.text(
    0.9,
    0.1,
    "Roll: 18, 25",
    ha="right",
    va="bottom",
    transform=plt.gca().transAxes,
    color="red",
    fontsize=18,
)
plt.savefig("entropy.png", dpi=300)


print(classification_report(y_test, predicted))


clf_gini = tree.DecisionTreeClassifier(criterion="gini", random_state=0)


# fit the model
clf_gini.fit(X_train, y_train)


y_pred_gini = clf_gini.predict(X_test)



conf_mat = confusion_matrix(y_test, y_pred_gini)
C = conf_mat / conf_mat.astype(np.float).sum(axis=1)
disp = ConfusionMatrixDisplay(confusion_matrix=C, display_labels=label)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap="Greens", xticks_rotation="vertical")

plt.title("Confusion Matrix")
plt.tight_layout()
plt.text(
    0.9,
    0.1,
    "Roll: 18, 25",
    ha="right",
    va="bottom",
    transform=plt.gca().transAxes,
    color="red",
    fontsize=18,
)
plt.savefig("entropy2.png", dpi=300)


print(classification_report(y_test, y_pred_gini))
