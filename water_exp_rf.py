#import warnings
#warnings.filterwarnings("ignore", category=FutureWarning)


import pandas as pd
import numpy as np
import pickle
import yaml
import mlflow
import mlflow.sklearn


import seaborn as sns
import matplotlib.pyplot as plt 

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import dagshub
dagshub.init(repo_owner='pritesh13590', repo_name='water-potability-dagshub', mlflow=True)

# -----------------------------
# Set Experiment Name
# -----------------------------
mlflow.set_experiment('water_exp_rf')

# -----------------------------
# Set Tracking uri to log artifact
# -----------------------------
mlflow.set_tracking_uri('https://dagshub.com/pritesh13590/water-potability-dagshub.mlflow')

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("c:/Data/project_datasets/water_potability.csv")


# -----------------------------
# Train Test Split
# -----------------------------
train_data, test_data = train_test_split(
    data,
    test_size=0.20,
    random_state=42
)


# -----------------------------
# Function to Fill Missing Values
# -----------------------------
def fill_missing_with_median(df):
    df = df.copy()
    df.fillna(df.median(), inplace=True)
    return df


# -----------------------------
# Data Preprocessing
# -----------------------------
train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)


# -----------------------------
# Feature Target Split
# -----------------------------
X_train = train_processed_data.iloc[:, :-1].values
y_train = train_processed_data.iloc[:, -1].values

X_test = test_processed_data.iloc[:, :-1].values
y_test = test_processed_data.iloc[:, -1].values


# -----------------------------
# Model Training
# -----------------------------
n_estimators = 500
max_depth = 10

with mlflow.start_run():

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth = max_depth,
        random_state=42
    )

    clf.fit(X_train, y_train)


    # -----------------------------
    # Save Model
    # -----------------------------
    with open("model_rf.pkl", "wb") as f:
        pickle.dump(clf, f)


    # -----------------------------
    # Load Model
    # -----------------------------
    with open("model_rf.pkl", "rb") as f:
        model = pickle.load(f)


    # -----------------------------
    # Prediction
    # -----------------------------
    y_pred = model.predict(X_test)


    # -----------------------------
    # Metrics
    # -----------------------------
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)


    # -----------------------------
    # Print Results
    # -----------------------------
    print("Accuracy:", acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)


    # -----------------------------
    # Log Metrics using mlflow
    # -----------------------------
   
    mlflow.log_metric("Accuracy", acc)
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("Recall", recall)
    mlflow.log_metric("F1 Score", f1)

    mlflow.log_param('n_estimators',n_estimators)
    mlflow.log_param('max_depth',max_depth)


    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm,annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion matrics')

    plt.savefig('confusion_matrix.png')

    mlflow.log_artifact('confusion_matrix.png')

    # to log and track model
    mlflow.sklearn.log_model(sk_model=clf, name="RandomForestClassifier")

    # to log and track code
    mlflow.log_artifact(__file__)

    # add tags
    mlflow.set_tag('author','Pritesh_kumar')
    mlflow.set_tag('model','RF')


