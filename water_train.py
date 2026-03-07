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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# -----------------------------
# Set Experiment Name
# -----------------------------
mlflow.set_experiment('water_exp2_gbdt')

# -----------------------------
# Set Tracking uri to log artifact
# -----------------------------
mlflow.set_tracking_uri('http://127.0.0.1:5000')

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

with mlflow.start_run():

    clf = GradientBoostingClassifier(
        n_estimators=n_estimators,
        random_state=42
    )

    clf.fit(X_train, y_train)


    # -----------------------------
    # Save Model
    # -----------------------------
    with open("model_gb.pkl", "wb") as f:
        pickle.dump(clf, f)


    # -----------------------------
    # Load Model
    # -----------------------------
    with open("model_gb.pkl", "rb") as f:
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


    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm,annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion matrics')

    plt.savefig('confusion_matrix.png')

    mlflow.log_artifact('confusion_matrix.png')

    # to log and track model
    mlflow.sklearn.log_model(sk_model=clf, artifact_path="GradientBoostingClassifier")

    # to log and track code
    mlflow.log_artifact(__file__)

    # add tags
    mlflow.set_tag('author','Pritesh')
    mlflow.set_tag('model','GB')


