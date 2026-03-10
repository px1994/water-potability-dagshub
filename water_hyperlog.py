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
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import dagshub
dagshub.init(repo_owner='pritesh13590', repo_name='water-potability-dagshub', mlflow=True)

# -----------------------------
# Set Experiment Name
# -----------------------------
mlflow.set_experiment('water_exp_hyper_rf')

# -----------------------------
# Set Tracking URI
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
with mlflow.start_run(run_name="RandomForest_Hyperparameter_Tuning"):

    clf = RandomForestClassifier(random_state=42)

    params = {
        'n_estimators':[100,200,300,500,1000],
        'max_depth':[None,10,20,30,40]
    }

    search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=params,
        n_iter=50,
        cv=5,
        n_jobs=-1,
        verbose=2,
        random_state=42
    )

    search.fit(X_train, y_train)

    # -----------------------------
    # Best Model
    # -----------------------------
    best_model = search.best_estimator_

    # -----------------------------
    # Print Best Parameters
    # -----------------------------
    print("Best Hyperparameters Found:")
    print(search.best_params_)

    # -----------------------------
    # Log Best Parameters
    # -----------------------------
    mlflow.log_params(search.best_params_)

    # -----------------------------
    # Log All Hyperparameter Combinations
    # -----------------------------
    for i in range(len(search.cv_results_['params'])):

        params = search.cv_results_['params'][i]
        score = search.cv_results_['mean_test_score'][i]

        with mlflow.start_run(run_name=f"Combination_{i+1}", nested=True):

            mlflow.log_params(params)
            mlflow.log_metric("mean_cv_score", score)

    # -----------------------------
    # Save Model
    # -----------------------------
    with open("model_rf_hyp.pkl", "wb") as f:
        pickle.dump(best_model, f)

    # -----------------------------
    # Load Model
    # -----------------------------
    with open("model_rf_hyp.pkl", "rb") as f:
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

    print("Accuracy:", acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # -----------------------------
    # Log Metrics
    # -----------------------------
    mlflow.log_metric("Accuracy", acc)
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("Recall", recall)
    mlflow.log_metric("F1 Score", f1)

    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    plt.savefig('confusion_matrix.png')

    mlflow.log_artifact('confusion_matrix.png')

    # -----------------------------
    # Log Model
    # -----------------------------
    mlflow.sklearn.log_model(
        sk_model=best_model,
        name="RandomForestClassifier"
    )

    # -----------------------------
    # Log Code
    # -----------------------------
    mlflow.log_artifact(__file__)

    # -----------------------------
    # Add Tags
    # -----------------------------
    mlflow.set_tag('author','Pritesh_kumar')
    mlflow.set_tag('model','RandomForest')

    # -----------------------------
    # Log Dataset
    # -----------------------------
    train_df = mlflow.data.from_pandas(train_processed_data)
    test_df = mlflow.data.from_pandas(test_processed_data)

    mlflow.log_input(train_df,"train")
    mlflow.log_input(test_df,"test")