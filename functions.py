import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, classification_report, roc_auc_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import torch.optim as optim

# Read and prepare the data
def read_and_clean_data(data_path: str) -> pd.DataFrame:
    # Load the dataset
    data = pd.read_csv(data_path)

    # Drop unnecessary columns
    data_cleaned = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    data_cleaned['Age'] = imputer.fit_transform(data_cleaned[['Age']])

    imputer = SimpleImputer(strategy='most_frequent')
    data_cleaned['Embarked'] = imputer.fit_transform(data_cleaned[['Embarked']]).ravel()

    # Convert categorical columns to numerical
    data_cleaned['Sex'] = data_cleaned['Sex'].map({'male': 0, 'female': 1})

    # One-hot encoding for Embarked
    data_cleaned = pd.get_dummies(data_cleaned, columns=['Embarked'], drop_first=True)
    
    return data_cleaned

def evaluate_classifiers(X_train, X_test, y_train, y_test):
    classifiers = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(probability=True),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }
    
    results = []
    
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]) if hasattr(clf, "predict_proba") else None
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        results.append({
            "classifier": clf,
            "name": name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "report": report,
            "conf_matrix": conf_matrix
        })
        
    return results

def plot_roc_curves(results, X_test, y_test, title):
    plt.figure(figsize=(12, 10))
    for res in results:
        if res["roc_auc"]:
            fpr, tpr, _ = roc_curve(y_test, res["classifier"].predict_proba(X_test)[:, 1])
            plt.plot(fpr, tpr, lw=2, label=f'{res["name"]} (area = {res["roc_auc"]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0, 1])
    plt.ylim([-0, 1])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(f'ROC Curve - {title}', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.show()

def print_confusion_matrices(results, title, samples):
    fig, axes = plt.subplots(1, len(results), figsize=(20, 5))
    fig.suptitle(f'Confusion Matrices - {title}', fontsize=16)
    for idx, res in enumerate(results):
        sns.heatmap(res['conf_matrix'], annot=True, fmt='d', cmap="Blues", cbar=False, ax=axes[idx])
        axes[idx].set_title(f'{res["name"]} ({samples["original"]} + {samples["generated"]} samples)')
        axes[idx].set_xlabel('Predicted labels')
        axes[idx].set_ylabel('True labels')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def evaluate_decision_tree(X_train, y_train):
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    return clf

def plot_decision_tree(clf, feature_names, title):
    plt.figure(figsize=(20, 10))
    plot_tree(clf, feature_names=feature_names, class_names=True, filled=True)
    plt.title(title)
    plt.show()

def plot_metric_table(results, metric):
    data = {
        "Classifier": [res["name"] for res in results],
        metric: [res[metric] for res in results]
    }
    df = pd.DataFrame(data)
    plt.figure(figsize=(8, 6))
    sns.barplot(x="Classifier", y=metric, data=df)
    plt.title(f'{metric.capitalize()} Scores of Different Classifiers')
    plt.ylim(0, 1.0)
    plt.show()

def plot_accuracy_comparison(results_all):
    accuracies = []
    for percent, results in results_all:
        for res in results:
            accuracies.append({
                "percent": percent,
                "classifier": res["name"],
                "accuracy": res["accuracy"]
            })
    df = pd.DataFrame(accuracies)
    plt.figure(figsize=(12, 8))
    sns.lineplot(x="percent", y="accuracy", hue="classifier", data=df, marker="o")
    plt.title('Accuracy Comparison of Different Classifiers with Varying GAN Data')
    plt.xlabel('Percentage of Synthetic Data')
    plt.ylabel('Accuracy')
    plt.legend(title="Classifier")
    plt.show()

def plot_combined_metrics_table(results_all):
    combined_data = []
    for percent, results in results_all:
        for res in results:
            combined_data.append({
                "Classifier": res["name"],
                "Percentage of GAN Data": percent,
                "Accuracy": res["accuracy"],
                "Precision": res["precision"],
                "Recall": res["recall"],
                "F1 Score": res["f1"]
            })
    df = pd.DataFrame(combined_data)
    plt.figure(figsize=(12, 8))
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    for metric in metrics:
        pivot_df = df.pivot(index="Classifier", columns="Percentage of GAN Data", values=metric)
        sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="YlGnBu")
        plt.title(f'{metric} Scores for Classifiers with Different GAN Data Percentages')
        plt.show()