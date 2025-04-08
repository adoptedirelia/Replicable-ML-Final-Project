"""
Code for using the classic decision tree algorithm. Keep all the hyperparameters as default.
The dataset used is the 'Invistico_Airline.csv' dataset from https://www.kaggle.com/datasets/yakhyojon/customer-satisfaction-in-airline/data
We would binary classify whether a customer will be satisfied or not based on a list of selected features.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import config

def plot_decision_tree(model, feature_names, class_names):
    tree = plot_tree(model, feature_names = feature_names, class_names = class_names,
                     rounded = True, proportion = True, precision = 2, filled = True, fontsize=10)
    
    return tree
def preProcess(df):
    """
    Categories the columns of the dataset.
    """
    result = df
    for col in result.columns[:-1]:
        result[col] = LabelEncoder().fit_transform(result[col])
    return result

def main(dataset_path,seed):
    
    ### Load the dataset
    df = pd.read_csv(dataset_path)
    
    ### For ease of implementation, we will only use a few features
    df = df[config.selected_features]

    ### Preprocess the dataset
    df_final = preProcess(df)
    ### train-test split
    X = df_final.drop('satisfaction', axis=1)
    y = df_final['satisfaction']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    print("Train set size:", X_train.shape[0], "Test set size:", X_test.shape[0])
    
    ### Define and train a tree model
    clf = DecisionTreeClassifier(max_depth=3,criterion='gini',random_state=seed)
    clf = clf.fit(X_train, y_train)

    # Prediction
    y_pred = clf.predict(X_test)

    # Evaluation
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:", conf_matrix)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", np.round(acc,2)) # baseline acc=0.82
    print("Classification Report:", classification_report(y_test, y_pred))

    plt.figure(figsize=(30,10))
    plot_decision_tree(clf, X.columns, clf.classes_)
    plt.show()
    

if __name__=="__main__":
    main(config.dataset_path,config.random_seed)