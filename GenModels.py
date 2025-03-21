"""
Generate a set of decision tree models.
"""
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
import config
from utils import setup_seed

setup_seed(config.random_seed)
num_samples = config.m
num_features = len(config.selected_features)-1
def get_one_model():
    tree = DecisionTreeClassifier()

    X_dummy = np.random.rand(num_samples, num_features)  
    y_dummy = np.random.randint(0, 2, num_samples)  # binary classification
    tree.fit(X_dummy, y_dummy)

    # un less we want to manually modify the tree
    #tree.tree_.threshold = np.random.uniform(-1, 1, size=tree.tree_.threshold.shape)
    #tree.tree_.value = np.random.rand(*tree.tree_.value.shape)

    return tree
def gen_model_sets(model_num):
    for i in range(model_num):
        tree = get_one_model()
        # save model
        joblib.dump(tree, config.model_path+str(i)+".pkl")
        # load: loaded_tree = joblib.load("decision_tree_model.pkl")
if __name__=="__main__":
    gen_model_sets(config.num_H)