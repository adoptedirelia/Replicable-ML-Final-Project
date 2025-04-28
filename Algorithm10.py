import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import zero_one_loss
from sklearn.utils import shuffle
import random
import config
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tqdm


def random_drawsample(X_train, y_train, sample_size, random_state=42):
    """
    X_train, y_train: full training set
    """
    np.random.seed(random_state)
    df = pd.concat([X_train, y_train], axis=1)  
    # random sample a subset of sample_size samples from the full dataset
    df_shuffled = df.sample(
        frac=1, random_state=random_state).reset_index(drop=True)
    temp = df_shuffled.iloc[:sample_size]

    X_train_shuffled = temp.iloc[:, :-1]
    y_train_shuffled = temp.iloc[:, -1]
    
    return X_train_shuffled, y_train_shuffled

def build_candidate_trees(X_train, y_train,sample_size, max_depth=3, num_trees=20, random_state=42):

    """
    X_train, y_train: full training set
    """
    np.random.seed(random_state)
    H = []
    errors = []
    for i in tqdm.tqdm(range(num_trees)):

        X_train_shuffled, y_train_shuffled = random_drawsample(
            X_train, y_train, sample_size, random_state=random_state+i)
        
        tree = DecisionTreeClassifier(
            max_depth=max_depth, random_state=random_state, criterion='gini')
        tree.fit(X_train_shuffled,  y_train_shuffled)
        error = empirical_error(tree, X_train_shuffled,  y_train_shuffled)
        
        H.append(tree)
        errors.append(error)
    return H


def empirical_error(tree, X, y):

    y_pred = tree.predict(X)
    return zero_one_loss(y, y_pred)


def replicable_learner(X_train, y_train, H,sample_size, random_seed=1234):

    random.seed(random_seed)
    np.random.seed(random_seed)

    # randomly draw a labeled sample from the full dataset
    X_train_shuffled, y_train_shuffled = random_drawsample(
    X_train, y_train, sample_size, random_state=random_seed)
    
    errors = {tree: empirical_error(tree, X_train_shuffled, y_train_shuffled) for tree in H}
    opt = min(errors.values())
    print("OPT error",opt,"errors",errors.values())
    
    #opt = 0  # the optimal error we can achieve is zero
    v_init = np.random.uniform(opt,opt + config.tau / 2)
    
    k = int(((config.alpha/4 - config.tau/2)-1.5*config.tau)/config.tau) + 1
    v_candidates = [v_init + (2 * i + 1) * config.tau / 2 for i in range(k)]
    # print("k:", k, "v max candidates:", np.max(v_candidates),
    #       "v min candidates:", np.min(v_candidates))
    # print(v_candidates)
    v = random.choice(v_candidates)
    print("v:", v)
    H_shuffled = shuffle(H, random_state=random_seed)
    res_trees = []
    for tree in H_shuffled:
        if errors[tree] <= v:
            res_trees.append(tree)
    return res_trees

    return min(errors.items(), key=lambda x: x[1])[0]


def preProcess(df):
    """
    Categories the columns of the dataset.
    """
    result = df.copy()
    for col in result.columns[:]:
        if result[col].dtype == 'object':
            result[col] = LabelEncoder().fit_transform(result[col])
    return result


def load_dataset(dataset_path, sample_size=None, test_size=0.2, random_state=42):
    df = pd.read_csv(dataset_path)

    # For ease of implementation, we will only use a few features
    df = df[config.selected_features]

    # Preprocess the dataset
    df_final = preProcess(df)
    # train-test split
    X = df_final.drop('satisfaction', axis=1)
    y = df_final['satisfaction']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=random_state, test_size=test_size)
    if sample_size is not None:
        X_train, y_train = shuffle(X_train, y_train, random_state=random_state)
        X_train = X_train[:sample_size]
        y_train = y_train[:sample_size]
    print("Train set size:", X_train.shape[0],
          "Test set size:", X_test.shape[0])

    return X_train, X_test, y_train, y_test

def load_full_dataset(dataset_path, random_state=42):
    df = pd.read_csv(dataset_path)

    # For ease of implementation, we will only use a few features
    df = df[config.selected_features]

    # Preprocess the dataset
    df_final = preProcess(df)
    # train-test split
    X = df_final.drop('satisfaction', axis=1)
    y = df_final['satisfaction']
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, random_state=random_state, test_size=test_size)
    X, y = shuffle(X, y, random_state=random_state)
    return X, y

if __name__ == '__main__':

    X_train, X_test, y_train, y_test = load_dataset(
        config.dataset_path, test_size=0.2, random_state=config.random_seed)

    H = build_candidate_trees(X_train, y_train, max_depth=config.max_depth,
                              num_trees=config.num_H, random_state=config.random_seed)
    tree = replicable_learner(
        X_train, y_train, H, random_seed=config.random_seed)
    a = tree.score(X_test, y_test)
    print(a)
    for t in H:
        r = export_text(t, feature_names=config.selected_features[:-1])
        print(r)
