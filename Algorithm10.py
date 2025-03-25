import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss
from sklearn.utils import shuffle
import random
import config
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tqdm

def build_candidate_trees(X_train, y_train, max_depth=3, num_trees=20, random_state=42):

    np.random.seed(random_state)
    H = []

    for i in tqdm.tqdm(range(num_trees)):
        df = pd.concat([X_train, y_train], axis=1)  # 合并成一个 DataFrame
        df_shuffled = df.sample(frac=1, random_state=random_state+i).reset_index(drop=True)
        temp = df_shuffled.iloc[:int(len(df_shuffled) * 0.7)]

        X_train_shuffled = temp.iloc[:, :-1]
        y_train_shuffled = temp.iloc[:, -1]

        tree = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state + i,criterion='gini')
        tree.fit(X_train_shuffled, y_train_shuffled)
        # tree.fit(X_train, y_train)
        H.append(tree)

    return H


def empirical_error(tree, X, y):

    y_pred = tree.predict(X)
    return zero_one_loss(y, y_pred)

def replicable_learner(X_train, y_train, H, random_seed=1234):

    random.seed(random_seed)
    np.random.seed(random_seed)

    errors = {tree: empirical_error(tree, X_train, y_train) for tree in H}

    opt = min(errors.values())
    v_init = opt + config.tau / 2
    k = int(((config.alpha/4 - config.tau/2)*2-1)/2) + 1
    v_candidates = [v_init + (2 * i + 1) * config.tau / 2 for i in range(k)]
    print(v_candidates)
    v = random.choice(v_candidates)  

    H_shuffled = shuffle(H, random_state=random_seed)
    for tree in H_shuffled:
        if errors[tree] <= v:
            return tree

    return min(errors.items(), key=lambda x: x[1])[0]


def preProcess(df):
    """
    Categories the columns of the dataset.
    """
    result = df
    for col in result.columns[:-1]:
        result[col] = LabelEncoder().fit_transform(result[col])
    return result


def load_dataset(dataset_path,test_size=0.2, random_state=42):
    df = pd.read_csv(dataset_path)
    
    ### For ease of implementation, we will only use a few features
    df = df[config.selected_features]

    ### Preprocess the dataset
    df_final = preProcess(df)
    ### train-test split
    X = df_final.drop('satisfaction', axis=1)
    y = df_final['satisfaction']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size)
    print("Train set size:", X_train.shape[0], "Test set size:", X_test.shape[0])

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    
    X_train, X_test, y_train, y_test = load_dataset(config.dataset_path, test_size=0.2, random_state=42)

    H = build_candidate_trees(X_train, y_train, max_depth=3, num_trees=20, random_state=42)
    tree = replicable_learner(X_train, y_train, H, random_seed=42)
    a = tree.score(X_test, y_test)
    print(a)
    for t in H:
        print(t.get_params())
