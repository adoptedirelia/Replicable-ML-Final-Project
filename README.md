# Replicable-ML-Final-Project  
Implement a *replicable* version of decision tree model.   

## 1. Task
Build a Classification Decision Tree to determine whether a customer will be satisfied or not with an airline based on a list of selected features.  

## 2. Dataset
We use the dataset from https://www.kaggle.com/datasets/yakhyojon/customer-satisfaction-in-airline/data
  
## 3. Roadmap
*Refer Algorithm 10 in paper "Stability is Stable: Connections between Replicability, Privacy, and
Adaptive Generalization"* 
### 3.1 Generate input:
S1: Define parameters.   
S2: Generate finite class H (save a set of randomly initialized decision trees?)   

### 3.2 Algorithm 10
S1: Sample a subset from the dataset, and compute the loss of each model in 3.1-S2. Note that the default loss function of decision tree in sklearn is 'gini'.  
S2: Sample a value from the interval [OPT, OPT+alphs/2]. Note that *OPT* loss in decision tree is 0.  
S3: Select random threshold v.  
S4. Randomly order all H.  

### 3.3 Output
Output the first hypothesis f in the order s.t. err_S(f) < v.

## 4. Code Structure  
`./dataset/`: directory for datasets.  
`./models/`: directory for saved models.
`./tutorial/`: directory for decision tree model tutorial.  
`ClassicDT.py`: implements a pipeline to use decision tree model from sklearn.  
`config.py`: defines the global variable for the repository.  
`utils.py`: defines some utility functions.  
`Algorithm10.py`: Our implementation fo the Algorithm 10 in the paper.  
`Experiment.py`: our experiment codes.
