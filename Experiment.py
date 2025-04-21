"""
Reference paper: stability is stable 
Experiment: vary the value of rho and see the minimum sample size required for replicability

func getConvergenceSampleNum(hyperparams)
    For sample_num range(min_subsets_size, max_subsets_size, step):
        For range(repeat_num):
            random draw a subset of the dataset, whose size = sample_num
            Use algorithm 10 to get a model
        Check whether the models we get are 'replicable' to each other, accoding to the hyperparams. (see definition in page 14 of the paper)
        if converged, return current sample_num

func experiment(): 
# vary the rho and see minumum sample size required for replicability. 
# To make it simple, we can fix the value of other hyperparamers, such as alpha and beta.
    for rho in range(min_rho, max_rho, step):
        sample_num = getConvergenceSampleNum(rho)
        theoretical_sample_num = getTheoreticalSampleNum(rho) # compute according to algorithm 10 in the paper
        print("Hyperparams: ", hyperparams, "Sample size: ", sample_num)
    plot(hyperparams, theoretical_sample_num) # draw the curve, where x-axis is the hyperparams and y-axis is the sample size
    plot(hyperparams, sample_num) # draw the curve, where x-axis is the hyperparams and y-axis is the sample size
"""
import pandas as pd
import numpy as np
import config
import Algorithm10 as a10
from sklearn.tree import export_text
import time
# function to check if two decision trees are equal
def are_trees_equal(tree1, tree2):
    # Check that both trees are fitted
    if not hasattr(tree1, 'tree_') or not hasattr(tree2, 'tree_'):
        raise ValueError("Both trees must be fitted before comparison.")

    # Compare parameters
    if tree1.get_params() != tree2.get_params():
        return False

    t1 = tree1.tree_
    t2 = tree2.tree_

    # Compare structure and splitting rules
    attributes_to_check = [
        'children_left', 'children_right',
        'feature', 'threshold',
        'impurity', 'n_node_samples', 'weighted_n_node_samples',
        'value'
    ]

    for attr in attributes_to_check:
        if not np.array_equal(getattr(t1, attr), getattr(t2, attr)):
            return False

    return True

def getConvergenceSampleNum(min_subset_size, max_subset_size, repeat_num, rho, sample_size_step=1):
    sample_size_replicablity_dict = {}
    
    X, y= a10.load_full_dataset(config.dataset_path, random_state=config.random_seed)
    for sample_size in range(min_subset_size, max_subset_size + 1, sample_size_step):
        #get dataset of size sample_size by sampling from the original dataset
        replicable_tree_list = []
        t0= time.time()
        H = a10.build_candidate_trees(X, y,sample_size, max_depth=config.max_depth, num_trees=config.num_H, random_state=config.random_seed)    
        print(f"average build candidate trees time: {(time.time()-t0)/config.num_H}")
        for i in range(repeat_num):
            print(f"sample size: {sample_size}, repeat: {i}")
            tree = a10.replicable_learner(X, y, H, sample_size, random_seed=config.random_seed+i)
            replicable_tree_list.append(tree)
            
        #check the probability if the trees in the replicable_tree_list are the same
        same_tree_count = 0
        for i in range(len(replicable_tree_list)):
            for j in range(i + 1, len(replicable_tree_list)):
                
                if are_trees_equal(replicable_tree_list[i], replicable_tree_list[j]):
                    # print(f"tree {i} and tree {j} are the same")
                    same_tree_count += 1
        prob = same_tree_count / (repeat_num * (repeat_num - 1) / 2)
        sample_size_replicablity_dict[sample_size] = prob
        if prob >= 1-rho:
            print(f"replicable at sample size: {sample_size}, prob: {prob}")
            return sample_size_replicablity_dict
        
    # return sample_size_replicablity_dict
    print(f"not replicable at sample size between {min_subset_size} and {max_subset_size}, prob: {prob}")
    return -1

def experiment():
    print("theoretical sample size: ", config.m_up_bound)
    ans_dict = getConvergenceSampleNum(min_subset_size=10, max_subset_size=100, repeat_num=10, rho=config.rho, sample_size_step=10)  
    print("sample size replicablity dict: ", ans_dict)
    
if __name__ == "__main__":
    # run the experiment
    experiment()

