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
from config import CFG
import Algorithm10 as a10
from sklearn.tree import export_text
import time
import matplotlib.pyplot as plt
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

def getConvergenceSampleNum(min_subset_size, max_subset_size, repeat_num, config, sample_size_step=1):
    sample_size_replicablity_dict = {}
    
    X, y= a10.load_full_dataset(config.dataset_path,config, random_state=config.random_seed)
    for sample_size in range(min_subset_size, max_subset_size + 1, sample_size_step):
        #get dataset of size sample_size by sampling from the original dataset
        replicable_tree_list = []
        H = a10.build_candidate_trees(X, y,sample_size, max_depth=config.max_depth, num_trees=config.num_H, random_state=config.random_seed)    
        for i in range(repeat_num):
            #print(f"sample size: {sample_size}, repeat: {i}")
            res_trees = a10.replicable_learner(X, y, H, sample_size, config,random_seed=config.random_seed+i)
            tree = res_trees[0]
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
    return sample_size_replicablity_dict

def experiment(start_size, end_size,step,config):

    m_up_bound =  config.get_m_up_bound(config.num_H, config.rho, config.alpha, config.beta)
    print("theoretical sample size: ",)
    ans_dict = getConvergenceSampleNum(min_subset_size=start_size, max_subset_size=end_size, repeat_num=10, config=config, sample_size_step=step)
    for key, value in ans_dict.items():
        #print(f"sample size: {key}, prob: {value}")
        if value >= 1 - config.rho:
            print(f"replicable at sample size: {key}, prob: {value}")
            return key,m_up_bound,value
    print("not replicable at sample size: ", end_size)
    return -1,-1

def Exp_rho(rho_start=0.05, rho_end=0.95, rho_step=0.05, sample_size_start=100, sample_size_end=1000, sample_size_step=10):
    real_sample_size = []
    theoretical_sample_size = []
    rhos = []
    alphas = []
    betas = []
    num_Hs = []
    values = []
    for rho in np.arange(rho_start, rho_end + rho_step, rho_step):
        print(f"rho: {rho}")
        cfg = CFG()
        cfg.rho = rho
        sample_size,m_up_bound = experiment(sample_size_start, sample_size_end,sample_size_step,cfg)
        print(f"sample size: {sample_size}, m_up_bound: {m_up_bound}")
        real_sample_size.append(sample_size)
        theoretical_sample_size.append(m_up_bound)
        rhos.append(rho)
        num_Hs.append(cfg.num_H)
        alphas.append(cfg.alpha)
        betas.append(cfg.beta)
        
    # save the result to a csv file
    df = pd.DataFrame({
        'rhos': rhos,
        'alphas': alphas,
        'betas': betas,
        'num_Hs': num_Hs,
        'real_sample_size': real_sample_size,
        'theoretical_sample_size': theoretical_sample_size,
        'values': values
        
    })
    df.to_csv('sample_size_vs_rho.csv', index=False)
    print("Results saved to sample_size_vs_rho.csv")
    # print the results
    print("Real Sample Size: ", real_sample_size)
    print("Theoretical Sample Size: ", theoretical_sample_size)

def Exp_numH(numH_start=10, numH_end=40, numH_step=5, sample_size_start=100, sample_size_end=1000, sample_size_step=10):
    real_sample_size = []
    theoretical_sample_size = []
    rhos = []
    alphas = []
    betas = []
    num_Hs = []
    for numH in range(numH_start, numH_end + numH_step, numH_step):
        print(f"num_H: {numH}")
        cfg = CFG()
        cfg.num_H = numH
        sample_size,m_up_bound = experiment(sample_size_start, sample_size_end,sample_size_step,cfg)
        print(f"sample size: {sample_size}, m_up_bound: {m_up_bound}")
        real_sample_size.append(sample_size)
        theoretical_sample_size.append(m_up_bound)
        rhos.append(cfg.rho)
        num_Hs.append(numH)
        alphas.append(cfg.alpha)
        betas.append(cfg.beta)
        
    # save the result to a csv file
    df = pd.DataFrame({
        'rhos': rhos,
        'alphas': alphas,
        'betas': betas,
        'num_Hs': num_Hs,
        'real_sample_size': real_sample_size,
        'theoretical_sample_size': theoretical_sample_size,
        
    })
    df.to_csv('sample_size_vs_numH.csv', index=False)
    print("Results saved to sample_size_vs_numH.csv")
    # print the results
    print("Real Sample Size: ", real_sample_size)
    print("Theoretical Sample Size: ", theoretical_sample_size)

def plot_rho():
    # load the result from the csv file
    df = pd.read_csv('sample_size_vs_rho.csv')
    # plot the results
    real_sample_size = df['real_sample_size'].tolist()
    theoretical_sample_size = df['theoretical_sample_size'].tolist()
    rhos = df['rhos'].tolist()
    alphas = df['alphas'].tolist()
    betas = df['betas'].tolist()
    num_Hs = df['num_Hs'].tolist()
    # plot the results in logarithmic scale
    plt.figure(figsize=(10, 6))
    plt.plot(rhos, np.log(np.log(real_sample_size)), label='Real-world bound', marker='o',linewidth=4)
    plt.plot(rhos, np.log(np.log(theoretical_sample_size)), label='Theoretical bound', marker='x',linewidth=4)
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'$ln(ln(m))$')
    plt.title(r'$m$ vs. $\rho$'+'\n'+r'$\alpha$={}, $\beta$={},$|H|$={}'.format(alphas[0], betas[0], num_Hs[0]))
    plt.legend()
    plt.xlim(0,1)
    plt.ylim(1.5, 3)
    plt.grid()
    plt.show()

    
def plot_numH():
    # load the result from the csv file
    df = pd.read_csv('sample_size_vs_numH.csv')
    # plot the results
    real_sample_size = df['real_sample_size'].tolist()
    theoretical_sample_size = df['theoretical_sample_size'].tolist()
    rhos = df['rhos'].tolist()
    alphas = df['alphas'].tolist()
    betas = df['betas'].tolist()
    num_Hs = df['num_Hs'].tolist()
    # plot the results in logarithmic scale
    plt.figure(figsize=(10, 6))
    plt.plot(num_Hs, np.log(np.log(real_sample_size)), label='Real-world bound', marker='o',linewidth=4)
    plt.plot(num_Hs, np.log(np.log(theoretical_sample_size)), label='Theoretical bound', marker='x',linewidth=4)
    plt.xlabel(r'$|H|$')
    plt.ylabel(r'$ln(ln(m))$')
    plt.title(r'$m$ vs. $|H|$'+'\n'+r'$\alpha$={}, $\beta$={},$\rho$={}'.format(alphas[0], betas[0], rhos[0]))
    plt.legend()
    plt.xlim(10,40)
    plt.ylim(1.5, 3)
    plt.grid()
    plt.show()

def run_param_sweep(param_name, param_values, cfg_modifier_fn,
                    csv_filename, x_label, x_values=None,
                    sample_size_start=100, sample_size_end=1000, sample_size_step=10):
    real_sample_size = []
    theoretical_sample_size = []
    rhos = []
    alphas = []
    betas = []
    num_Hs = []
    max_depths = []
    x_vals = []

    for val in param_values:
        cfg = CFG()
        cfg_modifier_fn(cfg, val)
        print(f"{param_name} = {val}")
        sample_size, m_up_bound, _ = experiment(sample_size_start, sample_size_end, sample_size_step, cfg)
        real_sample_size.append(sample_size)
        theoretical_sample_size.append(m_up_bound)
        rhos.append(cfg.rho)
        alphas.append(cfg.alpha)
        betas.append(cfg.beta)
        num_Hs.append(cfg.num_H)
        max_depths.append(cfg.max_depth)
        x_vals.append(val)

    df = pd.DataFrame({
        x_label: x_vals,
        'real_sample_size': real_sample_size,
        'theoretical_sample_size': theoretical_sample_size,
        'rho': rhos,
        'alpha': alphas,
        'beta': betas,
        'num_H': num_Hs,
        'max_depth': max_depths
    })
    df.to_csv(csv_filename, index=False)
    print(f"Saved to {csv_filename}")
    return df

def plot_sweep(csv_path, x_key, x_label, title_prefix=""):
    df = pd.read_csv(csv_path)
    real_sample_size = df['real_sample_size'].tolist()
    theoretical_sample_size = df['theoretical_sample_size'].tolist()
    x_vals = df[x_key].tolist()

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, np.log(np.log(real_sample_size)), label='Real-world bound', marker='o', linewidth=4)
    plt.plot(x_vals, np.log(np.log(theoretical_sample_size)), label='Theoretical bound', marker='x', linewidth=4)
    plt.xlabel(x_label)
    plt.ylabel(r'$\ln(\ln(m))$')
    title = f"{title_prefix} vs. Sample Size (log-log scale)"
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig(f"./result/{title_prefix}_vs_sample_size.png")

def set_depth(cfg, val): cfg.max_depth = val
def set_alpha(cfg, val): cfg.alpha = val
def set_beta(cfg, val): cfg.beta = val
def set_rho(cfg, val): cfg.rho = val
def set_numH(cfg, val): cfg.num_H = val

if __name__ == "__main__":


    # df = run_param_sweep(
    #     param_name="rho",
    #     param_values=[0.05, 0.1, 0.15, 0.2],
    #     cfg_modifier_fn=set_rho,
    #     csv_filename="sample_size_vs_rho.csv",
    #     x_label="rho"
    # )
    # plot_sweep("sample_size_vs_rho.csv", x_key="rho", x_label=r"$\rho$", title_prefix=r"$\rho$")
    # df = run_param_sweep(
    #     param_name="num_H",
    #     param_values=[10, 20, 30, 40],
    #     cfg_modifier_fn=set_numH,
    #     csv_filename="sample_size_vs_numH.csv",
    #     x_label="num_H"
    # )
    # plot_sweep("sample_size_vs_numH.csv", x_key="num_H", x_label=r"$|H|$", title_prefix=r"$|H|$")

    df = run_param_sweep(
        param_name="beta",
        param_values=range(0.05, 1.0, 0.05),
        cfg_modifier_fn=set_beta,
        csv_filename="sample_size_vs_beta.csv",
        x_label="beta"
    )
    plot_sweep("sample_size_vs_beta.csv", x_key="beta", x_label=r"$\beta$", title_prefix=r"$\beta$")

    df = run_param_sweep(
        param_name="alpha",
        param_values=range(0.05, 1.0, 0.05),
        cfg_modifier_fn=set_alpha,
        csv_filename="sample_size_vs_alpha.csv",
        x_label="alpha"
    )
    plot_sweep("sample_size_vs_alpha.csv", x_key="alpha", x_label=r"$\alpha$", title_prefix=r"$\alpha$")


    df = run_param_sweep(
        param_name="max_depth",
        param_values=[1, 2, 3, 4, 5,6,7,8,9,10],
        cfg_modifier_fn=set_depth,
        csv_filename="sample_size_vs_max_depth.csv",
        x_label="max_depth"
    )
    plot_sweep("sample_size_vs_max_depth.csv", x_key="max_depth", x_label="max_depth", title_prefix="max_depth")