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