"""
Some utility functions
"""
import numpy as np
import random
def setup_seed(seed):
     np.random.seed(seed)
     random.seed(seed)