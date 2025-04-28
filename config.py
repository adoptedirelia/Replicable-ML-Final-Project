"""
Config the global variables
"""
import numpy as np 

dataset_path = './dataset/Invistico_Airline.csv'
model_path = './models/'
max_depth = 3
random_seed = 42
# note that last column is the label column
selected_features = ['Class','Seat comfort','Food and drink','Cleanliness','satisfaction'] 

# Parameters for replicable algorithm
rho = 0.3 # Replicability
alpha = 0.3 # Accuracy 
beta = 0.2 # Confidence
num_H = 10 # Number of hypotheses (model weights)
m_up_bound = (np.log(num_H)**2*np.log(1.0/rho) + rho**2*np.log(1.0/beta))/(alpha**2*rho**4)

tau_up_bound = (alpha*rho)/(np.log(num_H))
tau=tau_up_bound*0.1 # Replicability bucket size

# the following parameters are not used for the experiment
m=100 # Number of samples

def print_config_variables():
    """print all the global variables"""
    print("##### Config Variables #####")
    for name, value in globals().items():
        if not name.startswith("__") and "print" not in name:  # to avoid printing built-in variables
            print(f"{name} = {value}")
    print("############################")
print_config_variables()

