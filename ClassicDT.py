"""
Code for using the classic decision tree algorithm.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset_path = "./dataset/loan.csv"
def main(dataset_path):
    
    ### Load the dataset
    df = pd.read_csv(dataset_path)
    df_test = df[['gender', 'education_level', 'age', 'credit_score', 'loan_status']].tail(30).reset_index(drop=True)

    