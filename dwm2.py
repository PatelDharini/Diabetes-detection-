import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
df=pd.read_csv("project_data")
original_features_count= df.columns
row_count = df.shape[0]
print('Original Features count:', len(original_features_count),)
print('list of the original features: ', original_features_count)
print("Number of rows:", row_count)
