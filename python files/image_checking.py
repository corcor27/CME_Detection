import os
import numpy as np
import pandas as pd
import shutil
import warnings
from sklearn.utils import shuffle

excel_path = "CME_label_information_refined/CME_combined_excel_with_metadata.xlsx"

data = pd.read_excel(excel_path)

count = []
count2 = []

for row in range(0, data.shape[0]):
    img = data["image"].iloc[row]
    path = os.path.join("CME_label_information_refined/negative_samples", img)
    if os.path.exists(path):
        count.append(1)
    else:
        count.append(0)

for row in range(0, data.shape[0]):
    img = data["image"].iloc[row]
    path = os.path.join("CME_label_information_refined/negative_samples_2", img)
    if os.path.exists(path):
        count2.append(1)
    else:
        count2.append(0)   

        
data["Neg_base"] = count
data["Neg_reduced"] = count2

df = data[data.Neg_base == 1]

df.to_excel("CME_label_information_refined/Check_with_manual.xlsx")