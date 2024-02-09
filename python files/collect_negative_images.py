import os
import numpy as np
import pandas as pd
import shutil
from sklearn.utils import shuffle

negative_path = "CME_label_information_refined/CMEs_final_training_list.xlsx"

collection_out = os.path.join("CME_label_information_refined", "all_negative_samples")
if os.path.exists(collection_out) == False:
    os.mkdir(collection_out)
data = pd.read_excel(negative_path)
data = data[data.Neg == 1]
for row in range(0, data.shape[0]):
    print(row)
    year = os.path.join(collection_out, str(data["year"].iloc[row]))
    if os.path.exists(year) == False:
        os.mkdir(year)
    mon = str(data["ref"].iloc[row])
    mon = mon[4:6]
    print(mon)
    month = os.path.join(year, mon)
    if os.path.exists(month) == False:
        os.mkdir(month)
    img_path = os.path.join("All_CME_Data_2000_2009", str(data["image"].iloc[row]) + ".jpg")
    save_path = os.path.join(month, str(data["image"].iloc[row])+".jpg")
    shutil.copy(img_path, save_path)
    
    

    
"""

negative_path = "CME_label_information_refined/CMEs_final_training_list.xlsx"

collection_out = os.path.join("CME_label_information_refined", "negative_samples_2")

data = pd.read_excel(negative_path)

Neg_list = os.listdir(collection_out)
for item in range(0, len(Neg_list)):
    tag = Neg_list[item]
    Neg_list[item] = tag.replace(".jpg", "")
rows = []
for row in range(0, data.shape[0]):
    item = data["Name"].iloc[row]
    if item in Neg_list:
        rows.append(1)
    else:
        rows.append(0)
data["check"] = rows
Neg = data[data.check == 1]
Pos = data[data.Pos == 1]
print(Pos.shape)
Pos = Pos.iloc[:Neg.shape[0]]

combined = pd.concat([Pos, Neg])

combined = shuffle(combined)
print(combined.shape)
combined.to_excel("CME_testing.xlsx")
""" 
    