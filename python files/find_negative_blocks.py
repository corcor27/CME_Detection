import os
import numpy as np
import pandas as pd
import shutil
import warnings
from sklearn.utils import shuffle

threshold_value = 40

excel_path = "CME_label_information_refined/CME_combined_excel_with_metadata.xlsx"

data = pd.read_excel(excel_path)

negative_blocks = {}

collection = []
count = 0
print(len(collection))
for row in range(0, data.shape[0]):
    acc = int(data["accept"].iloc[row])
    ref = data["TEL"].iloc[row]
    if acc == 0 and ref == 0:
        collection.append(row)
    else:
        if len(collection) != 0:
            negative_blocks["{}".format(count)] = collection
            count += 1
            collection = []
keys_to_remove = []
key_list = list(negative_blocks.keys())
print(len(key_list))
for key in key_list:
    length = len(negative_blocks[key])
    if length < threshold_value:
        del negative_blocks[key]

key_list = list(negative_blocks.keys())
print(len(key_list))
print(negative_blocks)
df = pd.DataFrame()
for key in key_list:
    values = negative_blocks[key]
    samples = data.iloc[min(values) +5: max(values) - 5]
    df = pd.concat([df, samples], axis=0)

fold = []
pos = []
neg = []
for row in range(0, df.shape[0]):
    img = df["image"].iloc[row]
    tel = df["TEL"].iloc[row]
    df["image"].iloc[row] = img.replace(".jpg", "")
    year = int(df["year"].iloc[row])
    if 2000 <= year < 2002:
        fold.append(1)
    elif 2002 <= year <2004:
        fold.append(2)
    elif 2004 <= year <2006:
        fold.append(3)
    elif 2006 <= year <2008:
        fold.append(4)
    elif 2008 <= year <2010:
        fold.append(5)
    if tel == "C2":
        pos.append(1)
        neg.append(0)
    else:
        pos.append(0)
        neg.append(1)
    
df["Fold"] = fold
df["Pos"] = pos
df["Neg"] = neg
print(df.shape)
#df.to_excel("CME_label_information_refined/final_negative_CMEs_after_refinement.xlsx")
"""
excel_path = "CME_label_information_refined/final_negative_CMEs_after_refinement.xlsx"

data = pd.read_excel(excel_path)

excel_path1 = "CME_label_information_refined/CMEs_final_training_list.xlsx"

data1 = pd.read_excel(excel_path1)

#combined = pd.concat([data, data1], axis=0)

pos = data1[(data1['Pos'] == 1)]
#neg = combined[(combined['Pos'] == 0)]
#print(pos.shape)
#pos = pos.iloc[:5000]
#neg = neg.iloc[:pos.shape[0]]

joint = pd.concat([pos, data], axis=0)
joint = shuffle(joint)
print(joint.shape)
joint.to_excel("CME_label_information_refined/CMEs_final_training_subset.xlsx")

"""

        
    
    