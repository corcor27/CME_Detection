import os
import numpy as np
import pandas as pd
import shutil
import warnings
from sklearn.utils import shuffle

excel_path = "CME_label_information_refined/CME_combined_excel_with_metadata.xlsx"

data = pd.read_excel(excel_path)
print(data.shape)
negatives = data[(data['ANGLE'] == 0)]
data = data[(data['ANGLE'] > 0)]

print(negatives.shape)
instances = data.drop_duplicates(subset=["image"])
issues = []
for row in range(0, instances.shape[0]):

    img = instances["image"].iloc[row]
    step = data[(data['image'] == img)]
    check = list(step["accept"])
    if 0 in check:
        issues.append(0)

    else:
        issues.append(1)


instances["accept2"] = issues
positives = instances[(instances['accept2'] == 1)]
positives = positives[(positives['TEL'] == "C2")]
positives.to_excel("CME_label_information_refined/CME_Positive_cases_for_training.xlsx")    
negatives.to_excel("CME_label_information_refined/CME_Negatives_cases_for_training.xlsx") 

check2 = []

for row in range(0, positives.shape[0]):
    height = positives["HEIGHT"].iloc[row]
    if 3 <= height <= 6:
        check2.append(1)
    else:
        check2.append(0)
positives["check2"] = check2
positives = positives[(positives['check2'] == 1)]

combined = pd.concat([positives, negatives], axis=0)

fold = []
pos = []
neg = []
for row in range(0, combined.shape[0]):
    img = combined["image"].iloc[row]
    tel = combined["TEL"].iloc[row]
    combined["image"].iloc[row] = img.replace(".jpg", "")
    year = int(combined["year"].iloc[row])
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
    
combined["Fold"] = fold
combined["Pos"] = pos
combined["Neg"] = neg

combined = shuffle(combined)

combined.to_excel("CME_label_information_refined/CMEs_final_training_list.xlsx")

pos = combined[(combined['Pos'] == 1)]
neg = combined[(combined['Pos'] == 0)]

neg = neg.iloc[:pos.shape[0]]

joint = pd.concat([pos, neg], axis=0)
joint = shuffle(joint)

joint.to_excel("CME_label_information_refined/CMEs_final_training_subset.xlsx")

