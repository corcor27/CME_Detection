import os
import numpy as np
import pandas as pd
import shutil
import warnings
from sklearn.utils import shuffle

positve_path = "CME_label_information_refined/final_positive_CMEs_after_refinement.xlsx"
negative_path = "CME_label_information_refined/final_negative_CMEs_after_refinement.xlsx"

pos = pd.read_excel(positve_path)
neg = pd.read_excel(negative_path)


ones = [1 for i in range(0, pos.shape[0])]
zeros = [0 for i in range(0, pos.shape[0])]
pos["Pos"] = ones
pos["Neg"] = zeros

ones = [1 for i in range(0, neg.shape[0])]
zeros = [0 for i in range(0, neg.shape[0])]
neg["Neg"] = ones
neg["Pos"] = zeros

combined = pd.concat([pos, neg], axis=0)

fold = []
for row in range(0, combined.shape[0]):
    img = combined["image"].iloc[row]
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
    
combined["Fold"] = fold

combined = shuffle(combined)

combined.to_excel("CME_label_information_refined/CMEs_final_training_list.xlsx")



