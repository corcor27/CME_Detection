import os
import numpy as np
import pandas as pd
import shutil
import warnings

excel_path = "CME_label_information_refined/CME_combined_excel_with_metadata.xlsx"

data = pd.read_excel(excel_path)

positive_cases = pd.DataFrame()

for row in range(0, data.shape[0]):
    val = float(data["HEIGHT"].iloc[row])
    acc = int(data["accept"].iloc[row])
    if val < 3 or val > 7:
        if acc == 1:
            data["accept"].iloc[row] = 0

df = data[data.accept == 1]

df = df.drop_duplicates(subset=["ref"])

df.to_excel("CME_label_information_refined/final_good_CMEs_after_refinement.xlsx")
            

