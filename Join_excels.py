import os
import numpy as np
import pandas as pd

Excel_folder = "CME_label_reject"

excel_list = os.listdir(Excel_folder)

base_p = os.path.join(Excel_folder, excel_list[0])

base_excel = pd.read_excel(base_p)

for item in excel_list[1:]:
    try:
        step_p = os.path.join(Excel_folder, item)
        step_excel = pd.read_excel(step_p)
        #print(base_excel.shape)
        base_excel = pd.concat([base_excel, step_excel])
        base_excel.drop_duplicates(subset=['DATE', 'TIME', 'TEL'], inplace=True)
        #print(base_excel.shape)
    except:
        print(item)
    

base_excel.to_excel("ALL_REJECT_CMES_DETAILS.xlsx")