import os
import numpy as np
import pandas as pd
import shutil

base = "CME_label_information_refined/CME_Solar_Cycle2000-2009"
folder_list = os.listdir(base)
#loop through case list
data = pd.DataFrame()
years = []
folder = []
images = []
ref_point = []
for year in folder_list:
    year_path = os.path.join(base, year)
    case_list = os.listdir(year_path)
    for case in case_list:
        case_path = os.path.join(year_path, case)
        image_list = os.listdir(case_path)
        ## now loop through images in each case
        for img in image_list:
            if ".jpg" in img:
                years.append(year)
                folder.append(case)
                images.append(img)
                ref = img.split("_")
                ref = ref[4] + "_" + ref[5]
                ref = ref.replace(".jpg", "")
                ref_point.append(ref)
                
data["year"] = years
data["case"] = folder
data["image"] = images           
data["ref"] = ref_point

data.to_excel("CME_label_information_refined/CME_case_image_list.xlsx")      
