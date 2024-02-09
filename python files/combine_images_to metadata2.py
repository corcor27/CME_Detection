import os
import numpy as np
import pandas as pd
import shutil
import warnings

def create_instance_list(data, ref):
    refined_data = data.drop_duplicates(subset=[ref])
    refined_list = list(refined_data[ref])
    return refined_list

def fix_ref_point(data):
    ref2 = []
    for row in range(0, data.shape[0]):
        ref = data["Image_Reference"].iloc[row]
        ref2.append(str(ref[:-3]))
    data["ref2"] = ref2
    return data
    

base_folder = "CME_label_information_refined"


    
image_list_excel = "CME_label_information_refined/CME_case_image_list.xlsx"
metadata_excel = "CME_label_information_refined/complete_metadata_positve_CMEs.xlsx"

images_data = pd.read_excel(image_list_excel)
metadata = pd.read_excel(metadata_excel)
combined_excel = pd.DataFrame()
meta_data_ref_points = fix_ref_point(metadata)

columns_list_meta = list(metadata.keys())
columns_list_meta.remove("year")
columns_list_images = list(images_data.keys())
print(columns_list_meta)
print(columns_list_images)
for row in range(0, images_data.shape[0]):
    image_ref = images_data["ref"].iloc[row]
    metadata_collect = meta_data_ref_points[meta_data_ref_points.ref2 == str(image_ref[:-3])]
    if metadata_collect.shape[0] > 0:
        for rw in range(0, metadata_collect.shape[0]):
            frame = pd.DataFrame()
            for item in columns_list_images:
                val = images_data[item].iloc[row]
                frame[item] = [val]
            for item in columns_list_meta:
                val = metadata_collect[item].iloc[rw]
                frame[item] = [val]
            combined_excel = pd.concat([combined_excel, frame], axis=0)
    else:
        frame = pd.DataFrame()
        for item in columns_list_images:
            val = images_data[item].iloc[row]
            frame[item] = [val]
        for item in columns_list_meta:
            frame[item] = [0]
        combined_excel = pd.concat([combined_excel, frame], axis=0)
        
combined_excel = combined_excel.sort_values('ref', ascending=True)
combined_excel.to_excel("CME_label_information_refined/CME_combined_excel_with_metadata.xlsx")           
        

