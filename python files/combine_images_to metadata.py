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
    for row in range(0, data.shape[0]):
        ref = data["Image_Reference"].iloc[row]
        data["Image_Reference"].iloc[row] = ref[:-3]
    return list(data["Image_Reference"])
    

warnings.filterwarnings("ignore")
base_folder = "CME_label_information_refined"

case_metadata = os.path.join(base_folder, "cases_metadata")
if os.path.exists(case_metadata) == False:
    os.mkdir(case_metadata)
    
image_list_excel = "CME_label_information_refined/CME_case_image_list.xlsx"
metadata_excel = "CME_label_information_refined/complete_metadata_positve_CMEs.xlsx"

images_data = pd.read_excel(image_list_excel)
metadata = pd.read_excel(metadata_excel)
combined_excel = pd.DataFrame()
meta_data_ref_points = fix_ref_point(metadata)

columns_list = list(metadata.keys())
print(columns_list)

year_list = create_instance_list(images_data, "year")

for year in year_list:
    year_metadata = os.path.join(case_metadata, str(year))
    if os.path.exists(year_metadata) == False:
        os.mkdir(year_metadata)
    refined_data_by_year = images_data[(images_data["year"] == year)]
    case_list = create_instance_list(refined_data_by_year, "case")
    for case in case_list:
        print(case)
        metadata_collected = pd.DataFrame()
        cases_metadata = os.path.join(year_metadata, str(case))
        if os.path.exists(cases_metadata) == False:
            os.mkdir(cases_metadata)
        refined_data_by_cases = refined_data_by_year[(refined_data_by_year["case"] == case)]
        for row in range(0, refined_data_by_cases.shape[0]):
            ref_point = refined_data_by_cases["ref"].iloc[row]
            frame = pd.DataFrame()
            try:
                POS = meta_data_ref_points.index(ref_point[:-3])
                
                for item in columns_list:
                    val = metadata[item].iloc[POS]
                    frame[item] = [val]
                
                metadata_collected = pd.concat([metadata_collected, frame], axis=0)    
                
                
            except:
                for item in columns_list:
                    val = metadata[item].iloc[POS]
                    frame[item] = [0]
                metadata_collected = pd.concat([metadata_collected, frame], axis=0) 
                continue
        output = os.path.join(cases_metadata, "{}.xlsx".format(case))
        refined_data_by_cases = refined_data_by_cases.reset_index()
        metadata_collected = metadata_collected.reset_index()
        refined_data_by_cases = pd.concat([refined_data_by_cases, metadata_collected], axis=1)
        print(refined_data_by_cases.shape)
        refined_data_by_cases = refined_data_by_cases.sort_values('ref', ascending=True)
        refined_data_by_cases.to_excel(output)
        combined_excel = pd.concat([combined_excel, refined_data_by_cases], axis=0)
combined_excel = combined_excel.sort_values('ref', ascending=True)
combined_excel.to_excel("CME_label_information_refined/CME_combined_excel_with_metadata.xlsx")        
        
        

