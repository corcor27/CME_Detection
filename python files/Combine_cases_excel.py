import os
import numpy as np
import pandas as pd
import shutil

base_folder = "CME_label_information_refined"

excel_case_list = os.path.join(base_folder, "correctly_collected_samples.xlsx")
save_excel_path = os.path.join(base_folder, "collected_samples_complete_metadata.xlsx")
excels_folder = os.path.join(base_folder, "collection")

data = pd.read_excel(excel_case_list)
df = pd.DataFrame() ## where all the samples will be compiled too
# from our list of cases we loop through all the cases and add them to another file dataframe to combine them
for row in range(0, data.shape[0]):
    excel_path = os.path.join(excels_folder, data["Folder_tag"].iloc[row])
    case_excel = pd.read_excel(excel_path)

    rem = str(data["remark"].iloc[row])
    if "Poor Event"  not in rem:
        if "Only C3" not in rem: 

            case_excel["accept"] = [1 for i in range(0, case_excel.shape[0])]
        else:
            case_excel["accept"] = [0 for i in range(0, case_excel.shape[0])]

    else:

        case_excel["accept"] = [0 for i in range(0, case_excel.shape[0])]
    df = pd.concat([df, case_excel], axis=0)


# now we need to remove anything that we dont need e.g. C3 samples 
#df = df[(df['TEL'] == "C2")]

df.to_excel(save_excel_path)


excel_case_list = os.path.join(base_folder, "rejected_collected_samples.xlsx")
save_excel_path = os.path.join(base_folder, "issues_samples_complete_metadata.xlsx")
excels_folder = os.path.join(base_folder, "rejected")

data = pd.read_excel(excel_case_list)
df = pd.DataFrame() ## where all the samples will be compiled too

# from our list of cases we loop through all the cases and add them to another file dataframe to combine them
for row in range(0, data.shape[0]):
    excel_path = os.path.join(excels_folder, data["Folder_tag"].iloc[row])
    case_excel = pd.read_excel(excel_path)
    rem = str(data["remark"].iloc[row])
    
    case_excel["accept"] = [0 for i in range(0, case_excel.shape[0])]
    df = pd.concat([df, case_excel], axis=0)
    
# now we need to remove anything that we dont need e.g. C3 samples 
#df = df[(df['TEL'] == "C2")]

df.to_excel(save_excel_path)

base_folder = "CME_label_information_refined"
save_excel_path = os.path.join(base_folder, "collected_samples_complete_metadata.xlsx")
save_excel_path1 = os.path.join(base_folder, "issues_samples_complete_metadata.xlsx")

save_excel = os.path.join(base_folder, "complete_metadata_positve_CMEs.xlsx")

df = pd.read_excel(save_excel_path)

df1 = pd.read_excel(save_excel_path1)






combined = pd.concat([df,df1], axis = 0)
ref_point = []
reduce = []
for row in range(0, combined.shape[0]):
    date = combined["DATE"].iloc[row]
    time = combined["TIME"].iloc[row]
    date = date.split("/")
    reduce.append(date[0])
    time = time.split(":")
    date = "".join(date)
    time = "".join(time)
    ref_point.append(date + "_" + time)
combined["Image_Reference"] = ref_point
combined["year"] = reduce
combined = combined.sort_values('Image_Reference', ascending=True)
combined.to_excel(save_excel)
"""
base_folder = "CME_label_information_refined"
save_excel_path = os.path.join(base_folder, "issues_samples_complete_metadata.xlsx")
data = pd.read_excel(save_excel_path)
data = data[data.TEL == "C2"]
h = list(data["HEIGHT"])
print(data.shape)
max_val = max(h)
min_val = min(h)
print(max_val)
print(min_val)
print(np.mean(h))
print(np.std(h))
print(max_val - min_val)
"""
