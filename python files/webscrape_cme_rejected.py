import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os


def create_dataframe(string, string2):
    title_len = 8
    title_string = string[:title_len]
    content_string = string2
    len_contents = int(len(content_string)/title_len)

    
    array = np.array(content_string)
    array = array.reshape(len_contents, title_len)
    
    dataframe = pd.DataFrame(array, columns= title_string)
    
    return dataframe

out_folder = "CME_label_information_refined"
if os.path.exists(out_folder) == False:
    os.mkdir(out_folder)

collection_out = os.path.join(out_folder, "rejected")
if os.path.exists(collection_out) == False:
    os.mkdir(collection_out)



base_url = "https://cdaw.gsfc.nasa.gov/CME_list/"

issues = pd.DataFrame()
completed = pd.DataFrame()
data  = requests.get(base_url)
soup = BeautifulSoup(data.text, 'lxml')
table = soup.find("table")
find_links = table.findAll('a')
links = [link['href'] for link in find_links]
### 2000 to 2009 or 44:164
for link in links[44:164]:
    print(link)
    new_url = base_url + link
    new_data = requests.get(new_url)
    soup = BeautifulSoup(new_data.text, 'lxml')
    table = soup.find("table")
    

    table_rows = table.findAll("tr")

    for level in range(1, len(table_rows)):
        remarks = table_rows[level].find_all("td")[-1].text
        find_links = table_rows[level].find_all("a")
        date_tag = str(table_rows[level].find_all("td")[0].text).strip()
        time_tag = str(table_rows[level].find_all("td")[1].text).strip()

        p = 0
        d = 0
        if p == 0:
            if d == 0:
                
                new_links = [link['href'] for link in find_links]
  
                for next_link in new_links:
                    if "yht" in next_link:
                        split_url = new_url.split("/")
                        combined_url = ""
                        for st in split_url[:-1]:
                            combined_url = combined_url + "/" + st
                        combined_url = combined_url[1:] + "/" + next_link
                        tag_link_split = next_link.split("/")
                        tag = tag_link_split[-1].replace(".yht", ".xlsx")
                        
                        tag2 = tag_link_split[-1].replace(".yht", "")
                        out_path = os.path.join(collection_out, tag)
                        try:
                            table_data = requests.get(combined_url)
                            htmlParse = BeautifulSoup(table_data.text, 'lxml')
                            para = htmlParse.find_all("p")
                            #print(len(para))
                            for para in htmlParse.find_all("p"):
                                infor = para.get_text()
                                POS = infor.index("HEIGHT")
                                new_str = infor[POS:]
                                split_new_string = new_str.split(" ")
                                while("" in split_new_string):
                                    split_new_string.remove("")
                                fixed_string = []
                                for item in range(0, len(split_new_string)):
                                    if "\n" in split_new_string[item]:
                                        split_new_string[item] = split_new_string[item].strip()
                                POS2 = infor.index("WDATA")
                                POS3 = infor.index("#HALO")
                                new_str2 = infor[POS2:POS3]
                                split_new_string2 = new_str2.split(" ")
                                while("" in split_new_string2):
                                    split_new_string2.remove("")
                                for item in range(0, len(split_new_string2)):
                                    if "\n" in split_new_string2[item]:
                                        split_new_string2[item] = split_new_string2[item].strip()
                                for item in range(0, len(split_new_string2)):
                                    if "\n#WDATA" in str(split_new_string2[item]):
                                        split_new_string2[item] = split_new_string2[item].replace("\n#WDATA:", "")
                                split_new_string2 = split_new_string2[1:]

                                pandas_frame = create_dataframe(split_new_string, split_new_string2)
                                name_tags = []
                                for ii in range(0, pandas_frame.shape[0]):
                                    name_tags.append(tag2)
                                pandas_frame["Folder_tag"] = name_tags
                                row_information = pd.DataFrame()
                                row_information["Folder_tag"] = [tag]
                                row_information["date"] = [date_tag]
                                row_information["time"] = [time_tag]
                                row_information["remark"] = [str(remarks).strip()]
                                completed = pd.concat([completed, row_information], axis=0)
                                pandas_frame.to_excel(out_path)

                        except:

                            row_information = pd.DataFrame()
                            row_information["Folder_tag"] = [tag]
                            row_information["date"] = [date_tag]
                            row_information["time"] = [time_tag]
                            row_information["remark"] = [remarks]
                            issues = pd.concat([issues, row_information], axis=0)
                            continue

completed.to_excel(os.path.join(out_folder, "rejected_collected_samples_extra.xlsx"))
issues.to_excel(os.path.join(out_folder, "rejeted_issues_samples.xlsx"))

            
