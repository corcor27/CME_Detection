import pandas as pd
import os 
import pydicom as dicom
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.transform import resize
from sklearn.utils import shuffle
import json
import shutil
import tarfile
import imageio.v2 as imageio
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
import tarfile
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.stats import tukey_hsd
from scipy.stats import f_oneway
from scipy import stats
from scipy.stats import wilcoxon
from sklearn.metrics import mean_squared_error
from scipy import stats
from numpy.random import default_rng

def t_test(mean1, std1, mean2, std2, n_samples1, n_samples2):
    diff_means = np.sqrt((mean1 - mean2)**2)

    s = (((n_samples1 - 1)*std1**2)+((n_samples2 - 1)*std2**2))/(n_samples1 + n_samples2 - 2)
    s = np.sqrt(s)

    t = diff_means/(s*(np.sqrt((1/n_samples1)+(1/n_samples2))))
    print(t)
    
    return t

ran_select = True

excel_file = "results.xlsx"
data = pd.read_excel(excel_file)
model_options = data.drop_duplicates(subset=["Backbone"])
model_options = list(model_options["Backbone"])
fold_options = data.drop_duplicates(subset=["test_fold"])
fold_options = fold_options.sort_values('test_fold', ascending=True)
fold_options = list(fold_options["test_fold"])

repeat_options = data.drop_duplicates(subset=["repeat"])
repeat_options = list(repeat_options["test_fold"])
groups1 = pd.DataFrame()
groups2 = pd.DataFrame()
eval_val1 = "Val"
eval_test1 = "Test"
eval_val2 = "Rocval"
eval_test2 = "Roctest"
files = []
dic = {}

for model in model_options:
    out_path = "{}_results.xlsx".format(model)
    files.append(out_path)
    
    results = []
    validation = []
    mean = []
    std = []
    
    for test in fold_options:
        #test_results = np.zeros((1, len(fold_options)))
        
            
        df = data[(data["Backbone"] == model) & (data['test_fold'] == test)]
        repeat_results = list(df[eval_test2])
        
        #std = np.mean(np.array(repeat_results))
        #best_pos = repeat_results.index(np.max(repeat_results))
        #best_result = df[eval_test2].iloc[best_pos]
        #best_result = df[eval_val2].iloc[best_pos]
        #test_results[:,fold_options.index(val)] = np.mean(repeat_results)
        #res = pd.DataFrame(data=test_results, index=["test_fold_{}".format(test)],   columns=fold_options)
        #results = pd.concat(([results, res]), axis=0)
        results.append(repeat_results)
    results = np.array(results)
    g = []
    print(results.shape)
    for ii in range(0, results.shape[1]):
        tag = np.mean(results[:, ii])
        g.append(tag)
    dic[model] = g
#print(dic)       
    #groups1[model] = results
#print(dic)
means = []
stds = []

for model in model_options:
    #print(dic[model])
    means.append(np.mean(dic[model]))
    stds.append(np.std(dic[model]))
print(means)


p_values = {}
for model1 in range(0, len(model_options)):
    for model2 in range(0, len(model_options)):
        if model1 != model2:
            print(model_options[model1], model_options[model2])
            #t_test(means[model1], stds[model1], means[model2], stds[model2], 5, 5)
            p = stats.ttest_rel(a=dic[model_options[model1]], b=dic[model_options[model2]])
            #tukey_hsd(dic[model_options[model1]], list(groups1[model_options[1]]), list(groups1[model_options[2]]), list(groups1[model_options[3]]))
            print(p)


    
        

