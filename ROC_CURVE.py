import numpy as np
import os
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer

from sklearn.metrics import RocCurveDisplay

Den = [2,2,2,2,0]
Res = [2,1,2,2,2]
Eff = [0,2,1,1,1]

results = {}

den_res = []
for fold in range(0, len(Den)):
    path = os.path.join("{}".format(fold), "{}_{}".format("Densenet121", Den[fold]), "{}_{}_{}_{}.npy".format("Densenet121", fold,  Den[fold],  Den[fold]))
    samples = np.load(path)
    if fold == 0:
        den_res = samples
        
    else:
        den_res = np.concatenate((den_res, samples), axis=0)


res_res = []
for fold in range(0, len(Den)):
    path = os.path.join("{}".format(fold), "{}_{}".format("Resnet101", Res[fold]), "{}_{}_{}_{}.npy".format("Resnet101", fold,  Res[fold],  Res[fold]))
    samples = np.load(path)
    if fold == 0:
        res_res = samples
        
    else:
        res_res = np.concatenate((res_res, samples), axis=0)
        
eff_res = []
for fold in range(0, len(Den)):
    path = os.path.join("{}".format(fold), "{}_{}".format("EfficientNetB4", Eff[fold]), "{}_{}_{}_{}.npy".format("EfficientNetB4", fold,  Eff[fold],  Eff[fold]))
    samples = np.load(path)
    if fold == 0:
        eff_res = samples
        
    else:
        eff_res = np.concatenate((eff_res, samples), axis=0)
print(eff_res.shape)
data = pd.read_excel("CMEs_final_training_subset.xlsx")
data = data.iloc[:eff_res.shape[0]]
pos = np.array(data["Pos"])
neg = np.array(data["Neg"])

pos = np.expand_dims(pos, axis=-1)
neg = np.expand_dims(neg, axis=-1)
ground = np.concatenate((pos,neg), axis=1)

den_fpr, den_tpr, den_threshold = metrics.roc_curve(ground.flatten(), den_res.flatten())
den_roc_auc = metrics.auc(den_fpr, den_tpr)

res_fpr, res_tpr, res_threshold = metrics.roc_curve(ground.flatten(), res_res.flatten())
res_roc_auc = metrics.auc(res_fpr, res_tpr)

eff_fpr, eff_tpr, eff_threshold = metrics.roc_curve(ground.flatten(), eff_res.flatten())
eff_roc_auc = metrics.auc(eff_fpr, eff_tpr)



plt.title('Receiver Operating Characteristic')
plt.plot(den_fpr, den_tpr, label = 'Densenet121 AUC = %0.2f' % den_roc_auc)
plt.plot(res_fpr, res_tpr, label = 'Resnet101 AUC = %0.2f' % res_roc_auc)
plt.plot(eff_fpr, eff_tpr, label = 'EfficientNetB4 AUC = %0.2f' % eff_roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')



    