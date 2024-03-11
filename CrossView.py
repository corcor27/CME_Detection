# from keras_cv_attention_models import coatnet
import tensorflow as tf
from tensorflow import keras
import pandas as pd

from CustomDataGen import CustomDataGen
from Model_Class import CustomModel
from Model_Class_Multi import CustomModel_Multi
import argparse
import sys

import os
import numpy as np
from sklearn.metrics import accuracy_score
import itertools
import cv2
from sklearn.utils import shuffle
#import sys

"""
to run use for example:
python 5Clust_Train_CustModel.py --root_path "DATASET" --excel_file "DCIS_LIST2_numpy.xlsx" --wieghts_output_dir "WIEGHTS" --wieghts_path "None" --mode train
"""
def create_folds(args, data):
    train = []
    test = []
    if args.fixed_test:
        args.test_fold = 0
    scale = int(round(data.shape[0]/args.Num_folds))
    for ii in range(0, data.shape[0]):
        if args.test_fold*scale < ii <= (args.test_fold +1)*scale:
            test.append(ii)
        else: 
            train.append(ii)
    train_data = data.iloc[train]
    
    test_data = data.iloc[test]
    train_split = int(round(train_data.shape[0] *0.8))
    t_data = train_data.iloc[:train_split]
    v_data = train_data.iloc[train_split:]
    return t_data, v_data, test_data

def create_folds2(args, data):
    
    
    train_data = data[data.Fold != args.test_fold]
    
    test_data = data[data.Fold == args.test_fold]
    train_split = int(round(train_data.shape[0] *0.6))
    t_data = train_data.iloc[:train_split]
    v_data = train_data.iloc[train_split:]
    return t_data, v_data, test_data



parser = argparse.ArgumentParser()

###model options
parser.add_argument('--model_type',type=str, default = "Classification", help='classification or segmentation')
parser.add_argument('--val_fold', type=int,default=0, help='Training fold')
parser.add_argument('--test_fold', type=int ,default=0, help='Training fold')
parser.add_argument('--mode',type=str, default="train", help='train, inference, features, collect_imgs')
parser.add_argument('--connection_type', default = "Baseline" , help='What crossview connection method would you like to use')
parser.add_argument('--pretrain', default = False, help='do you want to use a pretrained version')
parser.add_argument('--Num_folds', type=int, default = 5, help='Number of testing folds')
parser.add_argument('--repeat', type=int, default = 0, help='repeat counter')
parser.add_argument('--tag', type=str, default = "CME_Detection", help='Run name')
#parser.add_argument('--tag', type=str, default = "Minst", hlp='Run name')
parser.add_argument('--load_weight', default = False, help='do you want to use a pretrained version')
parser.add_argument('--multi_process', default = False, help='more than one GPU')
parser.add_argument('--class_weights', default = False, help='more than one GPU')
###path selections
#parser.add_argument('--dataset_path_img', type=str, default="DATASET/All_images_Optimum/Evenscale_segs_bb_imgs", help='dataset_path')
parser.add_argument('--dataset_path_img', type=str, default="All_CME_Data_2000_2009", help='dataset_path')
#parser.add_argument('--dataset_path_img', type=str, default="DATASET/MINST/Image", help='dataset_path')
parser.add_argument('--dataset_path_mask', type=str, default='DATASET/All_images_Optimum/Masks', help='Segmentation mask path')

parser.add_argument('--excel_file', type=str, help='list dir', default="CMEs_final_training_subset.xlsx")
#parser.add_argument('--excel_file', type=str, help='list dir', default="INV_DCIS_DUAL_VIEW2.xlsx")
#parser.add_argument('--excel_file', type=str, help='list dir', default="MINST_LABELS.xlsx")
parser.add_argument('--wieghts_output_dir', type=str, help='output dir', default="WIEGHTS")
parser.add_argument('--weight_path', type= str, default="None",help='weights path')
parser.add_argument('--training_type', type= str, default="normal",help='normal, balanced etc')
parser.add_argument('--backbone', type= str, default="Densenet121",help='what backbone model, e.g. Resenet18')
parser.add_argument('--num_pool', type= int, default=0,help='weights path')
parser.add_argument('--fixed_test',  default=False, help='weights path')
parser.add_argument('--logits',  default=False, help='weights path')
parser.add_argument('--use_output_bias',  default=False, help='weights path')
parser.add_argument('--two_way_shuffle',  default=False, help='if you want to shuflle along axis 1')

##training parmeters"DCISGRADE_DUALVIEW.xlsx"

parser.add_argument('--num_output_classes', type=int, default=2, help='number of output channels')
parser.add_argument('--num_input_objects', type=int, default=1, help='number of inputs for network')
parser.add_argument('--num_output_objects', type=int, default=1, help='number of outputs for network')
parser.add_argument('--Optimizer', type=str, default="Adam", help='Type of Optimizer')
parser.add_argument('--max_epochs', type=int, default=40, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=64, help='batch_size per gpu')
parser.add_argument('--per_worker_batch_size', type=int, default=64, help='batch_size per gpu')
parser.add_argument('--img_size', type=int,default=256, help='input patch size of network input')
parser.add_argument('--aug', default = True, help='do you want augmentation? true or false')
parser.add_argument('--shuffle', default = True, help='shuffle ?')
parser.add_argument('--image_format',type=str, default = "image", help='image or numpy files')
parser.add_argument('--img_colour', default = True, help='do you want to use colour')
parser.add_argument('--loss_type', type=str, default = "cross", help='type of loss function you would like to use')
parser.add_argument('--dropout', type=int, default=0.4, help='dropout percentage as decimal')
parser.add_argument('--threashold', type=int, default=199, help='dropout percentage as decimal')
parser.add_argument('--nn', type=int, default=200, help='dropout percentage as decimal')

parser.add_argument('--ext_type', type=str, default = ".jpg", help='Image extention')
parser.add_argument('--ext_type_mask', type=str, default = ".npy", help='Segmentation mask extention')

###Learning rate
parser.add_argument('--Max_lr', type=float,  default=0.001,help='network max learning rate')
parser.add_argument('--Min_lr', type=float,  default=0.00000001,help='network min learning rate')
parser.add_argument('--Starting_lr', type=float,  default=0.001,help='intial lr')
parser.add_argument('--lr_schedule', type= str, default="Exp_decay",help='learning rate schedule')
parser.add_argument('--warmup_epochs', type=int, default = 10, help='Number of epochs at reduced lr')
parser.add_argument('--lr_cooldown', type=int, default = 10, help='Number of epochs at reduced lr')
###General parmeters

parser.add_argument('--seed', type=int,default=1234, help='random seed')
parser.add_argument('--cooldown', type=int, default=40, help='batch_size per gpu')
parser.add_argument('--features', default=False, help='features extracted per layer')
###Excel Parmeters

#parser.add_argument('--image_col',type=str, default = "MARKID1", help='column name in excel that detail image names')
#parser.add_argument('--second_image_col',type=str, default = "MARKID2", help='if dual input add 2nd col name')
parser.add_argument('--image_col',type=str, default = "image", help='column name in excel that detail image names')
parser.add_argument('--second_image_col',type=str, default = "CC_img", help='if dual input add 2nd col name')
parser.add_argument('--third_image_col',type=str, default = "P_MLO", help='if dual input add 3nd col name')
parser.add_argument('--four_image_col',type=str, default = "P_CC", help='if dual input add 4nd col name')

#parser.add_argument('-n' , '--prediction_classes', nargs='+', default =["X1", "Y1", "X2", "Y2"])
parser.add_argument('-n' , '--prediction_classes', nargs='+', default =["Pos", "Neg"])
#parser.add_argument('-n' , '--prediction_classes', nargs='+', default =["six", "eight"])
#parser.add_argument('-n' , '--prediction_classes', nargs='+', default =["DCIS", "INV"])#, "Medium"])
#parser.add_argument('-n' , '--prediction_classes', nargs='+', default =["MAG", "NORM"])
#parser.add_argument('--norm_val',type=str, default = "Stand_Diff", help='if dual input add 2nd col name')

parser.add_argument('--model_save', type=str, default = False, help='save at the end of training')
parser.add_argument('--early_finish', type=str, default = "True", help='early finish if model not improving')
parser.add_argument('--post_analysis_folder', type=str, default = "Cross_analysis_CME" , help='output directory for analysis')
parser.add_argument('--testing_metric', type=str, default = "Standard" , help='where to save the analysis')
parser.add_argument('--load_pretain', type=str, default = True , help='load earlier wieghts')
parser.add_argument('--analysis_type', type=str, default = "test" , help='where to save the analysis')

### for vision transformer

parser.add_argument('--patch_size', type=int, default = 6, help='save at the end of training')
parser.add_argument('--projection_dim', type=int, default = 64, help='save at the end of training')
parser.add_argument('--num_heads', type=int, default = 8, help='save at the end of training')
parser.add_argument('-nargs-int-type' , '--mlp_head_units', nargs='+', default =[2048, 1024])
parser.add_argument('--transformer_layers', type=int, default = 4, help='save at the end of training')





args = parser.parse_args()
print(args)


if args.val_fold == args.test_fold:
    args.mode = None



if os.path.exists(args.post_analysis_folder) == False:
    os.mkdir(args.post_analysis_folder)
Fold_out_analysis = os.path.join(args.post_analysis_folder, str(args.test_fold))
if os.path.exists(Fold_out_analysis) == False:
    os.mkdir(Fold_out_analysis)



data = pd.read_excel(args.excel_file)


if args.connection_type == "Multihead":
    args.batch_size = 16
    
    
if args.mode == "train":
    p = "{}_{}_{}_{}_{}_{}_{}_{}.h5".format(args.connection_type, args.val_fold,args.test_fold, args.repeat, args.img_size, args.max_epochs, args.backbone, args.tag)
    args.wieghts_output_dir = os.path.join("WIEGHTS", p)
    
    train_data, val_data, test_data = create_folds2(args, data)
    print(train_data.shape)
    print(train_data.shape, val_data.shape, test_data.shape)
    
    print('data.shape in loader = ', data.shape)
    
    #AUTOTUNE = tf.data.AUTOTUNE
    
    train_generator = CustomDataGen(train_data, args, mode = "training")
    print(train_generator.df.shape)
    train_generator.pre_check_images()
    print(train_generator.df.shape)
    print('Total data len = ', train_generator.n)
    print('Class Count = ', train_generator.Class_Count())
    if args.use_output_bias:
        classes = list(train_generator.Class_Count())
        args.output_bias = tf.keras.initializers.Constant(np.log([classes[0]/classes[1]]))
    valid_generator = CustomDataGen(val_data, args, mode = "validation")
    valid_generator.pre_check_images()
    print('Total val len = ', valid_generator.n)
    print('Class Count = ', valid_generator.Class_Count())
    if args.features:
        test_generator = CustomDataGen(data, args, mode = "validation", batch=1)
        test_generator.pre_check_images()
        my_model = CustomModel(args)
        my_model.Prep_training(train_generator, valid_generator, test_generator = test_generator)
    else:
        if args.multi_process:
            my_model = CustomModel_Multi(args)
            my_model.Prep_training(train_generator, valid_generator)
        else:
            my_model = CustomModel(args)
            #my_model = CustomModel_Multi(args)
            my_model.Prep_training(train_generator, valid_generator)
    if args.training_type == "Regression":
        my_model.Train_model_regression()
    else:
        my_model.Train_model()
        


if args.mode == "train_finetune":
    p = "{}_{}_{}_{}_{}_{}_{}_{}.h5".format(args.connection_type, args.val_fold,args.test_fold, args.repeat, args.img_size, args.max_epochs, args.backbone, args.tag)
    args.wieghts_output_dir = os.path.join("WIEGHTS2", p)
    p2 = "{}_{}_{}_{}_{}_{}_{}_{}.h5".format(args.connection_type, args.val_fold,args.test_fold, args.repeat, args.img_size, 50, args.backbone, args.tag)
    args.weight_path = os.path.join("WIEGHTS", p2)
    args.load_weight = True
    train_data, val_data, test_data = create_folds(args, data)
    print(train_data.shape)
    print(train_data.shape, val_data.shape, test_data.shape)
    
    print('data.shape in loader = ', data.shape)
    
    #AUTOTUNE = tf.data.AUTOTUNE
    
    train_generator = CustomDataGen(train_data, args, mode = "training")
    print(train_generator.df.shape)
    train_generator.pre_check_images()
    print(train_generator.df.shape)
    print('Total data len = ', train_generator.n)
    print('Class Count = ', train_generator.Class_Count())
    if args.use_output_bias:
        classes = list(train_generator.Class_Count())
        args.output_bias = tf.keras.initializers.Constant(np.log([classes[0]/classes[1]]))
    valid_generator = CustomDataGen(test_data, args, mode = "validation")
    valid_generator.pre_check_images()
    print('Total val len = ', valid_generator.n)
    print('Class Count = ', valid_generator.Class_Count())
    if args.features:
        test_generator = CustomDataGen(data, args, mode = "validation", batch=1)
        test_generator.pre_check_images()
        my_model = CustomModel(args)
        my_model.Prep_training(train_generator, valid_generator, test_generator = test_generator)
    else:
        if args.multi_process:
            my_model = CustomModel_Multi(args)
            my_model.Prep_training(train_generator, valid_generator)
        else:
            my_model = CustomModel(args)
            #my_model = CustomModel_Multi(args)
            my_model.Prep_training(train_generator, valid_generator)
    if args.training_type == "Regression":
        my_model.Train_model_regression()
    else:
        my_model.Train_model()


if args.mode == "inference2":
    p = "{}_{}_{}_{}_{}_{}_{}_{}.h5".format(args.connection_type, args.val_fold,args.test_fold, args.repeat, args.img_size, args.max_epochs, args.backbone, args.tag)
    args.wieghts_output_dir = os.path.join("WIEGHTS", p)
    
    train_data, val_data, test_data = create_folds2(args, data)
    print(train_data.shape)
    print(train_data.shape, val_data.shape, test_data.shape)
    
    print('data.shape in loader = ', data.shape)
    
    #AUTOTUNE = tf.data.AUTOTUNE
    
    train_generator = CustomDataGen(train_data, args, mode = "training")
    print(train_generator.df.shape)
    train_generator.pre_check_images()
    print(train_generator.df.shape)
    print('Total data len = ', train_generator.n)
    print('Class Count = ', train_generator.Class_Count())
    if args.use_output_bias:
        classes = list(train_generator.Class_Count())
        args.output_bias = tf.keras.initializers.Constant(np.log([classes[0]/classes[1]]))
    valid_generator = CustomDataGen(val_data, args, mode = "validation")
    valid_generator.pre_check_images()
    print('Total val len = ', valid_generator.n)
    print('Class Count = ', valid_generator.Class_Count())
    args.load_weight = True
    args.weight_path = os.path.join("WIEGHTS", p)
    if args.multi_process:
        my_model = CustomModel_Multi(args)
        my_model.Prep_training(train_generator, valid_generator)
        Acc = my_model.predict_model()
        print(Acc)
        


    
elif args.mode == "inference":
    
    p = "{}_{}_{}_{}_{}_{}_{}_{}.h5".format(args.connection_type, args.val_fold,args.test_fold, args.repeat, args.img_size, args.max_epochs, args.backbone, args.tag)
    args.weight_path = os.path.join("WIEGHTS", p)
    print(args.weight_path)
    print('data.shape in loader = ', data.shape)
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_data, val_data, test_data = create_folds2(args, data)
    print(train_data.shape, val_data.shape, test_data.shape)
    
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_data = shuffle(train_data)
    train_generator = CustomDataGen(train_data, args, mode = "training")
    train_generator.pre_check_images()
    
    print('Total data len = ', train_generator.n)
    
    print('Class Count = ', train_generator.Class_Count())
    valid_generator = CustomDataGen(test_data, args, mode = "validation")
    valid_generator.pre_check_images()
    print('Total val len = ', valid_generator.n)
    print('Class Count = ', valid_generator.Class_Count())
    args.batch_size = 1
    args.load_weight = True
    args.weight_path = os.path.join("WIEGHTS", p)
    my_model = CustomModel(args)
    my_model.Prep_training(train_generator, valid_generator)
    print('Class weights = ', my_model.Class_weights)
    if args.model_type == 'Classification':
        if args.training_type != "Regression":
            if args.num_input_objects == 4:
                Acc = my_model.predict_model2()
                #ACC = my_model.tensorflow_analysis()
            else:
                #Acc - my_model.tensorflow_analysis()
                Acc = my_model.predict_model()
                #Acc = my_model.predict_model_batch()
                
            #file_p = os.path.join(args.post_analysis_folder, "Run_Record.txt")
            #if os.path.exists(file_p) == False:
                #file1 = open(file_p, "w")
                #L = ["Record of runs starting 25 Jan \n model:{} got ACC:{} on fold {}, image size:{}".format(args.connection_type, Acc, args.val_fold, args.img_size)]
                #file1.writelines(L)
                #file1.close()
            #else:
                #file1 = open(file_p, "a")  # append mode
                #file1.write("\n model:{} got ACC:{} on fold {}, image size:{}".format(args.connection_type, Acc, args.val_fold, args.img_size))
                #file1.close()
        elif args.training_type == "Regression":
            iou = my_model.predict_regression()
    elif args.model_type == 'Segmentation':
        Acc = my_model.collect_imgs()
        
elif args.mode == "inference_out":
    
    p = "{}_{}_{}_{}_{}_{}_{}_{}.h5".format(args.connection_type, args.val_fold,args.test_fold, args.repeat, args.img_size, args.max_epochs, args.backbone, args.tag)
    args.weight_path = os.path.join("WIEGHTS", p)
    print(args.weight_path)
    print('data.shape in loader = ', data.shape)
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_data, val_data, test_data = create_folds2(args, data)
    print(train_data.shape, val_data.shape, test_data.shape)
    
    
    AUTOTUNE = tf.data.AUTOTUNE
    #train_data = shuffle(train_data)
    train_generator = CustomDataGen(test_data, args, mode = "validation")
    train_generator.pre_check_images()
    
    print('Total data len = ', train_generator.n)
    
    print('Class Count = ', train_generator.Class_Count())
    valid_generator = CustomDataGen(val_data, args, mode = "validation")
    valid_generator.pre_check_images()
    print('Total val len = ', valid_generator.n)
    print('Class Count = ', valid_generator.Class_Count())
    args.batch_size = 1
    args.load_weight = True
    args.weight_path = os.path.join("WIEGHTS", p)
    my_model = CustomModel(args)
    my_model.Prep_training(train_generator, valid_generator)
    print('Class weights = ', my_model.Class_weights)
    Acc_test, trscore, tmscore, tadmscore, roc_score_test = my_model.predict_model_train()
    Acc_val, vrscore, vmscore, vadmscore, roc_score_val = my_model.predict_model_val()
    excel_p = os.path.join(args.post_analysis_folder, "results.xlsx")
    if os.path.exists(excel_p) == False:
        d = {'Backbone':[args.backbone] , "val_fold": [args.val_fold], "test_fold": [args.test_fold], "repeat":[args.repeat], 'Val': [Acc_val] , "Test":[Acc_test], "Randval":[vrscore], "Randtest":[trscore], "Mscoreval":[vmscore],"Mscoretest":[tmscore], "AdjMscoreval":[vadmscore], "AdjMscoretest":[tadmscore], "Rocval":[roc_score_val], "Roctest":[roc_score_test]}
        df = pd.DataFrame.from_dict(d)
        df.to_excel(excel_p)
    else:
        df = pd.read_excel(excel_p)
        df = df.drop(df.columns[0],axis=1)
        d = {'Backbone':[args.backbone] ,"val_fold": [args.val_fold], "test_fold": [args.test_fold], "repeat":[args.repeat], 'Val': [Acc_val] , "Test":[Acc_test], "Randval":[vrscore], "Randtest":[trscore], "Mscoreval":[vmscore],"Mscoretest":[tmscore], "AdjMscoreval":[vadmscore], "AdjMscoretest":[tadmscore], "Rocval":[roc_score_val], "Roctest":[roc_score_test]}
        data = pd.DataFrame.from_dict(d)
        df = pd.concat([df, data], axis=0)
        df.to_excel(excel_p)
        
        
    
   
        
elif args.mode == "prediction_comparision":
    
    p = "{}_{}_{}_{}_{}_{}_{}_{}.h5".format(args.connection_type, args.val_fold,args.test_fold, args.repeat, args.img_size, args.max_epochs, args.backbone, args.tag)
    args.weight_path = os.path.join("WIEGHTS", p)
    print(args.weight_path)
    print('data.shape in loader = ', data.shape)
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_data, val_data, test_data = create_folds(args, data)
    print(train_data.shape, val_data.shape, test_data.shape)
    
    
    AUTOTUNE = tf.data.AUTOTUNE
    
    train_generator = CustomDataGen(train_data, args, mode = "training")
    train_generator.pre_check_images()
    
    print('Total data len = ', train_generator.n)
    
    print('Class Count = ', train_generator.Class_Count())
    valid_generator = CustomDataGen(test_data, args, mode = "validation")
    valid_generator.pre_check_images()
    print('Total val len = ', valid_generator.n)
    print('Class Count = ', valid_generator.Class_Count())
    args.batch_size = 1
    args.weight_path = os.path.join("WIEGHTS", p)
    my_model = CustomModel(args)
    my_model.Prep_training(train_generator, valid_generator)
    print('Class weights = ', my_model.Class_weights)
    if args.model_type == 'Classification':
        Acc = my_model.prediction_fixtures()
    
elif args.mode == "fold_testing":
    Ground = []
    prediction = []
    Names = []
    scale = int(round(data.shape[0]/5))
    for fod in range(0, args.Num_folds):
        train_data, val_data, test_data = create_folds(args.Num_folds, fod, data)
        args.batch_size = 1
        p = "{}_{}_{}_{}_{}_{}_{}.h5".format(args.connection_type, fod, args.repeat, args.img_size, args.max_epochs, args.backbone, args.tag)
        args.weight_path = os.path.join("WIEGHTS", p)
        args.load_weight = True
        valid_generator = CustomDataGen(test_data, args, mode = "validation")
        valid_generator.pre_check_images()
        print('Total val len = ', valid_generator.n)
        my_model = CustomModel(args)
        my_model.Prep_training(valid_generator, valid_generator)
        print('Class weights = ', my_model.Class_weights)
        Val_GRON, Val_PRED, Nam = my_model.predict_model_fold()
        Names.append(Nam)
        Ground.append(Val_GRON), prediction.append(Val_PRED)
    Ground = np.concatenate(Ground)
    prediction = np.concatenate(prediction)
    Names = np.concatenate(Names)
    prediction = prediction.astype(np.uint8)
    Score = []
    
    for row in range(0, data.shape[0]):
        nam = str(data[args.image_col].iloc[row])
        try:
            pos = list(Names).index(nam)
            if (Ground[pos,:] == prediction[pos,:]).all():
                Score.append(1)
            else:
                Score.append(0)
        except:
            Score.append(5)
        
        
    data["Score"] = Score
    #data = data[data.Score != 5]
    #data.to_excel("Size_information.xlsx")
        
    #ACC = my_model.predict_model_analysis(Ground, prediction)
    
    
elif args.mode == "folds_to_excel":
    Names = []
    Acc_list = []
    DC = []
    IN = []
    scale = int(round(data.shape[0]/5))
    for fod in range(0, args.Num_folds):
        test_data = data.iloc[scale*fod:scale*(fod+1)]
        args.batch_size = 1
        p = "{}_{}_{}_{}_{}_{}_{}.h5".format(args.connection_type, args.fold, args.repeat, args.img_size, args.max_epochs, args.backbone, args.tag)
        args.weight_path = os.path.join("WIEGHTS", p)
        valid_generator = CustomDataGen(test_data, args, mode = "validation")
        valid_generator.pre_check_images()
        print('Total val len = ', valid_generator.n)
        my_model = CustomModel(args)
        my_model.Prep_training(valid_generator, valid_generator)
        print('Class weights = ', my_model.Class_weights)
        NAMES, ACC_LIST, INV, DCIS = my_model.predict_model_excel()
        Names.append(NAMES), Acc_list.append(ACC_LIST)
        DC.append(DCIS)
        IN.append(INV)
    Names = list(itertools.chain(*Names))
    Acc_list = list(itertools.chain(*Acc_list))
    DC = list(itertools.chain(*DC))
    IN = list(itertools.chain(*IN))
    frame = pd.DataFrame()
    frame["Names"] = Names
    frame["prediction"] = Acc_list
    frame["DCIS"] = DC
    frame["IN"] = IN
    frame.to_excel("{}_{}.xlsx".format(args.connection_type, args.img_size))
    
elif args.mode == "features":
    
    p = "{}_{}_{}_{}_{}_{}_{}_{}.h5".format(args.connection_type, args.val_fold,args.test_fold, args.repeat, args.img_size, args.max_epochs, args.backbone, args.tag)
    args.weight_path = os.path.join("WIEGHTS2", p)
    print(args.weight_path)
    print('data.shape in loader = ', data.shape)
    
    AUTOTUNE = tf.data.AUTOTUNE
    
    
    AUTOTUNE = tf.data.AUTOTUNE
    
    train_generator = CustomDataGen(data, args, mode = "training")
    train_generator.pre_check_images()
    args.load_weight = True
    print('Total data len = ', train_generator.n)
    
    print('Class Count = ', train_generator.Class_Count())
    valid_generator = CustomDataGen(data, args, mode = "validation")
    valid_generator.pre_check_images()
    print('Total val len = ', valid_generator.n)
    print('Class Count = ', valid_generator.Class_Count())
    args.batch_size = 1
    args.weight_path = os.path.join("WIEGHTS2", p)
    my_model = CustomModel(args)
    my_model.Prep_training(train_generator, valid_generator)
    print('Class weights = ', my_model.Class_weights)
    Extract = my_model.Extract_features()
    
elif args.mode == "threshold_model":
    
    p = "{}_{}_{}_{}_{}_{}_{}.h5".format(args.connection_type, args.fold, args.repeat, args.img_size, args.max_epochs, args.backbone, args.tag)
    args.weight_path = os.path.join("WIEGHTS", p)
    print(args.weight_path)
    print('data.shape in loader = ', data.shape)
    train_data, val_data, test_data = create_folds(args.Num_folds, args.fold, data)
    print(train_data.shape, val_data.shape, test_data.shape)
    AUTOTUNE = tf.data.AUTOTUNE
    training_data = pd.concat([train_data, val_data], axis=0)
    
    train_generator = CustomDataGen(training_data, args, mode = "training")
    train_generator.pre_check_images()
    
    print('Total data len = ', train_generator.n)
    
    print('Class Count = ', train_generator.Class_Count())
    valid_generator = CustomDataGen(test_data, args, mode = "validation")
    valid_generator.pre_check_images()
    print('Total val len = ', valid_generator.n)
    print('Class Count = ', valid_generator.Class_Count())
    args.batch_size = 1
    args.weight_path = os.path.join("WIEGHTS", p)
    my_model = CustomModel(args)
    my_model.Prep_training(train_generator, valid_generator)
    print('Class weights = ', my_model.Class_weights)
    Extract, tag = my_model.Extract_features_model()
    thresh_extract = Extract[(Extract[tag] <= args.threashold)]
    additional_training_data = Extract[(Extract[tag] > args.threashold) & (Extract["belong_set"] == 1)]
    print(thresh_extract.shape)
    train_data2 = thresh_extract[(thresh_extract["belong_set"] == 1)]
    test_data2 = thresh_extract[(thresh_extract["belong_set"] == 0)]
    print(train_data2.shape, test_data2.shape)
    #gap = 1000 - train_data2.shape[0]
    additional_training_data = additional_training_data.iloc[:,:train_data2.shape[0]]
    train_data2 = pd.concat([train_data2, additional_training_data], axis=0)
    train_data2 = shuffle(train_data2)
    print(train_data2.shape, test_data2.shape)
    split = int(round(len(train_data2)*0.8,0))
    train, val = train_data2[:split], train_data2[split:]
    testing_data = Extract[(Extract["belong_set"] == 0)]
    #print(testing_data.shape)
    
    #Val_PRED = np.array(testing_data.loc[:, [args.prediction_classes[0]+ "_pred", args.prediction_classes[1]+ "_pred"]])
    #Val_GRON = np.array(testing_data.loc[:, [args.prediction_classes[0], args.prediction_classes[1]]])
    #ACC = accuracy_score(Val_GRON, Val_PRED)
    #print(ACC)
    args.batch_size = 16
    args.repeat = "{}_{}".format(args.repeat,args.repeat)
    d = "{}_{}_{}_{}_{}_{}_{}.h5".format(args.connection_type, args.fold, args.repeat, args.img_size, args.max_epochs, args.backbone, args.tag)
    args.wieghts_output_dir = os.path.join("WIEGHTS", d)
    args.weight_path = "None"
    train_generator = CustomDataGen(train, args, mode = "training")
    train_generator.pre_check_images()
    print('Total data len = ', train_generator.n)
    
    print('Class Count = ', train_generator.Class_Count())
    valid_generator = CustomDataGen(val, args, mode = "validation")
    valid_generator.pre_check_images()
    print('Total val len = ', valid_generator.n)
    print('Class Count = ', valid_generator.Class_Count())
    my_model = CustomModel(args)
    my_model.Prep_training(train_generator, valid_generator)
    my_model.Train_model()
    train.to_excel("Train.xlsx")
    val.to_excel("Val.xlsx")
    testing_data.to_excel("Test.xlsx")
    test_data2.to_excel("Test2.xlsx")

elif args.mode == "threshold_testing":
    
    p = "{}_{}_{}_{}_{}_{}_{}.h5".format(args.connection_type, args.fold, args.repeat, args.img_size, args.max_epochs, args.backbone, args.tag)
    args.weight_path = os.path.join("WIEGHTS", p)
    print(args.weight_path)
    print('data.shape in loader = ', data.shape)
    train_data, val_data, test_data = create_folds(args.Num_folds, args.fold, data)
    print(train_data.shape, val_data.shape, test_data.shape)
    AUTOTUNE = tf.data.AUTOTUNE
    training_data = pd.concat([train_data, val_data], axis=0)
    
    train_generator = CustomDataGen(training_data, args, mode = "training")
    train_generator.pre_check_images()
    
    print('Total data len = ', train_generator.n)
    
    print('Class Count = ', train_generator.Class_Count())
    valid_generator = CustomDataGen(test_data, args, mode = "validation")
    valid_generator.pre_check_images()
    print('Total val len = ', valid_generator.n)
    print('Class Count = ', valid_generator.Class_Count())
    args.batch_size = 1
    args.weight_path = os.path.join("WIEGHTS", p)
    my_model = CustomModel(args)
    my_model.Prep_training(train_generator, valid_generator)
    print('Class weights = ', my_model.Class_weights)
    Test = my_model.Extract_features_threasholding()

elif args.mode == "thresh_inference":
    args.repeat = "{}_{}".format(args.repeat,args.repeat)
    p = "{}_{}_{}_{}_{}_{}_{}.h5".format(args.connection_type, args.fold, args.repeat, args.img_size, args.max_epochs, args.backbone, args.tag)
    args.weight_path = os.path.join("WIEGHTS", p)
    print(args.weight_path)
    print('data.shape in loader = ', data.shape)
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_data, val_data, test_data = pd.read_excel("Train.xlsx"), pd.read_excel("Val.xlsx"), pd.read_excel("Test.xlsx")
    print(train_data.shape, val_data.shape, test_data.shape)
    
    
    AUTOTUNE = tf.data.AUTOTUNE
    
    train_generator = CustomDataGen(train_data, args, mode = "training")
    train_generator.pre_check_images()
    
    print('Total data len = ', train_generator.n)
    
    print('Class Count = ', train_generator.Class_Count())
    valid_generator = CustomDataGen(test_data, args, mode = "validation")
    valid_generator.pre_check_images()
    print('Total val len = ', valid_generator.n)
    print('Class Count = ', valid_generator.Class_Count())
    args.batch_size = 1
    args.weight_path = os.path.join("WIEGHTS", p)
    my_model = CustomModel(args)
    my_model.Prep_training(train_generator, valid_generator)
    print('Class weights = ', my_model.Class_weights)
    if args.model_type == 'Classification':
        if args.num_input_objects == 6:
            Acc = my_model.predict_model2()
        else:
            Acc = my_model.predict_model()
        file_p = os.path.join(args.post_analysis_folder, "Run_Record.txt")
        if os.path.exists(file_p) == False:
            file1 = open(file_p, "w")
            L = ["Record of runs starting 25 Jan \n model:{} got ACC:{} on fold {}, image size:{}".format(args.connection_type, Acc, args.fold, args.img_size)]
            file1.writelines(L)
            file1.close()
        else:
            file1 = open(file_p, "a")  # append mode
            file1.write("\n model:{} got ACC:{} on fold {}, image size:{}".format(args.connection_type, Acc, args.fold, args.img_size))
            file1.close()
    elif args.model_type == 'Segmentation':
        Acc = my_model.collect_imgs()    
    
    
    
    

elif args.mode == "collect_imgs":
    
    p = "{}_{}_{}_{}_{}_{}.h5".format(args.connection_type, args.fold, args.repeat, args.img_size, args.max_epochs, args.backbone)
    args.weight_path = os.path.join("WIEGHTS", p)
    print(args.weight_path)
    print('data.shape in loader = ', data.shape)
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_data, val_data, test_data = create_folds(args.Num_folds, args.fold, data)
    print(train_data.shape, val_data.shape, test_data.shape)
    
    
    AUTOTUNE = tf.data.AUTOTUNE
    
    train_generator = CustomDataGen(val_data, args, mode = "training")
    train_generator.pre_check_images()
    
    print('Total data len = ', train_generator.n)
    
    print('Class Count = ', train_generator.Class_Count())
    valid_generator = CustomDataGen(test_data, args, mode = "validation")
    valid_generator.pre_check_images()
    print('Total val len = ', valid_generator.n)
    print('Class Count = ', valid_generator.Class_Count())
    args.batch_size = 1
    args.weight_path = os.path.join("WIEGHTS", p)
    my_model = CustomModel(args)
    my_model.Prep_training(train_generator, valid_generator)
    print('Class weights = ', my_model.Class_weights)
    base = "DATASET/All_images"
    Img_list = os.listdir(base)
    predict_list = my_model.predict_images_from_Dictory(Img_list, base)
    
elif args.mode == "attention_maps":
    #tf.compat.v1.disable_eager_execution()
    
    args.batch_size = 1
    p = "{}_{}_{}_{}_{}_{}_{}_{}.h5".format(args.connection_type, args.val_fold,args.test_fold, args.repeat, args.img_size, args.max_epochs, args.backbone, args.tag)
    args.weight_path = os.path.join("WIEGHTS", p)
    print(args.weight_path)
    print('data.shape in loader = ', data.shape)
    
    #AUTOTUNE = tf.data.AUTOTUNE
    train_data, val_data, test_data = create_folds(args, data)
    print(train_data.shape, val_data.shape, test_data.shape)
    
    
    #AUTOTUNE = tf.data.AUTOTUNE
    
    train_generator = CustomDataGen(val_data, args, mode = "training")
    train_generator.pre_check_images()
    
    print('Total data len = ', train_generator.n)
    
    print('Class Count = ', train_generator.Class_Count())
    valid_generator = CustomDataGen(test_data, args, mode = "validation")
    valid_generator.pre_check_images()
    print('Total val len = ', valid_generator.n)
    print('Class Count = ', valid_generator.Class_Count())
    args.weight_path = os.path.join("WIEGHTS", p)
    my_model = CustomModel(args)
    my_model.Prep_training(train_generator, valid_generator)
    tf.config.run_functions_eagerly(True)
    index=0
    create_attention_maps = my_model.run_hi_res_cam2(index)       
    
elif args.mode == "fold_checker":
    checker = []
    val_checker = []
    white = []
    white2 = []
    for row in range(0, data.shape[0]):
        name = data[args.image_col].iloc[row]
        count = args.Num_folds + 1
        step = 0
        for ii in range(0, args.Num_folds):
            
            train_data, val_data, test_data = create_folds(args.Num_folds, ii, data)
            test_list = list(test_data[args.image_col])
            if name in test_list:
                count = ii
            
        checker.append(count)
        val_checker.append(step)

    data["fold"] = checker
    
    for row in range(0, data.shape[0]):
        name = str(data[args.image_col].iloc[row])
        img_path = os.path.join(args.dataset_path_mask, name + args.ext_type_mask)
        img = cv2.imread(img_path,0)
        size = img.shape[0] * img.shape[1]
        POS = np.where(img > 0)
        scale = round(POS[0].shape[0]/size, 3)
        white.append(scale)
    for row in range(0, data.shape[0]):
        name = str(data[args.image_col].iloc[row])
        img_path = os.path.join(args.dataset_path_mask.replace("2", "3"), name + args.ext_type_mask)
        img = cv2.imread(img_path,0)
        size = img.shape[0] * img.shape[1]
        POS = np.where(img > 0)
        scale = round(POS[0].shape[0]/size, 3)
        white2.append(scale)
    
    data["fold"] = checker
    data["region"] = white
    
    data["thresh_region"] = white2
    data = data[data.fold != args.Num_folds + 1]
    df = pd.read_excel("Size_information.xlsx")
    name_list = list(df[args.image_col])
    score = []
    for row in range(0, data.shape[0]):
        name = data[args.image_col].iloc[row]
        pos = name_list.index(name)
        score.append(df["Score"].iloc[pos])
    data["Score"] = score
    data.to_excel("Size_information2.xlsx")
if args.mode == "test":
    strategy = tf.distribute.MirroredStrategy()
    p = "{}_{}_{}_{}_{}_{}_{}_{}.h5".format(args.connection_type, args.val_fold,args.test_fold, args.repeat, args.img_size, args.max_epochs, args.backbone, args.tag)
    args.wieghts_output_dir = os.path.join("WIEGHTS", p)

    train_data, val_data, test_data = create_folds(args, data)
    print(train_data.shape)
    print(train_data.shape, val_data.shape, test_data.shape)
    
    print('data.shape in loader = ', data.shape)
    
    #AUTOTUNE = tf.data.AUTOTUNE
    
    train_generator = CustomDataGen(train_data, args, mode = "training")
    print(train_generator.df.shape)
    train_generator.pre_check_images()
    Train_data = tf.data.Dataset.from_generator(train_generator, output_types=(tf.float64, tf.int64))
    train_dist_dataset = strategy.experimental_distribute_dataset(Train_data)
    count = 0
    for materials, labels in train_dist_dataset:
        print(labels)
        count +=1 
    print(count)           

        
    
    

