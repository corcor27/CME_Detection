import tensorflow as tf
import numpy as np
import cv2
import os
import imgaug.augmenters as iaa
import random
from sklearn.utils import shuffle
import pandas as pd
import matplotlib.pyplot as plt


class CustomDataGen(tf.keras.utils.Sequence):
    def __init__(self,  df, args, mode, batch = 0):
        """
        Intialize dataloader, that returns batches of images
        Inputs; df: Dataframe of images, args: training arguements, mode: dataloader mode

        Returns
        -------
        None.

        """
        self.df = df.copy()
        self.i = 0
        self.args = args
        self.mode = mode
        self.n = len(self.df)
        if mode == "training" and self.args.aug == True:
            self.augmentation = True
        else:
            self.augmentation = False
        if self.args.features and batch == 1:
            self.args.batch_size = 1
            
            
    def pre_check_images(self):
        """
        Checks whether all sample paths exist, appends dataframe by removing broken samples.
        Also intializes any custom data training method, e.g. balance where dataset ratio is equaled between samples and altered every epoch

        Returns
        -------
        None.

        """
        
        rem_ind_lst = []
        if self.args.num_input_objects == 1:
            print("running single input image mode")
        elif self.args.num_input_objects > 1:
            print("running dual input image mode")
        for ind in range(self.df.shape[0]):
            IMG_PATH = os.path.join(self.args.dataset_path_img, str(self.df[self.args.image_col].iloc[ind]) + self.args.ext_type)
            if not os.path.exists(IMG_PATH):
                rem_ind_lst.append(ind)
                continue
            if self.args.model_type == 'Segmentation':
                Mask_PATH = os.path.join(self.args.dataset_path_mask, str(self.df[self.args.image_col].iloc[ind]) + self.args.ext_type_mask)
                if not os.path.exists(Mask_PATH):
                    rem_ind_lst.append(ind)
                    continue
            if self.args.num_input_objects == 2 or self.args.num_input_objects == 6:

                IMG_PATH = os.path.join(self.args.dataset_path_img, str(self.df[self.args.second_image_col].iloc[ind]) + self.args.ext_type)
                if not os.path.exists(IMG_PATH):
                    if ind not in rem_ind_lst:
                        rem_ind_lst.append(ind)
                        continue
                if self.args.model_type == 'Segmentation':
                    Mask_PATH = os.path.join(self.args.dataset_path_mask, str(self.df[self.args.second_image_col].iloc[ind]) + self.args.ext_type_mask)
                    if not os.path.exists(Mask_PATH):
                        rem_ind_lst.append(ind)
                        continue
            if self.args.num_input_objects == 4:
                IMG_PATH = os.path.join(self.args.dataset_path_img, str(self.df[self.args.second_image_col].iloc[ind]) + self.args.ext_type)
                if not os.path.exists(IMG_PATH):
                    if ind not in rem_ind_lst:
                        rem_ind_lst.append(ind)
                        continue
                    
                IMG_PATH = os.path.join(self.args.dataset_path_img, str(self.df[self.args.third_image_col].iloc[ind]) + self.args.ext_type)
                if not os.path.exists(IMG_PATH):
                    if ind not in rem_ind_lst:
                        rem_ind_lst.append(ind)
                        continue
                IMG_PATH = os.path.join(self.args.dataset_path_img, str(self.df[self.args.four_image_col].iloc[ind]) + self.args.ext_type)
                if not os.path.exists(IMG_PATH):
                    if ind not in rem_ind_lst:
                        rem_ind_lst.append(ind)
                        continue
                

        self.df.drop(self.df.index[rem_ind_lst], inplace=True)
        self.n = len(self.df)  # append length of dataset list
        if self.args.training_type == "split" and self.mode == "training":
            self.df1 = self.df.copy()
            self.val = int(round(self.df1.shape[0]/self.args.warmup_epochs,0)) - 1
            self.length = int(round(1*self.val,0))
            self.df = self.df.iloc[0:self.length]
            print("THIS is the image size {}".format(self.df.shape[0]))
            self.n = len(self.df)
        elif self.args.training_type == "balanced" and self.mode == "training":
            values = []
            for item in self.args.prediction_classes:
                values.append(sum(list(self.df[item])))
            max_index = values.index(max(values))
            if max_index == 0:
                self.df1 = self.df.copy()
                df_min = self.df1[(self.df1[self.args.prediction_classes[1]] == 1)]
                df_max = self.df1[(self.df1[self.args.prediction_classes[0]] == 1)]
                self.length = len(df_min)
                scale_df_max = df_max.iloc[:self.length]
                self.df = pd.concat([df_min, scale_df_max])
                self.df = shuffle(self.df)
                self.n = len(self.df)
                
            else:
                self.df1 = self.df.copy()
                df_min = self.df1[(self.df1[self.args.prediction_classes[0]] == 1)]
                df_max = self.df1[(self.df1[self.args.prediction_classes[1]] == 1)]
                self.length = len(df_min)
                scale_df_max = df_max.iloc[:self.length]
                self.df = pd.concat([df_min, scale_df_max])
                self.df = shuffle(self.df)
                self.n = len(self.df)
            
            
            
                
        

    def on_epoch_end(self):
        """
        Updates dataframe based on any custom training configuration as described at the end of __init__
        Inputs: Epoch
        Returns
        -------
        Updated dataframe

        """
        print('end of epoch')
        if self.args.shuffle:
            if self.args.training_type == "balanced" and self.mode == "training":
                values = []
                for item in self.args.prediction_classes:
                    values.append(sum(list(self.df1[item])))
                max_index = values.index(max(values))
                if max_index == 0:
                    df_min = self.df1[(self.df1[self.args.prediction_classes[1]] ==1)]
                    df_max = self.df1[(self.df1[self.args.prediction_classes[0]] ==1)]
                    df_max = shuffle(df_max)
                    scale_df_max = df_max.iloc[:self.length]
                    self.df = pd.concat([df_min, scale_df_max])
                    self.df = shuffle(self.df)
                    
                else:
                    df_min = self.df1[(self.df1[self.args.prediction_classes[0]] ==1)]
                    df_max = self.df1[(self.df1[self.args.prediction_classes[1]] ==1)]
                    df_max = shuffle(df_max)
                    scale_df_max = df_max.iloc[:self.length]
                    self.df = pd.concat([df_min, scale_df_max])
                    self.df = shuffle(self.df)
            else:
                self.df = shuffle(self.df)
                self.n = len(self.df)

    def __get_input_image(self, path):
        """
        Reads in any input image, be it numpy or image file using opencv or numpy 

        Returns
        -------
        Normalised Image

        """
        if self.args.image_format == "image":

            if self.args.img_colour:
                image = cv2.imread(path)
                org_height, org_width, nc = image.shape
            else:
                image = cv2.imread(path, 0)
                image = np.expand_dims(image, axis=-1)
                org_height, org_width, nc = image.shape

        elif self.args.image_format == "numpy":
            if self.args.img_colour:
                image = np.load(path)
                image = image/np.max(image)
                image = image*255
                image = image.astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                org_height, org_width, nc = image.shape
            else:
                image = np.load(path)
                image = image/np.max(image)
                image = np.expand_dims(image, axis=-1)
                org_height, org_width, nc = image.shape

        image_r = cv2.resize(image, (self.args.img_size, self.args.img_size),
                             interpolation=cv2.INTER_CUBIC)  # set to 256x256

        image_r = image_r/np.max(image_r)

        return image_r, org_height, org_width

    def Class_Count(self):
        
        """
        Counts class, for dataset ratio weights used in training. Calculated by counting the classes
        Returns
        -------
        dataset ratio weights

        """
        
        if self.args.model_type == 'Classification':
            C_count = np.zeros(len(self.args.prediction_classes))
            for ind in range(len(self.args.prediction_classes)):
                C_count[ind] = self.df[self.df[self.args.prediction_classes[ind]] == 1].shape[0]
            return C_count
        elif self.args.model_type == 'tool':
            left_zeros = []
            right_zeros = []
            left_ones = []
            right_ones = []
            C_count = np.zeros((self.args.num_output_objects, self.args.num_output_classes + 1))
            for ind in range(self.df.shape[0]):
                MASK_PATH = os.path.join(self.args.dataset_path_mask, str(self.df[self.args.image_col].iloc[ind]) + self.args.ext_type_mask)
                mask = self.__get_input_mask(MASK_PATH)

                area = mask.shape[0]*mask.shape[1]
                pos_area = np.sum(mask)
                left_zeros.append((area - pos_area)/area)
                left_ones.append(pos_area/area)
                
                if self.args.num_output_objects == 2:
                    MASK_PATH = os.path.join(self.args.dataset_path_mask, str(self.df[self.args.second_image_col].iloc[ind]) + self.args.ext_type_mask)
                    mask = self.__get_input_mask(MASK_PATH)
                    area = mask.shape[0]*mask.shape[1]
                    pos_area = np.sum(mask)
                    right_zeros.append((area - pos_area)/area)
                    
                    right_ones.append(pos_area/area)
            C_count[0,0] = np.mean(left_zeros)
            C_count[0,1] = np.mean(left_ones)
            C_count[1,0] = np.mean(right_zeros)
            C_count[1,1] = np.mean(right_ones)
            
            return C_count
        else:
            return np.zeros(len(self.args.prediction_classes))
            
    def __get_input_mask(self, path, mask_label):
        """
        Reads in any input mask, be it numpy or image file using opencv or numpy 

        Returns
        -------
        Binary Mask
        """

        mask = cv2.imread(path, 0)
    
        mask = cv2.resize(mask, (self.args.img_size, self.args.img_size),
                              interpolation=cv2.INTER_CUBIC)  # set to 256x256
        #plt.hist(mask)
        mask = mask/255
        #mask = mask.astype(np.uint8)
        mask[mask > 0] = 1
        mask = mask.astype(np.float32)
        #base  = np.zeros((mask.shape[0], mask.shape[1], len(self.args.prediction_classes)+1), dtype=np.float32)
        #base[:,:, mask_label] = mask
        mask = np.expand_dims(mask, axis=-1)
        #plt.hist(mask)
        #plt.imshow(mask[:,:,0])
        return mask
    
    def __get_input_mask_regression(self, path):
        
        """
        Reads in any input mask, be it numpy or image file using opencv or numpy 

        Returns
        -------
        Binary Mask
        """

        mask = cv2.imread(path, 0)
    
        mask = cv2.resize(mask, (self.args.img_size, self.args.img_size),
                              interpolation=cv2.INTER_CUBIC)  # set to 256x256
        #plt.hist(mask)
        mask = mask/255
        #mask = mask.astype(np.uint8)
        #mask[mask > 0] = 1
        mask = mask.astype(np.float32)
        #mask = np.expand_dims(mask, axis=-1)
        
        return mask


    # Augmentation Horizontal Flip
    def __aug_flip_hr(self, img):
        hflip = iaa.Fliplr(p=1.0)
        img_hf = hflip.augment_image(img)

        return img_hf

    # Augmentation Vertical Flip
    def __aug_flip_vr(self, img):
        vflip = iaa.Flipud(p=1.0)
        img_vf = vflip.augment_image(img)

        return img_vf

    # Augmentation Rotation
    def __aug_rotation90(self, img, rot_deg):
        rot1 = iaa.Affine(rotate=rot_deg)
        img_rot = rot1.augment_image(img)

        return img_rot
    def __aug_rotationminus90(self, img, rot_deg):
        rot1 = iaa.Affine(rotate=-rot_deg)
        img_rot = rot1.augment_image(img)

        return img_rot

    # Augmentation Cropping
    def __aug_crop(self, img, crop_ratio):
        crop1 = iaa.Crop(percent=(0, crop_ratio))
        img_crop = crop1.augment_image(img)

        return img_crop

    # Augmentation Adding noise
    def __aug_add_noise(self, img, mean_noise, var_noise):
        noise = iaa.AdditiveGaussianNoise(mean_noise, var_noise)
        img = img * 255
        img = img.astype(np.uint8)
        img_noise = noise.augment_image(img)
        img_noise = img_noise / 255
        return img_noise

    # Augmentation Shear
    def __aug_shear(self, img, shear_deg):
        shearX = iaa.ShearX((-shear_deg, shear_deg))
        img_shear = shearX.augment_image(img)

        shearY = iaa.ShearY((-shear_deg, shear_deg))
        img_shear = shearY.augment_image(img_shear)

        return img_shear

    # Augmentation Translation
    def __aug_translation(self, img, trans_pix):
        TransX = iaa.TranslateX(px=(-trans_pix, trans_pix))
        img_trans = TransX.augment_image(img)

        TransY = iaa.TranslateY(px=(-trans_pix, trans_pix))
        img_trans = TransY.augment_image(img_trans)

        return img_trans

    # Augmentation Scale
    def __aug_scale(self, img, scale_ratio):
        ScaleX = iaa.ScaleX((scale_ratio, 3 * scale_ratio))
        img_scale = ScaleX.augment_image(img)
        ScaleY = iaa.ScaleY((scale_ratio, 3 * scale_ratio))
        img_scale = ScaleY.augment_image(img_scale)

        return img_scale

    def batch_augmentation(self, batch_img, random_number):
        
        for ind in range(batch_img.shape[0]):
            if self.args.img_colour == True:
                img = batch_img[ind, :, :, :]
            else:
                img = batch_img[ind, :, :]
            if random_number == 1:
                img = self.__aug_flip_hr(img)
            elif random_number == 2:
                img = self.__aug_flip_vr(img)
            elif random_number == 3:
                img = self.__aug_rotation90(img, 90)
            elif random_number == 4:
                img = self.__aug_rotationminus90(img, 90)
            if self.args.img_colour == True:
                batch_img[ind, :, :, :] = img
            else:
                batch_img[ind, :, :] = img

            batch_img[ind, :, :] = img

        return batch_img
    """
    
    Collects a allocated bactch of data based on number of inputs and indexed value. 
    
    
    """
    def __getitem__(self, index):

        batches = self.df[index *self.args.batch_size:(index + 1) * self.args.batch_size]
        images = []
        images2 = []
        images3 = []
        images4 = []
        labels = []
        labels2 = []
        norm = []
        bbox = []
        bbox2 = []
        for ind in range(batches.shape[0]):
            if self.args.model_type == "Classification":
                
                IMG_PATH1 = os.path.join(self.args.dataset_path_img, str(batches[self.args.image_col].iloc[ind])+ self.args.ext_type)
                _img, org_height, org_width = self.__get_input_image(IMG_PATH1)
                images.append(_img)
                    
                if self.args.num_input_objects > 1:
                    IMG_PATH2 = os.path.join(self.args.dataset_path_img, str(batches[self.args.second_image_col].iloc[ind]) + self.args.ext_type)
                    _img2, org_height2, org_width2 = self.__get_input_image(IMG_PATH2)
                    images2.append(_img2)
                elif self.args.num_input_objects > 2:
                    IMG_PATH3 = os.path.join(self.args.dataset_path_img, str(batches[self.args.third_image_col].iloc[ind]) + self.args.ext_type)
                    _img3, org_height3, org_width3 = self.__get_input_image(IMG_PATH3)
                    images3.append(_img3)
                elif self.args.num_input_objects > 3:
                    IMG_PATH4 = os.path.join(self.args.dataset_path_img, str(batches[self.args.four_image_col].iloc[ind]) + self.args.ext_type)
                    _img4, org_height4, org_width4 = self.__get_input_image(IMG_PATH4)
                    images4.append(_img4)
                    #norm.append([batches[self.args.norm_val].iloc[ind]])
                if self.args.training_type != "Regression":
                    labels.append([batches[self.args.prediction_classes].iloc[ind].values])
                    if self.args.num_output_objects > 1:
                        labels2.append([batches[self.args.prediction_classes].iloc[ind].values])
                else:
                    xmin,zmin,xmax,zmax = batches[self.args.prediction_classes].iloc[ind].values
                    zmin,xmin,zmax,xmax = round(zmin/org_height,2), round(xmin/org_width,2), round(zmax/org_height,2), round(xmax/org_width,2)
                    labels.append([zmin,xmin,zmax,xmax])
                
            elif self.args.model_type == 'Segmentation':
                IMG_PATH1 = os.path.join(self.args.dataset_path_img, str(batches[self.args.image_col].iloc[ind])+ self.args.ext_type)
                _img, org_height, org_width = self.__get_input_image(IMG_PATH1)
                images.append(_img)
                if self.args.num_input_objects == 2:
                    IMG_PATH2 = os.path.join(self.args.dataset_path_img, str(batches[self.args.second_image_col].iloc[ind]) + self.args.ext_type)
                    _img2, org_height2, org_width2 = self.__get_input_image(IMG_PATH2)
                    images2.append(_img2)
                MASK_PATH1 = os.path.join(self.args.dataset_path_mask, str(batches[self.args.image_col].iloc[ind])+ self.args.ext_type_mask)
                
                CLASS_LIST = [batches[self.args.prediction_classes].iloc[ind].values]
                CLASS_LIST = np.array(CLASS_LIST)
                mask_label = np.argmax(CLASS_LIST) + 1
                _label = self.__get_input_mask(MASK_PATH1, mask_label)
                _label = np.squeeze(_label)
                if self.args.training_type == "Regression":
                    POS = np.where(_label == 1)
                    zmax,zmin,xmax,xmin = np.max(POS[0]), np.min(POS[0]), np.max(POS[1]), np.min(POS[1])
                    bbox.append([zmin, zmax,xmin, xmax])
                    labels.append([batches[self.args.prediction_classes].iloc[ind].values])
                else:
                    labels.append(_label)
                if self.args.num_output_objects > 1 :
                    if self.args.training_type == "Regression":
                        MASK_PATH2 = os.path.join(self.args.dataset_path_mask, str(batches[self.args.second_image_col].iloc[ind]) + self.args.ext_type_mask)
                        _label2 = self.__get_input_mask(MASK_PATH2, mask_label)
                        POS = np.where(_label == 1)
                        zmax,zmin,xmax,xmin = np.max(POS[0]), np.min(POS[0]), np.max(POS[1]), np.min(POS[1])
                        bbox2.append([zmin, zmax,xmin, xmax])
                        labels2.append([batches[self.args.prediction_classes].iloc[ind].values])
                    else:
                    
                        MASK_PATH2 = os.path.join(self.args.dataset_path_mask, str(batches[self.args.second_image_col].iloc[ind]) + self.args.ext_type_mask)
                        _label2 = self.__get_input_mask(MASK_PATH2, mask_label)
                        _label2 = np.squeeze(_label2)
                        labels2.append(_label2)
                    

        X = np.array(images, dtype=np.float32)
        

        if self.args.model_type == 'Classification':
            y = np.array(labels, dtype=np.int64)
            y = y.squeeze(axis=1)
            
            if self.args.num_input_objects == 2 and self.args.num_output_objects > 1:
                Z = np.array(images2, dtype=np.float32)
                w = np.array(labels2, dtype=np.int64)
                w = w.squeeze(axis=1)
                if random.uniform(0, 1) < (0.5) and self.args.model_type == 'Classification' and self.augmentation:
                    random_number = random.randint(1, 5)
                    X, Z = self.batch_augmentation(X, random_number), self.batch_augmentation(Z, random_number)
                
                return X, Z, y, w
            elif self.args.num_input_objects == 1 and self.args.num_output_objects == 1:
                if random.uniform(0, 1) < (0.5) and self.args.model_type == 'Classification' and self.augmentation:
                    random_number = random.randint(1, 5)
                    X = self.batch_augmentation(X, random_number)
                return X, y
            elif self.args.num_input_objects == 2 and self.args.num_output_objects == 1:
                Z = np.array(images2, dtype=np.float32)
                if random.uniform(0, 1) < (0.5) and self.args.model_type == 'Classification' and self.augmentation:
                    random_number = random.randint(1, 5)
                    X, Z = self.batch_augmentation(X, random_number), self.batch_augmentation(Z, random_number)
                if random.uniform(0, 1) > (0.5) and self.args.two_way_shuffle:
                    return Z, y, X
                else:
                    return X, y, Z
            elif self.args.num_input_objects == 1 and self.args.num_output_objects > 1:
                w = np.array(labels2, dtype=np.int64)
                w = w.squeeze(axis=1)
                if random.uniform(0, 1) < (0.5) and self.args.model_type == 'Classification' and self.augmentation:
                    random_number = random.randint(1, 5)
                    X = self.batch_augmentation(X, random_number)
                return X, y, w
            elif self.args.num_input_objects == 4 and self.args.num_output_objects == 1:
                Z = np.array(images2, dtype=np.float32)
                W = np.array(images3, dtype=np.float32)
                U = np.array(images4, dtype=np.float32)
                if random.uniform(0, 1) < (0.5) and self.args.model_type == 'Classification' and self.augmentation:
                    random_number = random.randint(1, 5)
                    X, Z, W, U = self.batch_augmentation(X, random_number), self.batch_augmentation(Z, random_number), self.batch_augmentation(W, random_number), self.batch_augmentation(U, random_number)
                return X, y, Z, W, U
            elif self.args.num_input_objects == 5 and self.args.num_output_objects == 1:
                Z = np.array(images2, dtype=np.float32)
                W = np.array(images3, dtype=np.float32)
                U = np.array(images4, dtype=np.float32)
                N = np.array(norm, dtype=np.float32)
                if random.uniform(0, 1) < (0.5) and self.args.model_type == 'Classification' and self.augmentation:
                    random_number = random.randint(1, 5)
                    X, Z, W, U = self.batch_augmentation(X, random_number), self.batch_augmentation(Z, random_number), self.batch_augmentation(W, random_number), self.batch_augmentation(U, random_number)
                return X, y, Z, W, U, N
            elif self.args.num_input_objects == 6 and self.args.num_output_objects == 1:
                Z = np.array(images2, dtype=np.float32)
                return X, y, Z, X, Z, X, Z
            
        elif self.args.model_type == 'Segmentation':
            y = np.array(labels)
            if self.args.training_type == "Regression":
                w = np.array(bbox)
                Z = np.array(images2, dtype=np.float32)
                u = np.array(bbox2)
                
                return X, Z, w, u
            else:
                if self.args.num_input_objects > 1 and self.args.num_output_objects > 1:
                    w = np.array(labels2)
                    Z = np.array(images2, dtype=np.float32)
                    return X, y, Z, w
                elif self.args.num_input_objects == 1 and self.args.num_output_objects == 1:
                    return X, y
                elif self.args.num_input_objects > 1 and self.args.num_output_objects == 1:
                    Z = np.array(images2, dtype=np.float32)
                    
                    return X, Z, y
                elif self.args.num_input_objects == 1 and self.args.num_output_objects > 1:
                    w = np.array(labels2)
                    return X, w
        
    def get_dataframe(self):
        return self.df
    def get_name_dataframe(self, obj, row):
        return self.df[obj].iloc[row]
    def num_steps_per_epoch(self):
        return int(np.ceil(np.array(self.n // self.args.batch_size)))

    def __len__(self):
        return self.n // self.args.batch_size
    def __next__(self):
        if self.i < self.num_steps_per_epoch:
            materials, labels = self.__getitem__(self.i)
            self.i += 1
        else:
            raise StopIteration()
        return materials, labels
  
    def __call__(self):
        self.i = 0
        return self

