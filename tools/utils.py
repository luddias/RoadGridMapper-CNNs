import numpy as np
import classes.Dataset as Dataset
import pandas as pd
import cv2
import albumentations as A
from tensorflow.keras.utils import to_categorical
import json
from tensorflow.keras.utils import normalize
# from typing import List
from tqdm import tqdm
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
# from tools.merge_classes import *
import gc
import tensorflow as tf
from tensorflow.keras import backend as K

def get_class_weights(dataset, img_size, cwp = None):
    '''This function calculates the class weights based on pixel frequency in images'''
    if cwp:
        with open(cwp, 'r') as f:
            l =json.load(f)
            dic = {}
            for i, weight in enumerate(l):
                dic[i] = weight
            return 
    
    n_classes = dataset.n_classes
    
    # Initialize an array to store class frequencies
    classes = np.zeros(n_classes, dtype=np.uint64)
    
    # Iterate over the dataset to calculate class frequencies
    for i in tqdm(range(len(dataset)), desc='Calculating class frequencies'):
        _, mask = dataset.__getimage__(i)
        
        # Count the frequency of each class in the mask
        unique, counts = np.unique(mask, return_counts=True)
        classes[unique] += counts.astype(np.uint64)
        
    # Calculate class weights
    class_weights = class_weights = [(dataset.__len__() * img_size * img_size) / (n_classes * freq) for freq in classes]
    
    # Save class weights to a JSON file
    with open('class_weights.json', 'w') as f:
        json.dump(class_weights.tolist(), f)
        print('Class weights saved in class_weights.json file')
    
    dict_classes = {}
    for idx, c in enumerate(class_weights):
        dict_classes[idx] = c

    print('The class weights are: '+ dict_classes)
    return dict_classes


def train_transforms():
    """
    Transforms/augmentations for training images and masks.

    :param img_size: Integer, for image resize.
    """
    
    train_image_transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
    ], is_check_shapes=False)
    return train_image_transform

def valid_transforms(img_size):
    """
    Transforms/augmentations for validation images and masks.

    :param img_size: Integer, for image resize.
    """
    
    valid_image_transform = A.Compose([
        A.Resize(img_size[1], img_size[0], always_apply=True),
    ], is_check_shapes=False)
    return valid_image_transform

def batch_generator(dataset:Dataset,batch_size = 8,
                    steps =1 , skiprows= 0, n_classes=2, mode='keras'):
    idx=1
    while True:
        # print('Importando dados do batch')
        yield load_data(dataset,batch_size, skiprows[idx-1], n_classes)## Yields data
        if idx<steps:
            idx+=1
        else:
            idx=1

def load_data(dataset, batch_size, sr, n_classes):

    x = []
    y = []

    for i in range(sr, sr+batch_size):
        filter = None
        if i<=(sr+batch_size)*0.25:
            filter = train_transforms()
        img, mask = dataset.__getimage__(i, filter)
        x.append(img)
        y.append(mask)


    y = np.asarray(y)
    x = np.asarray(x)

    train_masks_cat = to_categorical(y, num_classes=n_classes)
    y = train_masks_cat.reshape((y.shape[0], y.shape[1], y.shape[2], n_classes))
    x = normalize(x, axis=1)
    
    # print('dados prontos')
    del img, mask, train_masks_cat
    return (x, y)

def generate_batchs(n_data, bs):
    skiprows = []
    
    for i in range(0, n_data-1, bs):
        skiprows.append(i)
    
    if skiprows[-1]+bs>= n_data:
        skiprows.pop()

            
    return skiprows
  
def strlist_to_intlist(values: str):
    if values == None:
      return []
    
    substrings = values.split(';')

    int_list=[[int(num) for num in substr.split(',')] for substr in substrings]
    
    return int_list



# Libere a memória após cada epoch
class MemoryCleaner(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        # K.clear_session()


def get_callbacks(filepath):
  checkpoint = ModelCheckpoint(filepath, monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max')
  es = EarlyStopping(monitor='val_categorical_accuracy', patience=8, mode='max', verbose=1)
  reduce_lr = ReduceLROnPlateau(monitor='val_categorical_accuracy', threshold=0.01, threshold_mode='abs', factor=0.001, patience=5, mode='max')
  return [checkpoint, es, reduce_lr, MemoryCleaner()]


# def merge_classes(mask, class_values):
#     # Inicializar nova máscara com zeros
#     new_mask = np.zeros(mask.shape, dtype=np.uint8)

#     # Converter class_values para numpy array para vetorização
#     class_values = np.array(class_values, dtype=object)

#     # Criar uma máscara booleana para cada classe de valores
#     for idx, value in enumerate(class_values):
#         bool_mask = np.isin(mask, value)
#         new_mask[bool_mask] = idx

#         # Adicionalmente, tratar o último caso se necessário
#         if idx == len(class_values) - 1:
#             bool_mask = mask > value[-1]
#             new_mask[bool_mask] = idx

#     return new_mask
