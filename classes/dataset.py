import cv2
import pandas as pd
import numpy as np
# from tools.merge_classes import *

class Dataset:
    def __init__(
        self,
        images_csv,
        n_classes,
        imgsz,
        n_channels=1
    ):
        self.images_csv = images_csv
        self.n_classes = n_classes
        self.imgsz = imgsz
        self.n_channels = n_channels

    def __len__(self):
        return self.count_csv_lines(self.images_csv)

    def __getimage__(self, index, filter=None):
        df = pd.read_csv(self.images_csv, skiprows=index+1, nrows=1)
        df = pd.DataFrame(df)

        image_path, mask_path = df.iloc[0]

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)

        padding = int(abs(self.imgsz - image.shape[0])/2)
        
        if self.n_channels == 3:
            image = add_padding_with_zeros(image, padding, 3)
        else:    
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype('uint8')
            image = add_padding_with_zeros(image, padding, 1)
            
        
        
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY).astype('uint8')
        
        # # Condition to handling with non-square images
        # if image.shape[0] != image.shape[1]: 
        #   width = int(image.shape[1] * (self.imgsz / image.shape[0]))
        # else:
        #   width = self.imgsz

        
        # image = cv2.resize(image, (self.imgsz, self.imgsz)).astype('uint8')
        # mask = cv2.resize(mask, (self.imgsz, self.imgsz)).astype('uint8')
        mask = add_padding_with_zeros(mask, padding, 1)

        # mask = merge_classes(mask, self.n_classes)
        
        if filter:
            transformed = filter(image=image)
            image = transformed['image'].astype('uint8')
        
        del padding, df

        return image, mask
    
    def count_csv_lines(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            number_lines = len(lines)
            return number_lines-1

    
def add_padding_with_zeros(image, padding, ch):
    if ch == 3:
        # Imagem com 3 canais
        padded_shape = (image.shape[0] + 2 * padding, image.shape[1] + 2 * padding, image.shape[2])
        padded_image = np.zeros(padded_shape, dtype=image.dtype)
        padded_image[padding:-padding, padding:-padding, :] = image
    elif ch == 1:
        # Imagem com 1 canal
        padded_shape = (image.shape[0] + 2 * padding, image.shape[1] + 2 * padding)
        padded_image = np.zeros(padded_shape, dtype=image.dtype)
        padded_image[padding:-padding, padding:-padding] = image
    
    return padded_image
