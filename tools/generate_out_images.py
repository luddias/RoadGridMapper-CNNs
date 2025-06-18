import random
import matplotlib.pyplot as plt
import numpy as np
# Importar imagem aleatória
import random
from tensorflow.keras.utils import normalize
num_imgs = 3555
from classes.dataset import Dataset as SegmentationDataset
from tensorflow.keras.models import load_model
from tools.utils import *

def load_image(test_dataset, num):
    x,y = test_dataset.__getimage__(num)

    x = np.asarray([np.expand_dims(x, axis = -1)])
    y = np.asarray([np.expand_dims(y, axis = -1)])
    
    
    return x, y
def generate_predict(out_path, test_img_number, model, test_dataset):
    #test_img_number = random.randint(0, num_imgs-1)
    
    x,y =load_image(test_dataset, test_img_number)
    print(x.shape, y.shape)
    
    data = normalize(x, axis = 1)
    prediction = (model.predict(data))
    predicted_img=np.argmax(prediction, axis=-1)

    plt.axis('off')
    plt.imshow(predicted_img[0], cmap='viridis')
    plt.colorbar()  # Adicionar uma barra de cores para referência
    plt.savefig(out_path+'p'+str(test_img_number)+'.png', format='png')
    
    plt.clf()
    
    plt.axis('off')
    plt.imshow(x[0], cmap='gray')
    plt.savefig(out_path+'i'+str(test_img_number)+'.png', format='png')
    
    plt.clf()
    
    plt.axis('off')
    plt.imshow(y[0], cmap='viridis')
    plt.colorbar()  # Adicionar uma barra de cores para referência
    plt.savefig(out_path+'r'+str(test_img_number)+'.png', format='png')
    
    plt.clf()
    



if __name__ == '__main__':
    # Ajuste as configurações manualmente aqui
    test_csv = 'DATASET_RGM/test.csv'
    test_dataset = SegmentationDataset(test_csv, 17, 128, n_channels=1)
    out_path = 'outputs/imgs_preds/unet/'
    model = load_model('checkpoints/treino5_pad/Unet_checkpoint.keras')
    imgs_idx = [1762, 1692, 1502, 1484, 1547, 827]
    for i in imgs_idx:
        generate_predict(out_path, i, model, test_dataset)
