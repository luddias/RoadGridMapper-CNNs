
from tqdm import tqdm
from prettytable import PrettyTable
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns



import tensorflow as tf
from keras.models import load_model
from tools.utils import get_class_weights, batch_generator, generate_batchs, get_callbacks
import os
from classes.dataset import Dataset
from classes.metrics import Metrics
from tensorflow.keras.utils import normalize

BATCH_SIZE = 8

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ["SM_FRAMEWORK"] = "tf.keras"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Configura o TensorFlow para alocar memória na GPU conforme necessário
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def evaluateModel(model, test_csv, nc = 17, img_shape = (128,128,1), model_name = 'None'):
    
    test_dataset =  __initialize(test_csv, nc, img_shape)
    
    out_dir = os.path.join('results', model_name)
    os.makedirs(out_dir, exist_ok=True)
    # model = load_model('checkpoints/PSPNet_checkpoint.keras') 
    # print(model.summary())
    
    __evaluate(model, test_dataset, out_dir, num_classes = nc, only_generate_images=False)
    
    print('[INFO]Test completed with Success!')
        

    
def __initialize(test_csv, nc, img_shape):
    
    test_dataset = Dataset(test_csv, nc, img_shape[0], n_channels=img_shape[2])
    print('[INFO]Datasets Sucessfully defined!')

    
    return test_dataset

def __evaluate(
    model,
    test_dataset,
    out_dir,
    num_classes,
    only_generate_images=False
):
    print('Evaluating')
    
    valid_running_loss = 0.0

    conf_mat = np.zeros((num_classes, num_classes)).astype(np.int64)

    metrics = Metrics()
    
    for i in range(len(test_dataset)-1):
        
        images, masks = test_dataset.__getimage__(i)
        images = normalize(images, axis=1)
        
        images = np.asarray([np.expand_dims(images, axis = -1)])
        masks = np.asarray([np.expand_dims(masks, axis = -1)])
        # print(images.shape, masks.shape)

        #forward
        outputs = model.predict(images)

        preds = np.argmax(outputs, axis=3) 
        
        preds, masks = reduzir_tamanho(preds, masks)
        
        # if i>100 and i<110:
        #     save_images(images, masks, preds, out_dir, counter)

        # elif i>=120 and only_generate_images:
        #     return
        
        preds = preds.squeeze().astype(np.uint8)

        masks_cpu = masks.squeeze().astype(np.uint8)

        metrics.calculate_metrics(preds, masks_cpu)
        

        
        del images, masks, outputs, preds, masks_cpu

        ##########################


    df, bp_values = metrics.get_results()
    print(df)
    print(bp_values)
    df.to_csv(os.path.join(out_dir, 'results.csv'), index=False)
    
    
def reduzir_tamanho(preds, masks):
    tam_img = 120
    dif = abs(masks.shape[1] - tam_img)
    if dif%2 !=0:
        raise ValueError("A diferença entre o tamanho atual e o tamanho desejado deve ser par.")
    if dif > 0:
        half_dif = dif // 2
        preds = preds[:, half_dif:-half_dif, half_dif:-half_dif]
        masks = masks[:, half_dif:-half_dif, half_dif:-half_dif]
        
    return preds, masks

def save_images(images, gts, preds, out_dir, counter):
    
    color_map = {
    0: [42, 0, 128],   # Azul Escuro
    1: [255, 84, 0],   # Laranja
    2: [142, 0, 0], # Vermelho
    3: [81, 0, 255],   # Azul Claro
    4: [170, 255, 40], # Verde Claro
    5: [0, 255, 211]   # Turquesa
}
    
    path = os.path.join(out_dir,'pred_imgs')
    os.makedirs(path, exist_ok=True)
    
    images = images.cpu()
    gts = gts.cpu()

    for i, img in enumerate(preds):
        # Plot e salva a imagem
        
        gt = gts[i]
        
        color_array = np.zeros((img.size(0), img.size(1), 3), dtype=nnp.uint8)
        gt_array = np.zeros((gt.size(0), gt.size(1), 3), dtype=np.uint8)
        
        for val in range(6):
            color_array[img == val] = np.array(color_map[val], dtype=np.uint8)
            gt_array[gt == val] =  np.array(color_map[val], dtype=np.uint8)
            

        # Converter o tensor de cores para uma imagem PIL
        pred_img = Image.fromarray(color_array)
        # Salvar a imagem
        pred_img.save(os.path.join(path,f'{counter*i}.png'))
        
        image = images[i]
        image = (image * 255).byte() 
        image = image.repeat(3, 1, 1).permute(1, 2, 0) 
        # Converter o tensor de cores para uma imagem PIL
        image = Image.fromarray(image.numpy())
        # Salvar a imagem
        image.save(os.path.join(path,f'{counter*i}_remission.png'))

        # Converter o tensor de cores para uma imagem PIL
        gt_img = Image.fromarray(gt_tensor.numpy())
        # Salvar a imagem
        gt_img.save(os.path.join(path,f'{counter*i}_gt.png'))

        del gt_img, image, pred_img, gt_tensor, color_tensor

if __name__ == '__main__':

    os.environ["SM_FRAMEWORK"] = "tf.keras"
    
    architectures = {
        'PSPNet_novembro': ('checkpoints/PSPNet_checkpoint.keras', (144,144,3)),
        'Linknet_novembro': ('checkpoints/Linknet_checkpoint.keras', (128,128,3)),
        'Unet_novembro': ('checkpoints/Unet_checkpoint.keras', (128,128,1)),
        
        'FPN_novembro': ('checkpoints/FPN_checkpoint.keras', (128,128,3))
    }
    for key, config in architectures.items():
        print('First Model to be trained: {}'.format(key))
        input_shape = config[1]
        model_path = config[0]
        
        evaluateModel(model_path, test_csv = 'DATASET_RGM/test.csv', nc = 17, img_shape=input_shape, model_name = key)
