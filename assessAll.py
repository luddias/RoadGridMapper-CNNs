import os
from trainModel import trainModel
import csv
import segmentation_models as sm
import tensorflow as tf
from keras.models import load_model
from evaluateModel import evaluateModel
import gc

BACKBONE = 'efficientnetb6'
CLASSES = 17
TRAIN_CSV = "DATASET_RGM/train.csv"
VAL_CSV ="DATASET_RGM/validation.csv"
class_weights = 'configs/class_weights.json'

# TREINAR TODOS COM PESOS :) depois treinar com borda de padding 8px ao inves de mudar o tamanho
print('Tudo pronto pra iniciar')

os.environ["SM_FRAMEWORK"] = "tf.keras"

architectures = {
    # 'Linknet': (sm.Linknet, (128,128,3)),
    # 'Unet': (sm.Unet, (128,128,1)),
    'FPN': (sm.FPN, (128,128,3)),
    # 'PSPNet': (None, (144,144,3)),
    # 'PSPNet': (sm.PSPNet, (144,144,3))
}

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Configura o TensorFlow para alocar memória na GPU conforme necessário
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        
for key, config in architectures.items():
    print('First Model to be trained: {}'.format(key))
    input_shape = config[1]
    
    print(input_shape)
    # arch = config[0](BACKBONE, classes=CLASSES, input_shape=input_shape, activation='softmax', encoder_weights=None)
    arch = load_model('checkpoints/FPN_checkpoint.keras')
    print(arch.summary())
    
    with tf.device('/GPU:0'): 
        # history = trainModel(arch, TRAIN_CSV, VAL_CSV, CLASSES, input_shape, class_weights, model_name = key)
        evaluateModel(arch, test_csv = 'DATASET_RGM/test.csv', nc = 17, img_shape=input_shape, model_name = key)
    
    # with open(key+'_history.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['epoch', 'loss', 'categorical_accuracy', 'val_loss', 'val_categorical_accuracy'])
    #     for i in range(len(history.history['loss'])):
    #         writer.writerow([i+1, history.history['loss'][i], history.history['categorical_accuracy'][i],
    #                      history.history['val_loss'][i], history.history['val_categorical_accuracy'][i]])
    del arch
    gc.collect()
