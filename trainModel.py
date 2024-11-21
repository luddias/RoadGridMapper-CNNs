from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import tensorflow as tf
from tools.utils import get_class_weights, batch_generator, generate_batchs, get_callbacks
import os
from classes.Dataset import Dataset

LR = 5e-3
BATCH_SIZE = 4
EPOCHS = 10
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ["SM_FRAMEWORK"] = "tf.keras"


def trainModel(model, train_csv, val_csv, nc, img_shape, cwp, model_name = ""):   
    # print("LETS START OUT TRAINNING! First lets check the data... it can take a while")
    train_dataset = Dataset(train_csv, nc, img_shape[0], img_shape[2])
    val_dataset = Dataset(val_csv, nc, img_shape[0], img_shape[2])
    print('[INFO]Datasets Sucessfully defined!')
    
    # Get the class weights
    print('[INFO]Starting to define the class weights...')
    class_weights = get_class_weights(train_dataset, nc, cwp=cwp)
    print('[INFO]Fetched all class weights successfully!')
    
    opt = tf.keras.optimizers.Adam(learning_rate=LR)
    
    keras.losses.CategoricalFocalCrossentropy(
        alpha=0.25,
        gamma=2.0,
        from_logits=False,
        label_smoothing=0.0,
        axis=-1,
        reduction= keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        name='categorical_focal_crossentropy'
    )
    print('[INFO]Defined the loss function and the optimizer')

    model.compile(optimizer=opt, loss='categorical_focal_crossentropy', metrics=['categorical_accuracy'])
    print('[INFO]Model compilated with success!')
    
    sr_train = generate_batchs(len(train_dataset), BATCH_SIZE)
    sr_eval = generate_batchs(len(val_dataset), BATCH_SIZE)
    
    bc_train = len(sr_train)
    bc_eval = len(sr_eval)

    pipe = batch_generator(train_dataset,steps = bc_train, batch_size = BATCH_SIZE, skiprows=sr_train ,n_classes=nc )
    eval_pipe = batch_generator(val_dataset,steps = bc_eval, batch_size = BATCH_SIZE, skiprows=sr_eval,n_classes=nc )
    
    os.makedirs('checkpoints', exist_ok=True)
    callbacks_list = get_callbacks('checkpoints/'+model_name+'_checkpoint.keras')
    
    
    # Training Loop starts
    print('[INFO]Starting Training...')
    print()
    
    history= model.fit(pipe,
                    epochs=EPOCHS,
                    steps_per_epoch=bc_train,
                    class_weight=class_weights,
                    verbose=1,
                    validation_data=eval_pipe,
                    validation_steps=bc_eval,
                    callbacks=[callbacks_list])
    
    print('[INFO]Training completed with Success!')
     
    return history
