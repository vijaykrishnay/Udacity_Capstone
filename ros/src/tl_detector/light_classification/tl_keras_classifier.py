#pip install --upgrade wandb pandas sklearn
#wandb login 37f757946de773540d1bdc5db25e71575eb94145
import pandas as pd
import wandb
import keras
from wandb.keras import WandbCallback
import shutil
import tensorflow as tf
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# lr: float >= 0. Learning rate.
# beta_1: float, 0 < beta < 1. Generally close to 1.
# beta_2: float, 0 < beta < 1. Generally close to 1.
# epsilon: float >= 0. Fuzz factor.
# decay: float >= 0. Learning rate decay over each update.
##Default params
# lr=0.001,
# beta_1=0.9,
# beta_2=0.999,
# epsilon=1e-08,
# decay=0.0,

hyper_params = dict(
    image_size = 224,
    num_layers_frozen = 60,
    dense_1_size = 1024,
    dense_2_size = 1024,
    dense_3_size = 512,
    batch_size = 32,
#     steps_per_epoch = 200,
    #Adam Optimizer
    lr=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-08,
    decay=0.000,
    epochs = 5
)

wandb.init(project="car_nanodegree_capstone", config=hyper_params)

config = wandb.config

dataBaseFolder = '/home/workspace/Udacity_Capstone/data/'
# datasdcnd = dataBaseFolder + 'dataset-sdcnd-capstone/data/real_training_data/'
dataTL = dataBaseFolder + 'tl_engineer5/'

base_model = keras.applications.mobilenet.MobileNet(input_shape=(config.image_size,config.image_size,3), include_top=False, weights='imagenet')

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(config.dense_1_size,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(config.dense_2_size,activation='relu')(x) #dense layer 2
x=Dense(config.dense_3_size,activation='relu')(x) #dense layer 3
preds=Dense(4,activation='softmax')(x) #final layer with softmax activation

model=Model(inputs=base_model.input,outputs=preds)

for layer in model.layers[:config.num_layers_frozen]:
    layer.trainable=False

for layer in model.layers[config.num_layers_frozen:]:
    layer.trainable=True

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    dataTL+'train',
    target_size=(config.image_size, config.image_size),
    batch_size=1,
)

train_generator = train_datagen.flow_from_directory(
        dataTL+'test',
        target_size=(config.image_size, config.image_size),
        batch_size=config.batch_size,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        dataTL+'val',
        target_size=(config.image_size, config.image_size),
        batch_size=config.batch_size,
        class_mode='categorical')

adam_opt = Adam(lr=config.lr, beta_1=config.beta_1, beta_2=config.beta_2, epsilon=config.epsilon, decay=config.decay)

model.compile(optimizer=adam_opt,loss='categorical_crossentropy',metrics=['accuracy'])

model.fit_generator(
        train_generator,
        epochs=config.epochs,
        steps_per_epoch=3521 // config.batch_size,
        validation_data=validation_generator,
        validation_steps = 755//config.batch_size,
        callbacks=[WandbCallback()])


model.save(os.path.join(wandb.run.dir, "model_70_frozen.h5"))

saver = tf.train.Saver()
saver.save(K.get_session(), '/tmp/keras_model.ckpt')

scores = model.evaluate_generator(generator=test_generator, steps=755//config.batch_size)
print('Accuracy: ', scores[1])

filenames = test_generator.filenames
nb_samples = len(filenames)
pred=model.predict_generator(test_generator, steps=nb_samples, verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)

#Confution Matrix and Classification Report
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, predicted_class_indices))

# labels = (train_generator.class_indices)
# labels = dict((v,k) for k,v in labels.items())
# predictions = [labels[k] for k in predicted_class_indices]

print('Classification Report')
print(classification_report(test_generator.classes, predicted_class_indices))

# filenames=test_generator.filenames
# results=pd.DataFrame({"Filename":filenames,
#                       "Predictions":predictions})

# results.to_csv(dataBaseFolder + 'results_baseline.csv',index=False)

