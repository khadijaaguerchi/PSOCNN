"""PSOCNN Algorithm
Copyright (c) 2023 Future Processing

@author:
@email: 
@date: 
"""
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from datetime import datetime
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Activation,Dense, Dropout,Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
import os
import pathlib 
import random
import random
import numpy as np

from tensorflow.keras import layers 


imgd=ImageDataGenerator(rescale=1/255)

tumor_dataset=imgd.flow_from_directory('../input/breastcancermasses/Dataset of Mammography with Benign Malignant Breast Masses/Dataset of Mammography with Benign Malignant Breast Masses/MIAS Dataset')

tumor_dataset.class_indices

classes=pd.DataFrame(tumor_dataset.classes)
classes.value_counts()

path_train = '../input/breastcancermasses/Dataset of Mammography with Benign Malignant Breast Masses/Dataset of Mammography with Benign Malignant Breast Masses/DDSM Dataset'
path_test = '../input/breastcancermasses/Dataset of Mammography with Benign Malignant Breast Masses/Dataset of Mammography with Benign Malignant Breast Masses/DDSM Dataset'

# **other version of code**

data_dir = pathlib.Path(path_test)

class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))
class_names

benignPath = os.path.join(data_dir,'Benign Masses')
malignantPath = os.path.join(data_dir,'Malignant Masses')

imageCount = len(list(data_dir.glob('*/*.png')))
imageCount

#path_train = '../input/breastcancermasses/Dataset of Mammography with Benign Malignant Breast Masses/Dataset of Mammography with Benign Malignant Breast Masses/MIAS Dataset'
#path_test = '../input/breastcancermasses/Dataset of Mammography with Benign Malignant Breast Masses/Dataset of Mammography with Benign Malignant Breast Masses/MIAS Dataset'
from tensorflow.keras.utils import image_dataset_from_directory

import tensorflow as tf
import tensorflow.keras.layers as tfl

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,validation_split=0.2)

train_data = train_datagen.flow_from_directory(
        data_dir,
        subset='training',
        class_mode = 'categorical',
        #seed=123,
        target_size=(200 , 200),
        batch_size=32
)

val_data = train_datagen.flow_from_directory(
        data_dir,
        subset='validation',
        class_mode = 'categorical',
       # seed=123,
        target_size=(200 , 200),
        batch_size=32 )
test_data = train_datagen.flow_from_directory(
        data_dir,
        target_size=(200 , 200),
        batch_size=32
        )

import numpy as np
import random
# Define the hyperparameter dictionary
hyperparameters = {
                   #'batch_size': [32, 64,2],
                   #'num_epochs': [10, 20,5],**/
                   'kernel size': [4, 8,1],
                   'stride': [2, 4,1],
                   'numberfilter': [16,64,16]}

# Define the number of particles for PSO
n_particles = 4
n_dims=len(hyperparameters)
c1 = 2 # cognitive weight
c2 = 2 # social weight
w = 0.7 # inertia weight
max_iter = 3 # maximum number of iterations
#SCSS
optimal_hyperparams, optimal_accuracy = run_pso(n_particles, n_dims,hyperparameters, c1, c2, w, max_iter)

print("Optimal Hyperparameters:", optimal_hyperparams)
print("Optimal Accuracy:", optimal_accuracy)

# %% [markdown]
# **other version******

# %% [code] {"execution":{"iopub.status.busy":"2023-09-01T21:11:31.967065Z","iopub.execute_input":"2023-09-01T21:11:31.967532Z","iopub.status.idle":"2023-09-01T21:11:33.660468Z","shell.execute_reply.started":"2023-09-01T21:11:31.967500Z","shell.execute_reply":"2023-09-01T21:11:33.659086Z"},"jupyter":{"outputs_hidden":false}}
from tensorflow.keras.utils import image_dataset_from_directory

train_data = image_dataset_from_directory(
                  data_dir,
                  validation_split=0.2,
                  subset="training",
                  seed=123,
                  image_size=(200, 200),
                  batch_size=32)


val_data = image_dataset_from_directory(data_dir,
                                        validation_split=0.2,
                                        subset="validation",
                                        seed=123,
                                        image_size=(200,200),
                                        batch_size=32)

model = tf.keras.Sequential([
  layers.Rescaling(1./255, input_shape=(200, 200, 3)),
    
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
    
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
    
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
    
  layers.Dropout(0.5),
  layers.Flatten(),
  layers.Dense(16,activation="relu"),
  layers.Dense(2,activation="sigmoid")
])

model.compile(optimizer="Adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"])
history = model.fit(train_data,
                    epochs=2,
                    validation_data=val_data, 
                    batch_size=32)


    # Implement the code to train and evaluate the model with optimal hyperparametrs
    #ks,s,lr =x[0],x[1],x[2]
    model =Sequential()
#convolution and maxpoollayer
    model.add(Conv2D(filters=48,kernel_size=(6,6),
                 strides=(4,4),padding='valid',input_shape=(200,200,3)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=2))
#flatten layer
    model.add(Flatten())
#hidden layer
    model.add(Dense(16))
    model.add(Activation('relu'))
#output layer
    model.add(Dense(2))
    model.add(Activation('sigmoid'))
    #opt = Adam(learning_rate=lr)

 model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])

history=model.fit (train_data,epochs=3, validation_data=val_data)

model.evaluate(val_data)

model.summary()

plt.figure(figsize=(15, 15))
class_names = val_data.class_indices
result = ' | False'
for images, labels in val_data.take(1):
    for i in range(25):
        ax = plt.subplot(5, 5, i + 1)
        img = images[i].numpy().astype("uint8")
        img = tf.expand_dims(img, axis=0)
        
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions)
        if class_names[predicted_class] == class_names[labels[i]]:
            result = ' | TRUE'
            
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[predicted_class]+result  )
        plt.axis("off")

# %% [markdown]
# **w3schools**

import matplotlib.pyplot as plt
import numpy
from sklearn import metrics

test_imgs, test_labels = next(val_data)
results = np.zeros( (624,2) ) 
for j in range(5):
    results = model.predict(test_imgs)
 #   results = results + model[j].predict_generator(test_imgs, steps=1, verbose=0)
results = np.argmax(results,axis = -1)

test_labels = np.argmax(test_labels,axis = -1)
#################
confusion_matrix = metrics.confusion_matrix(test_labels, results)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()

test_imgs, test_labels = next(val_data)

# ENSEMBLE PREDICTIONS AND SUBMIT
results = np.zeros( (624,2) ) 
for j in range(5):
    results = model.predict(test_imgs)
 #   results = results + model[j].predict_generator(test_imgs, steps=1, verbose=0)
results = np.argmax(results,axis = -1)

results

test_labels = np.argmax(test_labels,axis = -1)
test_labels

from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score

cm = confusion_matrix(y_true=test_labels, y_pred=results)

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# %% [code] {"jupyter":{"outputs_hidden":false}}
 from sklearn.metrics import confusion_matrix

        # compute the confusion matrix
cm = confusion_matrix(class_names,predicted_class)
 
#Plot the confusion matrix.
sns.heatmap(cm,
            annot=True,
            fmt='g',
            xticklabels=['malignant', 'benign'],
            yticklabels=['malignant', 'benign'])
plt.ylabel('Prediction',fontsize=13)
plt.xlabel('Actual',fontsize=13)
plt.title('Confusion Matrix',fontsize=17)
plt.show()

import itertools
cm_plot_labels = ['benign', 'malignat']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='')
