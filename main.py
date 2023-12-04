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
def evaluate_model2(x):
    """Evaluate the performance of a CNN model with the given hyperparameters.
    Returns the accuracy of the model."""
    # Implement the code to train and evaluate the model
    ks,s,lr =x[0],x[1],x[2]
    
    model = tf.keras.Sequential([
    layers.Rescaling(1./255, input_shape=(200, 200, 3)),
    layers.Conv2D(lr, ks,strides=(s,s), padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, ks, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.5),
    layers.Flatten(),
    layers.Dense(16,activation="relu"),
    layers.Dense(2,activation="sigmoid")
    ])
    
    #early stop  to avoid overfitting
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5)
    
    #opt = Adam(learning_rate=lr)
    model.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])
    history=model.fit (train_data,epochs=3, validation_data=val_datan,callbacks=[early_stop])
    
    ############################## Plot Accuracy and Loss ############################
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
# summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    score=model.evaluate(test_data)
 # loss, val
    print('current config:',x,'val test accuracy:',score[1])
    return score[1]

# %% [code] {"execution":{"iopub.status.busy":"2023-09-25T01:20:48.173859Z","iopub.execute_input":"2023-09-25T01:20:48.174260Z","iopub.status.idle":"2023-09-25T01:20:48.180451Z","shell.execute_reply.started":"2023-09-25T01:20:48.174228Z","shell.execute_reply":"2023-09-25T01:20:48.179174Z"},"jupyter":{"outputs_hidden":false}}
#early stop  to avoid overfitting
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-09-25T01:20:51.249519Z","iopub.execute_input":"2023-09-25T01:20:51.249965Z","iopub.status.idle":"2023-09-25T01:20:51.271473Z","shell.execute_reply.started":"2023-09-25T01:20:51.249931Z","shell.execute_reply":"2023-09-25T01:20:51.269993Z"}}
def initialize_particles(n_particles,n_dims, hyperparameters):
    result = hyperparameters.values()
# Convert object to a list
    data = list(result)
# Convert list to an array
    numpyArray = np.array(data)
# Generate initial particles randomly
    particles = []
    for i in range(n_particles):
        particle = []
        for j in range(n_dims):
            particle.append(random.randrange(numpyArray[j][0], numpyArray[j][1]+1,numpyArray[j][2]))
        #particle.append(random.uniform(numpyArray[n_dims-1][0], numpyArray[n_dims-1][1]))

        particles.append(particle)
    return particles

def update_velocity(particle, velocity, best_particle, global_best, c1, c2, w):
    """Update the velocity of a particle based on its previous velocity, its personal best,
    and the global best."""
    for i in range(len(particle)):
        r1 = random.randrange(0,2)
        r2 = random.randrange(0,2)
    # if type(particle[i])==int:
        vel_cognitive = c1 * r1 * (best_particle[i] - particle[i])
        vel_social = c2 * r2 * (global_best[i] - particle[i])
        velocity[i] = velocity[i] + vel_cognitive + vel_social
          
        """vel_cognitive = c1 * r1 * (best_particle[i] - particle[i])
            vel_social = c2 * r2 * (global_best[i] - particle[i])
            velocity[i] = w*velocity[i] + vel_cognitive + vel_social"""
    return velocity

def update_particle(particle, velocity):
    """Update the position of a particle based on its velocity."""
    for i in range(len(particle)):
        particle[i] += velocity[i]
    return particle

def run_pso(n_particles, n_dims, hyperparameters, c1, c2, w, max_iter):
    """Run the PSO algorithm to optimize the hyperparameters of a CNN model."""
    particles = initialize_particles(n_particles,n_dims, hyperparameters)
    velocities = [[0 for j in range(n_dims)] for i in range(n_particles)]
    personal_best_accuracies = [0 for i in range(n_particles)]
    personal_best_positions = particles.copy()
    global_best = particles[0].copy()
    global_best_accuracy = evaluate_model(global_best)
    
    for t in range(max_iter):
        print("Iteration:",t)
        for i in range(n_particles):
            accuracy = evaluate_model(particles[i])
            if accuracy > personal_best_accuracies[i]:
                personal_best_accuracies[i] = accuracy
                personal_best_positions[i] = particles[i].copy()
                if accuracy > global_best_accuracy:
                    global_best_accuracy = accuracy
                    global_best = particles[i].copy()
        for i in range(n_particles):
            velocity = update_velocity(particles[i], velocities[i], personal_best_positions[i], global_best, c1,c2, w)
            particle = update_particle(particles[i], velocity)
            particles[i] = particle

    return global_best, global_best_accuracy

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-09-25T01:20:56.744473Z","iopub.execute_input":"2023-09-25T01:20:56.744961Z","iopub.status.idle":"2023-09-25T01:20:56.751178Z","shell.execute_reply.started":"2023-09-25T01:20:56.744918Z","shell.execute_reply":"2023-09-25T01:20:56.749875Z"}}
imgd=ImageDataGenerator(rescale=1/255)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-09-25T01:20:59.816707Z","iopub.execute_input":"2023-09-25T01:20:59.817101Z","iopub.status.idle":"2023-09-25T01:21:06.329538Z","shell.execute_reply.started":"2023-09-25T01:20:59.817068Z","shell.execute_reply":"2023-09-25T01:21:06.328601Z"}}
tumor_dataset=imgd.flow_from_directory('../input/breastcancermasses/Dataset of Mammography with Benign Malignant Breast Masses/Dataset of Mammography with Benign Malignant Breast Masses/MIAS Dataset')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-09-25T01:21:08.501715Z","iopub.execute_input":"2023-09-25T01:21:08.502669Z","iopub.status.idle":"2023-09-25T01:21:08.512074Z","shell.execute_reply.started":"2023-09-25T01:21:08.502626Z","shell.execute_reply":"2023-09-25T01:21:08.510905Z"}}
tumor_dataset.class_indices

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-09-25T01:21:11.387473Z","iopub.execute_input":"2023-09-25T01:21:11.387845Z","iopub.status.idle":"2023-09-25T01:21:11.415745Z","shell.execute_reply.started":"2023-09-25T01:21:11.387815Z","shell.execute_reply":"2023-09-25T01:21:11.414162Z"}}
classes=pd.DataFrame(tumor_dataset.classes)
classes.value_counts()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-09-25T01:21:36.253204Z","iopub.execute_input":"2023-09-25T01:21:36.253701Z","iopub.status.idle":"2023-09-25T01:21:36.260351Z","shell.execute_reply.started":"2023-09-25T01:21:36.253665Z","shell.execute_reply":"2023-09-25T01:21:36.258936Z"}}
path_train = '../input/breastcancermasses/Dataset of Mammography with Benign Malignant Breast Masses/Dataset of Mammography with Benign Malignant Breast Masses/DDSM Dataset'
path_test = '../input/breastcancermasses/Dataset of Mammography with Benign Malignant Breast Masses/Dataset of Mammography with Benign Malignant Breast Masses/DDSM Dataset'

# %% [markdown]
# **other version of code**

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-09-25T01:21:38.694917Z","iopub.execute_input":"2023-09-25T01:21:38.695476Z","iopub.status.idle":"2023-09-25T01:21:38.701948Z","shell.execute_reply.started":"2023-09-25T01:21:38.695429Z","shell.execute_reply":"2023-09-25T01:21:38.700860Z"}}
data_dir = pathlib.Path(path_test)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-09-25T01:21:41.764353Z","iopub.execute_input":"2023-09-25T01:21:41.764734Z","iopub.status.idle":"2023-09-25T01:21:41.787160Z","shell.execute_reply.started":"2023-09-25T01:21:41.764704Z","shell.execute_reply":"2023-09-25T01:21:41.786051Z"}}
class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))
class_names

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-09-25T01:21:45.166435Z","iopub.execute_input":"2023-09-25T01:21:45.167617Z","iopub.status.idle":"2023-09-25T01:21:45.176880Z","shell.execute_reply.started":"2023-09-25T01:21:45.167578Z","shell.execute_reply":"2023-09-25T01:21:45.175596Z"}}
benignPath = os.path.join(data_dir,'Benign Masses')
malignantPath = os.path.join(data_dir,'Malignant Masses')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-09-25T01:21:48.637511Z","iopub.execute_input":"2023-09-25T01:21:48.637920Z","iopub.status.idle":"2023-09-25T01:21:50.255165Z","shell.execute_reply.started":"2023-09-25T01:21:48.637887Z","shell.execute_reply":"2023-09-25T01:21:50.254285Z"}}
imageCount = len(list(data_dir.glob('*/*.png')))
imageCount

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-09-25T01:21:53.292044Z","iopub.execute_input":"2023-09-25T01:21:53.292568Z","iopub.status.idle":"2023-09-25T01:22:20.787466Z","shell.execute_reply.started":"2023-09-25T01:21:53.292529Z","shell.execute_reply":"2023-09-25T01:22:20.786248Z"}}
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

# %% [code] {"_kg_hide-output":true,"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-09-25T01:22:31.534725Z","iopub.execute_input":"2023-09-25T01:22:31.535127Z","iopub.status.idle":"2023-09-25T02:26:51.576866Z","shell.execute_reply.started":"2023-09-25T01:22:31.535094Z","shell.execute_reply":"2023-09-25T02:26:51.574990Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2023-09-01T23:17:27.820941Z","iopub.execute_input":"2023-09-01T23:17:27.821335Z","iopub.status.idle":"2023-09-01T23:17:28.098092Z","shell.execute_reply.started":"2023-09-01T23:17:27.821303Z","shell.execute_reply":"2023-09-01T23:17:28.097016Z"},"jupyter":{"outputs_hidden":false}}
from tensorflow.keras import layers 
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

# %% [code] {"execution":{"iopub.status.busy":"2023-09-01T23:17:45.247374Z","iopub.execute_input":"2023-09-01T23:17:45.247772Z","iopub.status.idle":"2023-09-01T23:17:45.271800Z","shell.execute_reply.started":"2023-09-01T23:17:45.247739Z","shell.execute_reply":"2023-09-01T23:17:45.270572Z"},"jupyter":{"outputs_hidden":false}}
model.compile(optimizer="Adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"])

# %% [code] {"jupyter":{"outputs_hidden":false}}
history = model.fit(train_data,
                    epochs=2,
                    validation_data=val_data, 
                    batch_size=32)

# %% [code] {"execution":{"iopub.status.busy":"2023-09-01T23:18:23.322441Z","iopub.execute_input":"2023-09-01T23:18:23.322858Z","iopub.status.idle":"2023-09-01T23:18:23.399641Z","shell.execute_reply.started":"2023-09-01T23:18:23.322826Z","shell.execute_reply":"2023-09-01T23:18:23.398527Z"},"jupyter":{"outputs_hidden":false}}

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

# %% [code] {"execution":{"iopub.status.busy":"2023-09-01T23:18:26.352895Z","iopub.execute_input":"2023-09-01T23:18:26.353292Z","iopub.status.idle":"2023-09-01T23:18:26.368205Z","shell.execute_reply.started":"2023-09-01T23:18:26.353260Z","shell.execute_reply":"2023-09-01T23:18:26.366990Z"},"jupyter":{"outputs_hidden":false}}
 model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])

# %% [code] {"execution":{"iopub.status.busy":"2023-09-01T23:18:28.941739Z","iopub.execute_input":"2023-09-01T23:18:28.942101Z","iopub.status.idle":"2023-09-01T23:19:42.879038Z","shell.execute_reply.started":"2023-09-01T23:18:28.942073Z","shell.execute_reply":"2023-09-01T23:19:42.877866Z"},"jupyter":{"outputs_hidden":false}}
    history=model.fit (train_data,epochs=3, validation_data=val_data)

# %% [code] {"execution":{"iopub.status.busy":"2023-09-01T23:34:29.067140Z","iopub.execute_input":"2023-09-01T23:34:29.067514Z","iopub.status.idle":"2023-09-01T23:34:31.174014Z","shell.execute_reply.started":"2023-09-01T23:34:29.067485Z","shell.execute_reply":"2023-09-01T23:34:31.172336Z"},"jupyter":{"outputs_hidden":false}}
model.evaluate(val_data)

# %% [code] {"jupyter":{"outputs_hidden":false}}
model.summary()

# %% [code] {"execution":{"iopub.status.busy":"2023-09-01T23:36:08.632578Z","iopub.execute_input":"2023-09-01T23:36:08.632958Z","iopub.status.idle":"2023-09-01T23:36:08.694939Z","shell.execute_reply.started":"2023-09-01T23:36:08.632930Z","shell.execute_reply":"2023-09-01T23:36:08.693515Z"},"jupyter":{"outputs_hidden":false}}
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

# %% [code] {"execution":{"iopub.status.busy":"2023-09-01T23:39:58.878884Z","iopub.execute_input":"2023-09-01T23:39:58.879288Z","iopub.status.idle":"2023-09-01T23:39:59.800460Z","shell.execute_reply.started":"2023-09-01T23:39:58.879258Z","shell.execute_reply":"2023-09-01T23:39:59.799550Z"},"jupyter":{"outputs_hidden":false}}
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

# %% [code] {"execution":{"iopub.status.busy":"2023-09-01T23:40:40.454213Z","iopub.execute_input":"2023-09-01T23:40:40.454614Z","iopub.status.idle":"2023-09-01T23:40:40.556168Z","shell.execute_reply.started":"2023-09-01T23:40:40.454584Z","shell.execute_reply":"2023-09-01T23:40:40.555069Z"},"jupyter":{"outputs_hidden":false}}
test_imgs, test_labels = next(val_data)

# %% [code] {"execution":{"iopub.status.busy":"2023-09-01T23:40:42.671855Z","iopub.execute_input":"2023-09-01T23:40:42.672243Z","iopub.status.idle":"2023-09-01T23:40:43.152976Z","shell.execute_reply.started":"2023-09-01T23:40:42.672214Z","shell.execute_reply":"2023-09-01T23:40:43.152039Z"},"jupyter":{"outputs_hidden":false}}
# ENSEMBLE PREDICTIONS AND SUBMIT
results = np.zeros( (624,2) ) 
for j in range(5):
    results = model.predict(test_imgs)
 #   results = results + model[j].predict_generator(test_imgs, steps=1, verbose=0)
results = np.argmax(results,axis = -1)

# %% [code] {"execution":{"iopub.status.busy":"2023-09-01T23:40:45.236667Z","iopub.execute_input":"2023-09-01T23:40:45.237031Z","iopub.status.idle":"2023-09-01T23:40:45.244259Z","shell.execute_reply.started":"2023-09-01T23:40:45.237004Z","shell.execute_reply":"2023-09-01T23:40:45.242778Z"},"jupyter":{"outputs_hidden":false}}
results

# %% [code] {"execution":{"iopub.status.busy":"2023-09-01T23:40:48.614106Z","iopub.execute_input":"2023-09-01T23:40:48.614737Z","iopub.status.idle":"2023-09-01T23:40:48.620906Z","shell.execute_reply.started":"2023-09-01T23:40:48.614701Z","shell.execute_reply":"2023-09-01T23:40:48.619885Z"},"jupyter":{"outputs_hidden":false}}
test_labels = np.argmax(test_labels,axis = -1)
test_labels

# %% [code] {"execution":{"iopub.status.busy":"2023-09-01T23:40:51.619273Z","iopub.execute_input":"2023-09-01T23:40:51.619706Z","iopub.status.idle":"2023-09-01T23:40:51.626396Z","shell.execute_reply.started":"2023-09-01T23:40:51.619675Z","shell.execute_reply":"2023-09-01T23:40:51.625378Z"},"jupyter":{"outputs_hidden":false}}
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score

cm = confusion_matrix(y_true=test_labels, y_pred=results)

# %% [code] {"execution":{"iopub.status.busy":"2023-09-01T23:40:54.986340Z","iopub.execute_input":"2023-09-01T23:40:54.986734Z","iopub.status.idle":"2023-09-01T23:40:54.997123Z","shell.execute_reply.started":"2023-09-01T23:40:54.986705Z","shell.execute_reply":"2023-09-01T23:40:54.995811Z"},"jupyter":{"outputs_hidden":false}}
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

# %% [code] {"execution":{"iopub.status.busy":"2023-09-01T23:41:28.610498Z","iopub.execute_input":"2023-09-01T23:41:28.610911Z","iopub.status.idle":"2023-09-01T23:41:28.831090Z","shell.execute_reply.started":"2023-09-01T23:41:28.610873Z","shell.execute_reply":"2023-09-01T23:41:28.829849Z"},"jupyter":{"outputs_hidden":false}}
import itertools
cm_plot_labels = ['benign', 'malignat']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='')
