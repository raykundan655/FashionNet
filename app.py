import tensorflow as tf 
import pandas as pd
import numpy as np
from tensorflow.keras import datasets,models,layers,Model   #models: Module for model utilities  Model: Class to build custom models
import matplotlib.pyplot as plt




dataset=datasets.fashion_mnist 

(x_train,y_train),(x_test,y_test)=dataset.load_data()

# print(x_train.shape)

# print(x_train[0].shape)

# print(x_train[0])

# plt.imshow(x_train[0],cmap='gray')
# plt.show()

class_names = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]

print(y_train[0])

# normalizing
x_train=x_train/255.0
x_test=x_test/255.0


# tensorflow model work with 4Dimention input
# TensorFlow CNN models expect 4D input: (batch(row), height, width, channels).

x_train=x_train.reshape(-1,28,28,1) # 1 is grayscale ,first para tell how many row -1 tell find auto dimention
x_test=x_test.reshape(-1,28,28,1)


# data augmentation

data_aug=models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])


model=models.Sequential([
    layers.Input(shape=(28,28,1)),  #if i'm using data aug we have take input previously
    data_aug,
    # layers.Conv2D(128,(3,3),padding='same',activation='relu',input_shape=(28,28,1)),  //if data aug was not there then we should to continue from here
    layers.Conv2D(128,(3,3),padding='same',activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    layers.Conv2D(256,(3,3),activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(10,activation='softmax')

    
])

# 2nd way to create model(functional API)

# inputs=layers.Input(shape=(28,28,1))

# x=data_aug(inputs)

# x=layers.Conv2D(128,(3,3),activation="relu",padding='same')(x)
# x=layers.MaxPooling2D((2,2))(x)
# x=layers.Dropout(0.25)(x)

# x=layers.Conv2D(256,(3,3),activation="relu")(x)
# x=layers.MaxPooling2D((2,2))(x)
# x=layers.Dropout(0.25)(x)

# x=layers.Flatten()(x)
# x=layers.Dense(64,activation="relu")(x)
# x=layers.Dropout(0.25)(x)

# outputs = layers.Dense(10, activation='softmax')(x)

# model = Model(inputs=inputs, outputs=outputs)







model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  #Because your labels are integers (0–9), not one-hot encoded.
    metrics=['accuracy']
)


hist=model.fit(
    x_train,y_train,epochs=10,
     validation_data=(x_test, y_test)
)

print(model.summary())

print(model.evaluate(x_test, y_test))

model.save("fashionClassifierModel.h5")

















