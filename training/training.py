"""
Code from notebook. Do not try to read this.

"""


import os
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                          GlobalAveragePooling2D, Input, MaxPooling2D)
from keras.models import Model, Sequential
from sklearn.metrics import accuracy_score
from tensorflow import keras
from tensorflow.keras.applications.xception import (Xception,
                                                    decode_predictions,
                                                    preprocess_input)
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tqdm import tqdm
from xgboost import XGBClassifier

# ____________________1st Model___________________
print("Preparing 1st model")
model = Xception(weights="imagenet", input_shape=(299, 299, 3))

print("Preparing training dataset generator")
train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_ds = train_gen.flow_from_directory(
    "./Training/", target_size=(150, 150), batch_size=32
)

print("Preparing validation dataset generator")
val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_ds = val_gen.flow_from_directory(
    "./Testing/", target_size=(150, 150), batch_size=32, shuffle=False
)

print("Building model...")
base_model = Xception(weights="imagenet", include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False
inputs = keras.Input(shape=(150, 150, 3))
base = base_model(inputs, training=False)
vectors = keras.layers.GlobalAveragePooling2D()(base)
outputs = keras.layers.Dense(4)(vectors)
model = keras.Model(inputs, outputs)


def make_model(learning_rate=0.01):
    base_model = Xception(
        weights="imagenet", include_top=False, input_shape=(150, 150, 3)
    )

    base_model.trainable = False
    inputs = keras.Input(shape=(150, 150, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    outputs = keras.layers.Dense(4)(vectors)
    model = keras.Model(inputs, outputs)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model


scores = {}

print("Tuning learning rate...")
for lr in [0.0001, 0.001, 0.01, 0.1]:
    print(lr)

    model = make_model(learning_rate=lr)
    history = model.fit(train_ds, epochs=2, validation_data=val_ds)
    scores[lr] = history.history

    print()
    print()

print("The best learning rate is 0.01. Lets train more epochs with it.")
learning_rate = 0.01
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
loss = keras.losses.CategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
history = model.fit(train_ds, epochs=10, validation_data=val_ds)
model.save("model1.h5")


# _________________2nd model________________
print("Preparing 2nd model")
labels = ["glioma_tumor", "no_tumor", "meningioma_tumor", "pituitary_tumor"]


def convert_image_to_dataset(file_location):
    label = 0
    df = pd.DataFrame()
    for category in glob(file_location + "/*"):
        for file in tqdm(glob(category + "/*")):
            img_array = cv2.imread(file)
            img_array = cv2.resize(img_array, (224, 224))
            data = pd.DataFrame({"image": [img_array], "label": [label]})
            df = df.append(data)  ##concat
        label += 1
    return df.sample(frac=1).reset_index(drop=True)


train_data = convert_image_to_dataset("./Training")
train_x = np.array(train_data.image.to_list())

print("Building model...")
model_cnn = Sequential()
model_cnn.add(Input(shape=(224, 224, 3)))
model_cnn.add(Conv2D(128, (3, 3)))
model_cnn.add(MaxPooling2D((2, 2)))
model_cnn.add(BatchNormalization())
model_cnn.add(Conv2D(64, (3, 3)))
model_cnn.add(MaxPooling2D((2, 2)))
model_cnn.add(BatchNormalization())
model_cnn.add(Conv2D(32, (3, 3)))
model_cnn.add(MaxPooling2D((2, 2)))
model_cnn.add(BatchNormalization())
model_cnn.add(Flatten())
model_cnn.add(Dense(128, activation="relu"))
model_cnn.add(Dropout(0.2))
model_cnn.add(Dense(64, activation="relu"))
model_cnn.add(Dense(4, activation="softmax"))
model_cnn.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

print("Training model...")
r1 = model_cnn.fit(train_x, train_data.label, validation_split=0.1, epochs=5)
xception_model = Xception(weights="imagenet", include_top=False)
for layers in xception_model.layers:
    layers.trainable = False
x = xception_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.15)(x)
output = Dense(4, activation="softmax")(x)
model2 = Model(inputs=xception_model.input, outputs=output)
model2.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

r2 = model2.fit(train_x, train_data.label, validation_split=0.1, epochs=2)

model2.save("model2.h5")

# ______________________3rd model: adding XGBOOST_____________________-

# ### Take the layer from previous model and train new xgboost model using that layer's output.
print("Preparing Xgboost model...")
new_model = tf.keras.models.Model(
    model2.input, model2.get_layer("global_average_pooling2d").output
)

X_train_features = new_model.predict(train_x)
X_train_features.shape

xgb = XGBClassifier(
    objective="multiclass:softmax", learning_rate=0.1, max_depth=15, n_estimators=500
)
print("Training XGBoost model...")
xgb.fit(X_train_features, train_data.label)

print("Testing...")
test_data = convert_image_to_dataset("./Testing")
test_x = np.array(test_data.image.to_list())
X_test_features = new_model.predict(test_x)
y_pred = xgb.predict(X_test_features)

print("Accuracy:")
print(accuracy_score(y_pred, test_data.label))

xgb.save_model("model_xgb.h5")
