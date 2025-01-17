# -*- coding: utf-8 -*-
"""jawad.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/126oSxJZYwFPGM-YDm4fdtHlnN2nnw8-y
"""

!pip install keras-tuner
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras_tuner
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from google.colab import drive
drive.mount('/content/drive')

# !unrar x /content/drive/MyDrive/datasets/images.rar

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2 as cv
import tensorflow

labels_df=pd.read_csv('/content/drive/MyDrive/datasets/image_label.csv')

labels_df.head()

import os
os.listdir('/content/')

# Define the directory where images are stored
image_directory = '/content/images/'

# Assuming the CSV file has a column with image filenames (e.g., 'image_name')
labels_df['image_path'] = labels_df['Image_name'].apply(lambda x: os.path.join(image_directory, x))

labels_df['image_path'] = labels_df['Image_name'].apply(lambda x: f"/content/images/{x}.png")

labels_df

labels_df['path_exists'] = labels_df['image_path'].apply(lambda x: os.path.exists(x))
print(labels_df[['image_path', 'path_exists']].head())

labels_df['Plane'].unique()

# Plot class distribution
labels_df['Plane'].value_counts().plot(kind='bar')
plt.xlabel('Plane')
plt.ylabel('Count')
plt.title('Distribution of Planes in Dataset')
plt.show()

# Check if each image path is valid
valid_paths = labels_df['image_path'].apply(lambda x: os.path.exists(x))
if not valid_paths.all():
    print("Some image paths are incorrect or missing.")

# Print the first few image paths to check if they are correct
print(labels_df['image_path'].head())

# Test if the first image path can be opened
from PIL import Image
img = Image.open(labels_df['image_path'].iloc[0])
img.show()

labels_df['Plane'].value_counts()

femur_df = labels_df[labels_df['Plane'] == 'Fetal femur']
oversampled_femur_df = femur_df.sample(n=2800, replace=True)  # Increase by 256 samples
labels_df = pd.concat([labels_df, oversampled_femur_df])

abdomen_df = labels_df[labels_df['Plane'] == 'Fetal abdomen']
oversampled_abdomen_df = abdomen_df.sample(n=2700, replace=True)  # Increase by 256 samples
labels_df = pd.concat([labels_df, oversampled_abdomen_df])

brain_df = labels_df[labels_df['Plane'] == 'Fetal brain']
oversampled_brain_df = brain_df.sample(n=2650, replace=True)  # Increase by 256 samples
labels_df = pd.concat([labels_df, oversampled_brain_df])


thorax_df = labels_df[labels_df['Plane'] == 'Fetal thorax']
oversampled_thorax_df = thorax_df.sample(n=2850, replace=True)  # Increase by 256 samples
labels_df = pd.concat([labels_df, oversampled_thorax_df])

labels_df['Plane'].value_counts()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras_tuner import RandomSearch
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D



from sklearn.model_selection import train_test_split

# Split data into train and test sets
train_df, test_df = train_test_split(labels_df, test_size=0.2, stratify=labels_df['Plane'], random_state=42)

# Further split train_df into training and validation sets
train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['Plane'], random_state=42)

train_df.shape, val_df.shape, test_df.shape

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_df['Plane']),
    y=train_df['Plane']
)
class_weights = dict(enumerate(class_weights))

# Define ImageDataGenerator with data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    rotation_range=20,            # Random rotation
    width_shift_range=0.2,        # Horizontal shift
    height_shift_range=0.2,       # Vertical shift
    zoom_range=0.2,               # Random zoom
    horizontal_flip=True,         # Flip horizontally
    validation_split=0.2,
    fill_mode='nearest'
)

# Training generator
train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="image_path",
    y_col="Plane",
    target_size=(300, 300),       # Resize images
    batch_size=32,
    class_mode="categorical",
    subset='training'
)

# Validation generator
val_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="image_path",
    y_col="Plane",
    target_size=(300, 300),
    batch_size=32,
    class_mode="categorical",
    subset='validation'
)

# Test generator (no augmentation, just rescaling)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="image_path",
    y_col="Plane",
    target_size=(300, 300),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add

def resdiual_block(x,filters,kernel_size=3,strides=1):
    res=x
    x=Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding='same')(x)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)

    x=Conv2D(filters=filters,kernel_size=kernel_size,strides=1,padding='same')(x)
    x=BatchNormalization()(x)

    if res.shape[-1]!=x.shape[-1] or strides!=1:
        res=Conv2D(filters=x.shape[-1],kernel_size=1,strides=strides,padding='same')(res)

    x=Add()([x,res])
    x=Activation('relu')(x)
    return x

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

def build_resnet(input_shape=(300,300,3),num_classes=4):
  inputs=Input(shape=input_shape)

  x=Conv2D(filters=64,kernel_size=(7,7),strides=2,padding='same')(inputs)
  x=BatchNormalization()(x)
  x=Activation('relu')(x)
  x=MaxPooling2D(pool_size=(3,3),strides=2,padding='same')(x)
  for filters in [64,128,256,512]:
    for _ in range(3):
      x=resdiual_block(x,filters,strides=1 if filters==x.shape[-1] else 2)

  x=GlobalAveragePooling2D()(x)

  outputs=Dense(num_classes,activation='softmax')(x)
  model=Model(inputs,outputs)
  return model

resnet_model=build_resnet(input_shape=(300,300,3),num_classes=4)
resnet_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

resnet_model.summary()

from tensorflow.keras.callbacks import  EarlyStopping,ReduceLROnPlateau
earlystopping=EarlyStopping(monitor='val_loss',patience=5,verbose=1,restore_best_weights=True)
lr_scheduler=ReduceLROnPlateau(monitor='val_loss',factor=0.1,)

history=resnet_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[earlystopping,lr_scheduler],
    class_weight=class_weights
)

# Evaluate on the test set
test_loss, test_accuracy = resnet_model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

import matplotlib.pyplot as plt

# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Evaluate on the test set
test_loss, test_accuracy = resnet_model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Predict on the test generator
predictions =  resnet_model.predict(test_generator)

# Get the predicted classes
predicted_classes = np.argmax(predictions, axis=1)

# Get the true classes
true_classes = test_generator.classes  # True class indices from the generator

# Map numeric indices to class labels
index_to_class = {v: k for k, v in test_generator.class_indices.items()}  # e.g., {0: 'Fetal brain', 1: 'Fetal abdomen', ...}

# Identify misclassified indices
misclassified = np.where(true_classes != predicted_classes)[0]
print(f"Number of misclassified samples: {len(misclassified)}")

from PIL import Image
import matplotlib.pyplot as plt

# Display misclassified images
for i in misclassified[:10]:  # Adjust the number (e.g., 10) based on how many you want to view
    # Get the misclassified image path
    img_path = test_generator.filepaths[i]

    # Get the true and predicted labels
    true_label = index_to_class[true_classes[i]]
    predicted_label = index_to_class[predicted_classes[i]]

    # Load and show the image
    img = Image.open(img_path)
    plt.imshow(img)
    plt.title(f"True: {true_label}, Predicted: {predicted_label}")
    plt.axis('off')
    plt.show()

from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Function to preprocess an image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(300, 300))  # Resize to match model input
    img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Define the random image path
random_image_path = "/content/images/Patient01018_Plane3_2_of_3.png"  # Replace with your image path

# Preprocess the image
preprocessed_img = preprocess_image(random_image_path)

# Predict the class
prediction = resnet_model.predict(preprocessed_img)

# Map the predicted class to the corresponding label
predicted_class = np.argmax(prediction, axis=1)[0]  # Index of the highest probability
predicted_label = index_to_class[predicted_class]  # Convert index to class label

print(f"Predicted Class: {predicted_label} (Confidence: {np.max(prediction) * 100:.2f}%)")

import matplotlib.pyplot as plt
from PIL import Image

img = Image.open(random_image_path)

plt.imshow(img)
plt.title(f"Predicted: {predicted_label} (Confidence: {np.max(prediction) * 100:.2f}%)")
plt.axis('off')
plt.show()

import os
import random
import matplotlib.pyplot as plt
from PIL import Image

# Directory containing your test images
test_images_dir = "/content/images"  # Replace with the directory containing your test images

# Get all image file paths from the directory
all_images = [os.path.join(test_images_dir, img) for img in os.listdir(test_images_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]

# Randomly select 10 images
random_images = random.sample(all_images, 10)

# Predict and visualize each image
for image_path in random_images:
    # Preprocess the image
    preprocessed_img = preprocess_image(image_path)

    # Predict the class
    prediction = resnet_model.predict(preprocessed_img)
    predicted_class = np.argmax(prediction, axis=1)[0]  # Index of the highest probability
    predicted_label = index_to_class[predicted_class]  # Convert index to class label
    confidence = np.max(prediction) * 100

    # Load and display the image
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_label} (Confidence: {confidence:.2f}%)")
    plt.axis('off')
    plt.show()

    print(f"Image: {os.path.basename(image_path)}")
    print(f"Predicted Label: {predicted_label}, Confidence: {confidence:.2f}%")
    print("-" * 50)

from tensorflow.keras.models import save_model

# Save the model to a file
resnet_model.save('fetal_ultrasound_model.h5')

import pickle

pickle.dump(resnet_model,open('fetal_ultrasound_model.pkl','wb'))

