#%%
import os
import cv2
import mtcnn
import time
import random
import numpy as np
import pandas as pd
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt

#%%
# Read the image
import cv2
import numpy as np

image = cv2.imread(r'D:\galactic_images\299491326712375296.jpg')  # Replace 'your_image.jpg' with the path to your image

# Convert the image from BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshape the image to a 2D array of pixels
pixels = image.reshape((-1, 3))

# Convert pixel values to list of tuples
pixels = [tuple(pixel) for pixel in pixels]

# Get unique colors
unique_colors = list(set(pixels))

print("Unique colors in the image:")
for color in unique_colors:
    print(color)
    # Create a square image of the color
    color_img = np.zeros((100, 100, 3), dtype=np.uint8)
    color_img[:, :] = color
    
    # Plot the color
    plt.imshow(color_img)
    plt.title(f"RGB: {color}")
    plt.axis('off')
    plt.show()
# Plot unique colors


#%%
start=time.perf_counter()
df = pd.read_csv(r'C:\Users\Matt\Desktop\dev\galaxy.csv')

#Load Images
def load_image(image_path):
    # Open the image using PIL (Python Imaging Library)
    img = Image.open(image_path)
    # Convert the image to a numpy array
    img_array = np.array(img)
    return img_array

def get_image_path(phot_id):
    return f'D:/galactic_images/{phot_id}.jpg'

def crop_images(images, target_size):
    cropped_images = []
    for image in images:
        height, width, _ = image.shape
        # Calculate the coordinates for cropping the center of the image
        start_y = max(0, height // 2 - target_size[0] // 2)
        end_y = start_y + target_size[0]
        start_x = max(0, width // 2 - target_size[1] // 2)
        end_x = start_x + target_size[1]
        # Crop the image
        cropped_image = image[start_y:end_y, start_x:end_x]
        cropped_images.append(cropped_image)
    return np.array(cropped_images)

# Assuming you have loaded your images into X_train and X_test numpy arrays
# X_train and X_test should have shape (num_samples, height, width, channels)

# Define the target size
target_size = (64, 64)

x = []  # To store images
y = []  # To store redshift values

for index, row in df.iterrows():
    phot_id = row['specobjid']
    image_path = get_image_path(phot_id)
    try:
        img_array = load_image(image_path)
        x.append(img_array)
        y.append(row['redshift'])
    except FileNotFoundError:
        # Handle missing images here
        pass

# Convert lists to numpy arrays
x = np.array(x)
x = crop_images(x,target_size)
x = x/255.0
y = np.array(y)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#%% Ynet

img_size=64
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, concatenate

#ynetwork function
def ynetwork(x_train, x_test, y_train, y_test, epochs, layer, optimizer, dropout):
    tf.keras.backend.clear_session()

    #left branch
    left_input = Input(shape=input_shape)
    xl=left_input
    filters=n_filters

    for l in range(layer):
        xl=Conv2D(filters=filters, kernel_size= (3,3), padding='same', activation='relu', dilation_rate=1)(xl)
        xl=Dropout(dropout)(xl)
        xl=MaxPooling2D(pool_size=(2,2))(xl)
        filters*=2

    #right branch
    right_input=Input(shape=input_shape)
    xr=right_input
    filters=n_filters

    for l in range(layer):
        xr=Conv2D(filters=filters, kernel_size = (3,3), padding='same', activation='relu', dilation_rate=2)(xr)
        xr=Dropout(dropout)(xr)
        xr=MaxPooling2D(pool_size=(2,2))(xr)
        filters*=2

    #perceptron
    x=concatenate([xl, xr])
    x=Flatten()(x)
    x=Dropout(dropout)(x)
    output=Dense(1, activation='linear')(x)

    model=Model([left_input, right_input], output)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error'])
    history=model.fit([x_train, x_train], y_train, epochs=epochs, validation_data =([x_test, x_test], y_test), batch_size=batch_size)
    test_loss, test_acc=model.evaluate([x_test, x_test], y_test, batch_size=batch_size, verbose=0)
    train_loss, train_acc=model.evaluate([x_train, x_train], y_train, batch_size=batch_size, verbose=0)
    
    return model, test_acc, history

# for layer in layer_list:
#     for optimizer in optimizer_list:
#         for dropout in dropout_list:
#             tf.keras.backend.clear_session()
#             model, accuracy, history = ynetwork(x_train, x_test, y_train, y_test, 30, layer, optimizer, dropout)
#             model_list.append([accuracy, layer, optimizer, dropout])

#             plt.plot(history.history['accuracy'])
#             plt.plot(history.history['val_accuracy'])
#             plt.title('model accuracy')
#             plt.ylabel('accuracy'), plt.xlabel('epoch')
#             plt.legend(['train', 'test'], loc='upper left')
#             plt.show()
            
#             plt.plot(history.history['loss'])
#             plt.plot(history.history['val_loss'])
#             plt.title('model loss')
#             plt.ylabel('loss'), plt.xlabel('epoch')
#             plt.legend(['train', 'test'], loc='upper left')
#             plt.show()

#constants
batch_size=8
n_filters=32
label_categories = 2
input_shape= (img_size, img_size, 3)

#experiments
layer_list=[2,3,4]
optimizer_list=['SGD','adam']
dropout_list=[0, 0.333]
model_list=[]

model, accuracy, history = ynetwork(x_train, x_test, y_train, y_test, 30, 1, 'adam', 0.35)
y_pred = model.predict([X_test, X_test])
results = pd.DataFrame({
    'True Regression Value': y_test.flatten(),  # Flatten y_test if it's a multi-dimensional array
    'Predicted Regression Value': y_pred.flatten()  # Flatten y_pred for consistency
})
results['Percentage Difference'] = (abs(results['True Regression Value'] - results['Predicted Regression Value']) / results['True Regression Value']) * 100
