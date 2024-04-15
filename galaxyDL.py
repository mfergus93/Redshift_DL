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

def pavlidis(img):
    c=0
    x=img.shape[1]//2
    y=img.shape[0]//2

    while c == 0:
        x+=1
        if img[y,x]!=0:
            c=1

    b1_value,b2_value,b3_value = 0,0,0
    b1_coord,b2_coord,b3_coord= 0,0,0
    directions = 'up', 'right', 'down', 'left'
    direction='right'
    input=(y,x)
    result=np.array([input])

    c=0
    start_pos=input
    while c<2:

        y,x=input
        if input==list(start_pos):
            c=c+1

        if direction =='up':
            b1_value, b1_coord=img[y-1,x-1], [y-1,x-1]
            b2_value, b2_coord=img[y-1,x], [y-1,x]
            b3_value, b3_coord=img[y-1,x+1], [y-1,x+1]
        elif direction =='right':
            b1_value, b1_coord=img[y-1,x+1], [y-1,x+1]
            b2_value, b2_coord=img[y,x+1], [y,x+1]
            b3_value, b3_coord=img[y+1,x+1], [y+1,x+1]
        elif direction =='down':
            b1_value, b1_coord=img[y+1,x+1], [y+1,x+1]
            b2_value, b2_coord=img[y+1,x], [y+1,x]
            b3_value, b3_coord=img[y+1,x-1], [y+1,x-1]
        elif direction =='left':
            b1_value, b1_coord=img[y+1,x-1], [y+1,x-1]
            b2_value, b2_coord=img[y,x-1], [y,x-1]
            b3_value, b3_coord=img[y-1,x-1], [y-1,x-1]

        block_values= b1_value,b2_value,b3_value
        block_coords=b1_coord,b2_coord,b3_coord

        if b1_value==255:
            bimg[y,x]=255
            direction=directions[(((directions.index(direction)-1))%4)]
        elif b1_value==0 and b2_value==0 and b3_value==0:
            direction=directions[(((directions.index(direction)+1))%4)]
        elif b2_value==255:
            bimg[y,x]=255
        elif b3_value==255:
            bimg[y,x]=255

        for i, value in enumerate(reversed(block_values)):
            if value==255:
                input = block_coords[2-i]

        result=np.append(result,[input],axis=0)

    result, ind=np.unique(result,axis=0, return_index=True)
    result=result[np.argsort(ind)]
    return (result)
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
#%% Keras VGGFace Model

from tensorflow.keras import Model
from keras.layers import Flatten, Dense, Input
from keras_vggface.vggface import VGGFace

c_batch = 32
c_classes = y_train.shape[1]
c_epochs = 10
c_img_size = 64
c_loss = 'categorical_crossentropy'
c_opt = 'adam'

vgg_model_0 = VGGFace(include_top = False, input_shape = (c_img_size, c_img_size, 3))
last_layer = vgg_model_0.get_layer('pool5').output
x = Flatten(name = 'flatten')(last_layer)
vgg_model_0.trainable = False
out = Dense(c_classes, activation = 'softmax', name = 'fc8')(x)

vgg_model_1 = Model(vgg_model_0.input, out)

for layer in range(len(vgg_model_1.layers)):
    
    if "BatchNormalization" in str(layer):
        
        vgg_model_1.layers[layer]= keras.Batch_Normalization  (Exzperimental param=xyz)

vgg_model_1.compile(optimizer = c_opt,
                    loss = c_loss,
                    metrics = ['accuracy'])

print('\nFit model on training data.')
history = vgg_model_1.fit(x_train, y_train,
                          batch_size = c_batch,
                          epochs = c_epochs,
                          validation_data = (x_test, y_test))
history.history
#%%

# from keract import display_activations, get_activations
# activations = get_activations(model, [x_test[None, 0],x_test[None, 0]])
# display_activations(activations, save=False)

def resblock(x, filters, n):
    #Creates n resblocks for a single filter size
    
    for i in range(n):
        
        fx=keras.layers.Conv2D(filters=filters, kernel_size= (3,3), padding='same')(x)
        fx=keras.layers.BatchNormalization()(fx)
        fx=keras.layers.ReLU()(fx)
        fx=keras.layers.Conv2D(filters=filters, kernel_size=(3,3), padding='same')(fx)
        fx=keras.layers.BatchNormalization()(fx)
        
        x=keras.layers.Conv2D(filters=filters, kernel_size=(1,1), padding='same')(x)
        x=keras.layers.BatchNormalization()(x)
        x=keras.layers.ReLU()(x)
        
        x=keras.layers.Add()([x,fx])
        x=keras.layers.ReLU()(x)
        x=keras.layers.BatchNormalization()(x)
        
    return x

def resnet(n):
    #Creates a resnet by calling resblock and passing n
    
    x=keras.layers.Input(shape=(img_size, img_size, 3))
    fx=keras.layers.Conv2D(filters=16, kernel_size=(3,3), padding='same')(x)
    fx=keras.layers.BatchNormalization()(fx)
    fx = keras.layers.Activation("relu")(fx)
    
    fx=resblock(fx, 16, n)
    fx=resblock(fx, 32, n)
    fx=resblock(fx, 64, n)
    
    fx=keras.layers.AveragePooling2D(pool_size=(8,8))(fx)
    fx=keras.layers.Flatten()(fx)
    output=keras.layers.Dense(len(y_train[0,:]),activation='softmax')(fx)

    model=keras.Model(x, output)
    return model

#Experiment
acc_list=[]

# for data in datasets:
#     for n in range(1,6):
#         for opt in ('adam', 'SGD'):
#             keras.backend.clear_session()
            
#             x_train=(data[0][0])
#             x_test=(data[1][0])
            
#             y_train=keras.utils.to_categorical(data[0][1])
#             y_test=keras.utils.to_categorical(data[1][1])

#             resnet_model=resnet(n)
#             resnet_model.compile(loss = 'categorical_crossentropy', optimizer=opt, metrics = ['accuracy'])
#             history=resnet_model.fit(x_train, y_train, epochs = 30, validation_data =(x_test, y_test), batch_size = batch_size)
#             test_loss, test_acc = resnet_model.evaluate(x_test, y_test, batch_size = batch_size)
#             train_loss, train_acc=resnet_model.evaluate(x_train, y_train, batch_size = batch_size)
            
#             acc_list.append([test_acc, train_acc, n, opt])
            
#             plt.plot(history.history['accuracy'])
#             plt.plot(history.history['val_accuracy'])
#             plt.title('Model Accuracy')
#             plt.ylabel('Accuracy'), plt.xlabel('Epoch')
#             plt.legend(['Train', 'Test'], loc='upper left')
#             plt.show()
            
#             plt.plot(history.history['loss'])
#             plt.plot(history.history['val_loss'])
#             plt.title('Model Loss')
#             plt.ylabel('Loss'), plt.xlabel('Epoch')
#             plt.legend(['Train', 'Test'], loc='upper left')
#             plt.show()

n=3
opt='adam'
resnet_model=resnet(n)
resnet_model.compile(loss = 'categorical_crossentropy', optimizer=opt, metrics = ['accuracy'])
history=resnet_model.fit(x_train, y_train, epochs = 30, validation_data =(x_test, y_test), batch_size = batch_size)
# test_loss, test_acc = resnet_model.evaluate(x_test, y_test, batch_size = batch_size)
# train_loss, train_acc=resnet_model.evaluate(x_train, y_train, batch_size = batch_size)

#%%
#Experiment 2

def pretrained_resnet(nodes,pool):
    
    keras.backend.clear_session()

    x= keras.layers.Input(shape=(32,32,3))
    fx = keras.layers.UpSampling2D(size=(7,7))(x)

    fx=keras.applications.resnet.ResNet50(input_shape=(224,224,3), include_top=False,
        weights='imagenet')(fx)
    
    fx=keras.layers.AveragePooling2D(pool)(fx)
    fx=keras.layers.Flatten()(fx)
    fx = keras.layers.Dense(nodes[0], activation="relu")(fx)
    fx = keras.layers.Dense(nodes[1], activation="relu")(fx)
    output=keras.layers.Dense(len(y_train[0,:]),activation='softmax')(fx)
    
    model=keras.Model(x,output)
    return model


nodes_list=[(1024, 2048)] #(256, 512), (512, 1024)
pool_list=[3]
acc_list_pt=[]
opt='SGD'