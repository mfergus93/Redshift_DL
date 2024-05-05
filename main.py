#main routine
import os
import cv2
import numpy as np
import pandas as pd
import pickle

from Flood_Fill import flood_fill
from Outlining import pavlidis, fillarea
from y_network import y_network
from color_Features import extract_color_histogram


# labels = pd.read_csv(r'C:\Users\Matt\Desktop\dev\galaxy.csv')

# folder_path= r'D:\galactic_images_raw'
folder_path=os.path.dirname(os.path.realpath(__file__))
os.chdir(folder_path)

image_dir=r'D:\galactic_images_raw'

#this can probably get moved to a preprocessing file at some point
def preprocess_directory(path): #probably change this to preprocess image or batch of images
    
    output_list=[]
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):# or filename.endswith('.png'):
            
            file_path = os.path.join(path, filename)
            img=cv2.imread(file_path)
            gray_img=cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            flood_img=flood_fill(gray_img)
            # contour_img=pavlidis(flood_img)
            histogram_image, h_hist, s_hist, v_hist = extract_color_histogram(img, flood_img)
            # output_list.append([img, flood_img, histogram_image, h_hist, s_hist, v_hist])
            feature_array=np.concatenate((h_hist, s_hist, v_hist), axis=1)
            output_list.append(feature_array)
            
    return output_list
                    
features = preprocess_directory(folder_path)

#write all preprocessed images as pickled features for the model to a directory 
def write_features(features):
    with open('features.pkl','wb') as f:
        pickle.dump(features, f)

write_features(features)

#%%

#call in the prerocessed features for the model
def load_features(feature_directory):
    
    with open(r'features.pkl', 'rb') as input_file:
        features=pickle.load(input_file)
    x=features
    y='labels'
    
    return features
x=features
y=[0,1]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.50, random_state=1)
    
    
    # model, accuracy, history = y_network(x_train, x_test, y_train, y_test, 30, 1, 'adam', 0.35)
    
    # X_test='novel contours and color features'
    # y_pred = model.predict([X_test, X_test])
    # results = pd.DataFrame({
    #     'True Regression Value': y_test.flatten(),  # Flatten y_test if it's a multi-dimensional array
    #     'Predicted Regression Value': y_pred.flatten()  # Flatten y_pred for consistency
    # })
    # results['Percentage Difference'] = (abs(results['True Regression Value'] - results['Predicted Regression Value']) / results['True Regression Value']) * 100
    


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Define your model
model = Sequential([
    Dense(64, activation='relu', input_shape=(input_shape,)),
    Dropout(0.2),  # Optional: helps prevent overfitting
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1)  # Output layer with 1 neuron for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')

