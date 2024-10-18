#main routine
import os
import cv2
import numpy as np
import pandas as pd
import pickle

folder_path=os.path.dirname(os.path.realpath(__file__))
os.chdir(folder_path)

from preproc.color_features import extract_rgb_histogram, extract_hsv_histogram
from flood_fill import flood_fill
# from outlining import pavlidis, fillarea
# from y_network import y_network

# folder_path= r'D:\galactic_images_raw'

image_dir=r'D:\galactic_images_production'
labels = pd.read_csv(r'galaxy.csv')

#this can probably get moved to a preprocessing file at some point
#its written in such a way that all features have to be loaded into memory at once
#if this becomes an issue then we can easily preprocess a batch of images and then write out


                    

#write all preprocessed images as pickled features for the model to a directory 
def write_features(features):
    # Create the features subfolder if it doesn't exist
    if not os.path.exists('features'):
        os.makedirs('features')
    
    # Define the file path to save the features.pkl file inside the features subfolder
    file_path = os.path.join('features', 'features.pkl')

    # Write features to the file
    with open(file_path, 'wb') as f:
        pickle.dump(features, f)


features = preprocess_directory(image_dir)
write_features(features)

#%%

#call in the prerocessed features for the model
def load_features(feature_directory):
    
    features_path = os.path.join(feature_directory, 'features.pkl')
    with open(features_path, 'rb') as input_file:
        features = pickle.load(input_file)
    x = features
    
    labels = pd.read_csv(r'galaxy.csv')
    y=labels.iloc[:, [18, -1]]

    
    return x, y

def strip_jpg_suffix(objid):
    return objid.split('.')[0]  # Splitting at the '.' and taking the first part

def align_features_with_labels(x, y):
    # Create dictionaries for fast lookup by objid
    x_lookup = {int(strip_jpg_suffix(row[0])): row[1:] for row in x}
    y_lookup = {int(row[0]): row[1] for row in y.values}

    aligned_x = []
    aligned_y = []

    for objid, features in x_lookup.items():
        if objid in y_lookup:
            aligned_x.append(features)
            aligned_y.append(y_lookup[objid])
    
    aligned_x = [np.vstack(aligned_x) for aligned_x in aligned_x]
    return np.array(aligned_x), np.array(aligned_y)

def ann(x,y):
    
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Input
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.callbacks import EarlyStopping
    
    input_shape=x[0].shape
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.30, random_state=1)

    # Define your model
    model = Sequential([
        
        Input(shape=input_shape), 
        Dense(64, activation='relu'),
        Dropout(0.2),  # Optional: helps prevent overfitting
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1)  # Output layer with 1 neuron for regression
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])

    # Define early stopping
    # early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model with validation data and early stopping
    history = model.fit(x_train, y_train, epochs=30, batch_size=32, validation_data=(x_test, y_test))#,
                        #callbacks=[early_stopping])

    # Evaluate the model on the test set
    loss, mae, mse = model.evaluate(x_test, y_test)
    print(f'Test loss: {loss}, MAE: {mae}, MSE: {mse}')

    return model, loss, mae, mse, history, x_test, y_test

#%%
x,y=load_features(r'features')
x,y=align_features_with_labels(x,y)
x = x[:, :, 0]

model, loss, mae, mse, history, x_test, y_test = ann(x,y)

y_pred = model.predict(x_test)
errors=np.abs(y_pred-y_test)
epsilon = 1e-8  # Small value to avoid division by zero
percent_error = np.mean(errors / (y_test + epsilon)) * 100

# from sklearn.neural_network import MLPRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.30, random_state=1)

# model = MLPRegressor(hidden_layer_sizes=(64, 32),  # Two hidden layers with 64 and 32 neurons
#                  activation='relu',  # ReLU activation function
#                  solver='adam',  # Optimizer algorithm
#                  max_iter=100,  # Maximum number of iterations
#                  random_state=1)  # Random state for reproducibility

# # Train the model
# model.fit(x_train, y_train)

# # Make predictions
# y_pred = model.predict(x_test)

# # Evaluate the model
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)

# print(f'Mean Absolute Error: {mae}')
# print(f'Mean Squared Error: {mse}')








        # model, accuracy, history = y_network(x_train, x_test, y_train, y_test, 30, 1, 'adam', 0.35)
        
        # X_test='novel contours and color features'
        # y_pred = model.predict([X_test, X_test])
        # results = pd.DataFrame({
        #     'True Regression Value': y_test.flatten(),  # Flatten y_test if it's a multi-dimensional array
        #     'Predicted Regression Value': y_pred.flatten()  # Flatten y_pred for consistency
        # })
        # results['Percentage Difference'] = (abs(results['True Regression Value'] - results['Predicted Regression Value']) / results['True Regression Value']) * 100
        
    
    
#%%
import numpy as np
import matplotlib.pyplot as plt

# Reshape y_pred if it has shape (n, 1)
if len(y_pred.shape) == 2:
    y_pred = y_pred.flatten()

# Plotting Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, s=1)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # Diagonal line for reference
plt.xlabel('Actual Values (y_test)')
plt.ylabel('Predicted Values (y_pred)')
plt.title('Actual vs Predicted')
plt.show()

