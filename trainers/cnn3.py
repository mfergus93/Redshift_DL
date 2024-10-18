import h5py
import os
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
from photo_ugriz.ugriz_functions import filter_incomplete_bands, stack_bands_by_specobjid

# Load the galaxy.csv DataFrame to get redshift labels
galaxy_df = pd.read_csv('galaxy.csv')

def get_redshift(specobjid, galaxy_df):
    # Look up the redshift using the specobjid from the DataFrame
    result = galaxy_df.loc[galaxy_df['specobjid'] == int(specobjid), 'redshift']
    if len(result) > 0:
        return result.values[0]
    else:
        return None
    
# def resize_image(image_array, size=(224, 224)):
#     # Convert the numpy array to a PIL image
#     img = Image.fromarray(image_array)
#     # Resize the image
#     img_resized = img.resize(size, Image.ANTIALIAS)
#     # Convert back to numpy array
#     return np.array(img_resized)

def resize_image(image_array, size=(224, 224)):
    # Convert to uint8 if necessary
    if image_array.dtype == np.float32:
        image_array = (image_array * 255).astype(np.uint8)  # Scale if necessary

    img = Image.fromarray(image_array)
    img_resized = img.resize(size, Image.ANTIALIAS)
    return np.array(img_resized)

# def resize_batch(images, size=(224, 224)):
#     resized_images = []
#     for img in images:
#         resized_img = resize_image(img[1], size)  # img[1] is the image stack (224, 224, 5)
#         resized_images.append([img[0], resized_img])  # img[0] is galaxy_id
#     return resized_images

def resize_batch(images, size=(224, 224)):
    resized_images = []
    for img in images:
        resized_img = resize_image(img[2], size)  # img[2] is the image array
        resized_images.append([img[0], img[1], resized_img])
    return resized_images

def load_from_hdf5(filename):
    with h5py.File(filename, 'r') as h5f:
        images = []
        for dataset_name in h5f:
            galaxy_id, band = dataset_name.split('_')
            images.append([galaxy_id, band, h5f[dataset_name][()]])
        return images

# def load_from_hdf5(filename):
#     with h5py.File(filename, 'r') as h5f:
#         galaxy_images = {}
#         for dataset_name in h5f:
#             galaxy_id, band = dataset_name.split('_')
#             if galaxy_id not in galaxy_images:
#                 galaxy_images[galaxy_id] = {}
#             galaxy_images[galaxy_id][band] = h5f[dataset_name][()]
        
#         # Now we have a dict of galaxy_id -> band -> image array
#         images = []
#         for galaxy_id, bands in galaxy_images.items():
#             # Stack the bands in the order u, g, r, i, z
#             band_order = ['u', 'g', 'r', 'i', 'z']
#             image_stack = np.stack([resize_image(bands[band]) for band in band_order], axis=-1)  # Shape: (224, 224, 5)
#             images.append([galaxy_id, image_stack])
        
#         return images
    
def create_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 5)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Adjust output layer as needed
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_on_batches(path):
    model = create_model()
    
    for filename in os.listdir(path):
        if filename.endswith('.h5'):
            print(f'Loading {filename}...')
            images = load_from_hdf5(os.path.join(path, filename))
            images = filter_incomplete_bands(images)
            
            # Resize images to 224x224 and stack grayscale into RGB
            images = resize_batch(images)
            images = stack_bands_by_specobjid(images)
            
            # Prepare your data for training
            x = np.array([img[1] for img in images])  # Get images
            y = np.array([get_redshift(img[0], galaxy_df) for img in images])  # Look up redshift by specobjid
            
            # Ensure your labels are in the correct shape
            y = y.reshape(-1, 1)  # Adjust this based on your labels
            
            print(f'Training on {len(images)} images from {filename}...')
            model.fit(x, y, epochs=5, batch_size=32)  # Adjust epochs and batch_size as needed
    # Save the trained model
    model.save('galaxy_model.h5')
path = 'D:/galactic_images_ugriz_test/'
# hdf5_filename = os.path.join(path, 'ugriz_images_batch_1.h5')
# batch=load_from_hdf5(hdf5_filename)
# batch=filter_incomplete_bands(batch)

def evaluate_on_batch(filename):
    # Load the saved model
    model = keras.models.load_model('galaxy_model.h5')
    
    # Load the batch you want to test on
    images = load_from_hdf5(filename)
    images = filter_incomplete_bands(images)
    
    # Resize and stack bands
    images = resize_batch(images)
    images = stack_bands_by_specobjid(images)
    
    # Prepare the data for evaluation
    x_test = np.array([img[1] for img in images])  # Get images
    y_test = np.array([get_redshift(img[0], galaxy_df) for img in images])  # Look up redshift by specobjid
    
    # Ensure your labels are in the correct shape
    y_test = y_test.reshape(-1, 1)
    
    # Evaluate the model on this batch
    loss, accuracy = model.evaluate(x_test, y_test)
    
    print(f'Loss: {loss}, Accuracy: {accuracy}')

def main():
    train_on_batches('D:/galactic_images_ugriz_test/')
    evaluate_on_batch('D:/galactic_images_ugriz_test/ugriz_images_batch_302.h5')

if __name__ == '__main__':
    main()