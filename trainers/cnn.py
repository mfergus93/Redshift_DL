import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
import numpy as np
import h5py

def load_from_hdf5(filename):
    with h5py.File(filename, 'r') as h5f:
        images = []
        for dataset_name in h5f:
            galaxy_id, band = dataset_name.split('_')
            images.append([galaxy_id, band, h5f[dataset_name][()]])
        return images

# Load redshift labels from CSV
labels_df = pd.read_csv('galaxy.csv')
redshift_labels = labels_df['redshift'].values  # Assuming the redshift column

# Load images from HDF5 files
all_images = []
# Load images from each HDF5 file (add filenames accordingly)
for filename in list_of_filenames:
    images = load_from_hdf5(filename)
    all_images.extend(img[2] for img in images)  # Extract the image arrays



# Assuming 'images' and 'labels' are your data tensors
dataset = TensorDataset(images, labels)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100)

# Training loop example
for epoch in range(num_epochs):
    for images, labels in train_loader:
        # Perform training steps
        pass