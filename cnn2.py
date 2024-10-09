import h5py
import os
import torch.nn as nn
from torchvision import transforms

def load_batch(filename):
    with h5py.File(filename, 'r') as h5f:
        images = h5f['data'][()]  # Adjust this key if needed
    return images


# Optional: Preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        # Define layers

    def forward(self, x):
        # Define forward pass
        return x


def train_on_batch(images):
    model = MyCNN()
    model.train()
    
    for img in images:
        img = transform(img)  # Preprocess image
        # Forward pass, loss calculation, backward pass, optimizer step
        # ...


def train_on_batches(path):
    for filename in os.listdir(path):
        if filename.endswith('.h5'):
            images = load_batch(os.path.join(path, filename))
            train_on_batch(images)

def main():
    train_on_batches('D:/galactic_images_ugriz_test/')

if __name__ == '__main__':
    main()