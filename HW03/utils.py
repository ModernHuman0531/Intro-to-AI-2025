from torchvision import transforms
from torch.utils.data import Dataset
import os
import PIL
from typing import List, Tuple
import matplotlib.pyplot as plt

class TrainDataset(Dataset):
    def __init__(self, images, labels):# images: path of images, labels: class of images
        self.transform = transforms.Compose([# transform the image certain size
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.images, self.labels = images, labels

    def __len__(self):# return the number of images
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]# Get specific image path
        image = PIL.Image.open(image_path)# Open the image

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

class TestDataset(Dataset):
    def __init__(self, image):
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.image = image

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image_path = self.image[idx]
        image = PIL.Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        return image, base_name
"""Implement load_train_dataset process: Load training dataset from given path, and give every image their corresponding label
1. Load the image from the path "data/train/", the subfolders have already been classified
2. Create correspomding dictionary for the labels
3. Scanning all the subfolders from "data/train/", for every subfolder
    a. Get the label of the subfolder(like elephant)
    b. Use dictionary to turn the label into number,0->elephant, 1->jaguar, 2->lion, 3->parrot, 4->penguin
    c. Run over the images in the subfolder, and get the path of every image
"""    
def load_train_dataset(path: str='data/train/')->Tuple[List, List]:
    # (TODO) Load training dataset from the given path, return images and labels
    images = []# All the training images corresponding path
    labels = []# Labels to the number, 
    # Because we know the name and we want to find the corresponding number, so we design to let name be the key
    label_dict = {
        'elephant': 0,
        'jaguar': 1,
        'lion': 2,
        'parrot': 3,
        'penguin': 4
    }
    # Use os.walk() to scan all the subfolders, os.walk() can handle more than 1 layer
    for root, _, files in os.walk(path):
        """
        root: Current folder's path
        dirs: sub folders list
        files: All files in current folder
        """
        # Ignore the ./daata/train/, we want it's subfolder,there have images
        if root == path:
            continue
        #Extract the last name of folder
        label_name = os.path.basename(root)
        #Loop through all the files in this folder
        for file in files:
            # Filter the file to make sure only read .jpg and .png file
            if file.endswith(('.jpg', '.png')):
                # Put path and corresponding label in each list
                labels.append(label_dict[label_name])
                # Use os.path.join function to paste path and file's name together
                images.append(os.path.join(root, file))        
    return images, labels
"""Implementation of load_data_set
1. Run all the images in the given folder
2. Store the path of each image to images and return it
"""
def load_test_dataset(path: str='data/test/')->List:
    # (TODO) Load testing dataset from the given path, return images
    images = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endwith(('.jpg', 'png')):
                images.append(os.path.join(root, file))
    return images

"""Implementation of plot function: 
1. Plot the training loss vs epoch and validation loss vs epoch
2. Set the x-axis label to 'Epoch' and y-axis label to 'Loss'
3. Use blue line for training loss abd red line for validation loss
4. Save the plot to 'loss.png'
"""
def plot(train_losses: List, val_losses: List):
    # (TODO) Plot the training loss and validation loss of CNN, and save the plot to 'loss.png'
    #        xlabel: 'Epoch', ylabel: 'Loss'

    plt.figure(figuresize=(10, 6), dpi=300)
    # Plot the training loss and validation loss
    plt.plot(train_losses, label="Training loss", color="blue", linewidth=2)
    plt.plot(val_losses, label="Validation loss", color="red",linewidth=2)

    # Add title and labels
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training and Validation Loss", fontsize=14)
    # Add grid and legend
    plot.grid(True)
    plot.legend(fontsize=12)

    plt.tight_layout()

    # Save the plot to 'loss.png'
    plt.savefig("loss.png")
    plt.close()  # Close the plot to free memory
    print("Save the plot to 'loss.png'")
    return