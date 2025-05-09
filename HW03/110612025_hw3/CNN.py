import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple
import pandas as pd
"""Implementation of design convolution layers
1. Baisc structure of CNN: (Conv2d-> ReLU->MaxPool2d)*N -> Flatten -> Fully Connected Layer -> Softmax
2. Design layer
    layer_num | Channel number changes | Space size changes
    1         | 3 -> 32                | 224 -> 224
    2         | 32 -> 64               | 224 -> 112(maxpooling set kernal = 2 will reduce the size by half)
    3         | 64 -> 128              | 112 -> 56 (maxpooling set kernal = 2 will reduce the size by half)
3. Goes to fully connected layer.
"""
class CNN(nn.Module):
    def __init__(self, num_classes=5):
        # (TODO) Design your CNN, it can only be less than 3 convolution layers
        super(CNN, self).__init__()
        # Use nn.Conv2d to design the convolution layer, nn.conv2d(in_channels, out_channels, kernel_size, stride, padding)
        #First convolution layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, padding = 1) # kernal_size = 3, means 3*3 kernal. padding=1, means input and output kernal size is the same
        # Second convo;ution layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)
        # Third convolution layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size = 3, padding = 1)
        # Use torch.nn.MaxPool2d(kernal_size, stride=None, padding=0, dilation=1) to design the maxpooling layer)
        self.pool = nn.MaxPool2d(kernel_size = 2)
        # Create a global average pool, which count eevery channel's average value, means the intensity of the channel
        # Input: (128, 56, 56)-> Output: (128, 1, 1), take every channel 56*56 matrix, and get the average value of every channel
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        #Create a fully connected layer, combining class score based on the importance of each channel
        # Use torch.nn.Linear(in_features, out_featires, bias=True), channel means in_features, out_features means the number of classes
        self.fc = nn.Linear(128, num_classes)
    def forward(self, x):
        # (TODO) Forward the model
        # Implement the structure of CNN: (Conv2d-> ReLU->MaxPool2d)*N -> Flatten -> Fully Connected Layer -> Softmax
        
        #Utilize F.relu to help CNN learn complicated patterns, and improve traning stability

        # x means the input of the model, which is a batch of images, and the shape is [B, 3, 224, 224], B means batch size, 3 means RGB channel, 224 means image size        
        
        # First layer: [B,3,224,224] -> [B, 32, 224, 224]
        x = F.relu(self.conv1(x))
        x = self.pool(x) # [B, 32, 224, 224]->[B, 32, 112, 112]

        # Second layer:-> [B, 64, 112, 112]
        x = F.relu(self.conv2(x))
        x = self.pool(x)# ->[B, 64, 56, 56]
        
        # Third layer: -> [B, 128, 56, 56]
        x = F.relu(self.conv3(x))
        
        # Global average it:[B, 128, 56, 56] -> [B, 128, 1, 1]
        x = self.global_avg_pool(x)

        # Flatten it becaues Linear isn matrix multiplication and input take[batch_size, feature]
        x = x.view(x.size(0), -1)
        # x.view means reshape the tensor, x.size(0) means batch size, -1 means the rest of the dimension, which is 128*1*1=128

        #Categorize it 128->5
        s = self.fc(x)
        return s
"""Implmentation of train function: To update the model's weight and bias, and return the average loss of the data
1. Set the model to the traning mode, and set the model to the device(GPU)
2. Keep cycling over the training dataset.
3. Clear the gradient of optimizer, or else the gradient will accumulate.
4. Foward pass the model, and get the output of the model.
5. Calculate the loss of the model, and backward pass the model to get the gradient of th emodel
6. Update the model's parameter by optimizer.step()
7. Accumulate the loss of the model, and return the average loss of the data

"""
def train(model: CNN, train_loader: DataLoader, criterion, optimizer, device)->float:
    # (TODO) Train the model and return the average loss of the data, we suggest use tqdm to know the progress
    model.train()

    
    total_loss = 0
    progress_bar = tqdm(train_loader, desc="Training")
    for images, labels in train_loader:
        # Take the data to the GPU
        images, labels = images.to(device), labels.to(device)

        # Clear the gradient of optimizer
        optimizer.zero_grad()

        # Foward pass the model, and get the output of the model
        outputs = model(images)

        # Use criterion(loss function) to calculate the difference between the output and the label
        loss = criterion(outputs, labels)

        # Backward pass the model to get the gradient of the model
        loss.backward()
        # Use optimizer (optimizer.step() function) to update the model's parameter
        optimizer.step()

        # Use loss.item() instead of loss is becauseto break the graph, and get the value of the loss
        # loss.item() means the value of the loss, and it is a float number
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    avg_loss = total_loss / len(train_loader)
    return avg_loss

"""Implementation of validate: Test the model on validation dataset 
1. Set the model to the evaluation mode, and set the model to the device(GPU)
2. Disable the gradient calculation, because we don't need to update the model's parameter
3. Keep cycling over the validation dataset.
4. Calculate the loss of the model, and calculate the accuracy of the model
5. Return the average loss and accuracy of the data
"""
def validate(model: CNN, val_loader: DataLoader, criterion, device)->Tuple[float, float]:
    # (TODO) Validate the model and return the average loss and accuracy of the data, we suggest use tqdm to know the progress
    model.eval()
    total_loss, correct = 0, 0

    # Disable the gradient calculation, can save the meomry and speed up the process
    with torch.no_grad():
        for images, labels in val_loader:
            # Take the data to the GPU
            images, labels = images.to(device), labels.to(device)
            # Foward pass the model, and get the output of the model
            outputs = model(images)
            # Use criterion(loss function) to calculate the difference between the output and label
            loss = criterion(outputs, labels)
            # Use loss.item() to break the graph, and get the value of the loss
            total_loss += loss.item()

            # Calculate the accuracy of the model, the return value of torch.max() is the maximum value and the index of the maximum value
            # torch.max(output, dim), dim means the which direction we want to find maximum value, 0 means the vertical direction, 1 means the horizontal direction 
            _, predicted = torch.max(outputs, 1) # We want to check maximum value in horizontal direction, so dim = 1
            # Calculate the number of correct prediction, and accumulate it
            # .sum is to count the number of true in tensor, and .item() is to transform the tensor to a float number
            # predicted and labels both value are tensor, so we don't need to use .numpy() to transform it to numpy array
            correct += (predicted == labels).sum().item()
    
    # len(val_holder) means the number of epochs in the validation dataset, and len(val_loader.dataset) means the number of samples in the validation dataset
    avg_loss, accuracy = total_loss/len(val_loader), correct/len(val_loader.dataset)
    return avg_loss, accuracy
"""Implementation of test function: Test the model on testing dataset and write the result to 'CNN.csv'
1. Set the model to evaluation mode
2. Disable the gradient calculation, because we don't need to update the model's parameter
3. Create Dataloader for test dataset, because we need to test the model on the test dataset
4. Keep cycling over the test dataset.
5. Foward pass the model, and get the output of the model

"""
def test(model: CNN, test_loader: DataLoader, criterion, device):
    # (TODO) Test the model on testing dataset and write the result to 'CNN.csv'
    model.eval()
    result = []

    with torch.no_grad():
        for images, img_ids in test_loader:
            # Take the data to GPU
            images = images.to(device)
            # Foward pass the model, and get the output of the model
            
            output = model(images)# The type of output is tensor, and the shape is [B, 5], B means batch size, 5 means the number of classes
            
            # Get the predicted class of the model, and get the index of the maximum value
            _, predicted = torch.max(output, 1)
            # Append the predicted class and image id to the result list
            #Use zip is to combine the result together, and the result is a list of tuple, each tuple is (image_id, predicted result)
            for img_id, pred in zip(img_ids, predicted.cpu().numpy()):# .cpu() is to transfer the tensor to CPU, and .numpy() is to transform the tensor to numpy array 
                result.append({"id": img_id, "prediction": int(pred)})
    # Turn the result to a pandas dataframe, and save it to 'CNN.csv'
    df = pd.DataFrame(result) #Turn the result to a pandas table
    df.to_csv('CNN.csv', index=False) # Save the result to 'CNN.csv'  
    print(f"Predictions saved to 'CNN.csv'")
    return df