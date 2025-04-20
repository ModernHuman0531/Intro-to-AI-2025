import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
from loguru import logger
from sklearn.metrics import accuracy_score

from decision_tree import ConvNet, DecisionTree, get_features_and_labels, get_features_and_paths
from CNN import CNN, train, validate, test
from utils import TrainDataset, TestDataset, load_train_dataset, load_test_dataset, plot

import time # Help us to calculate the time of the training process

"""
Notice:
    1) You can't add any additional package
    2) You can ignore the suggested data type if you want
"""

def main():
    """
    load data
    """
    logger.info("Start loading data")
    images, labels = load_train_dataset()
    images, labels = shuffle(images, labels, random_state=777)
    train_len = int(0.8 * len(images))

    train_images, val_images = images[:train_len], images[train_len:]
    train_labels, val_labels = labels[:train_len], labels[train_len:]
    test_images = load_test_dataset()

    train_dataset = TrainDataset(train_images, train_labels)
    val_dataset = TrainDataset(val_images, val_labels)
    test_dataset = TestDataset(test_images)
    
    """
    CNN - train and validate
    """
    logger.info("Start training CNN")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()

    # Optimizer configuration
    base_params = [param for name, param in model.named_parameters() if param.requires_grad]
    optimizer = optim.Adam(base_params, lr=1e-4)

    train_losses = []
    val_losses = []
    val_accuracies = []
    max_acc = 0
    start_time = time.time()

    EPOCHS = 10
    for epoch in range(EPOCHS): #epoch
        """ Design the training log, to help us monitor and keep track of the training process
        1. Print each epoch's time and training loss and validation loss and validation accuracy
        2. Keep track of the best accuracy and sace the model if the accuracy is better than the previous best accuracy
        
        """
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Append the result into each list
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # (TODO) Print the training log to help you monitor the training process
        #        You can save the model for future usage
        
        # If the accuracy is better than the previous best accuracy, save the model
        if val_acc > max_acc:
            max_acc = val_acc
            # Use torch.save() tp save the model, torch.save(obj, f), other parameter just use default value
            # obj in here means the model, and f means the path to save the model, better way is to store dict in obj
            #, we can save the model and other information in the dict
            torch.save({
                'epoch': epoch, # Save times of training
                'model_state_dict': model.state_dict(), # Save all the parameters of ther model, most important part
                'val_accuracy': val_acc # Save the accuracy of the model
           }, 'best_model.pth')
        
        epoch_time = time.time() - start_time
        # Learning rate means how many change we wamt to make to the model's parameter for every epoch
        # Higher learning rate means we wan to make more change to the model's parameter for every epoch
        lr = optimizer.param_group[0]['lr']

        # Print the training log use logger.info()
        # Add f is to let us get variable value in the string, and we can use {} to get the variable value     
        logger.info = {
            f"Epoch: {epoch+1}/{EPOCHS} | "
            f"Time: {epoch_time}s | "
            f" Learning Rate: {lr:.2e} | " # use {:.2e} to print the learning rate in scientific notation
            f"Train Loss: {train_loss:4f} | " #use {:.4f} to print the loss in 4 decimal places
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracies:.2%} | " # use{.2%} is to print tyhe accuracy in percentage in 2 decimal places
            f"Best Acc: {max_acc:.2%}"
        } 


    logger.info("Traning completed:")
    logger.info(f"Best Accuracy: {max_acc:.4f}")

    """
    CNN - plot
    """
    plot()

    """
    CNN - test
    """
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    test()

    """
    Decision Tree - grow tree and validate
    """
    logger.info("Start training Decision Tree")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    conv_model = ConvNet().to(device)
    tree_model = DecisionTree(max_depth=7)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    train_features, train_labels = get_features_and_labels(conv_model, train_loader, device)
    tree_model.fit(train_features, train_labels)

    val_features, val_labels = get_features_and_labels(conv_model, val_loader, device)
    val_predictions = tree_model.predict(val_features)

    val_accuracy = accuracy_score(val_labels, val_predictions)
    logger.info(f"Validation Accuracy: {val_accuracy:.4f}")

    """
    Decision Tree - test
    """
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    test_features, test_paths = get_features_and_paths(conv_model, test_loader, device)

    test_predictions = tree_model.predict(test_features)

    results = []
    for image_name, prediction in zip(test_paths, test_predictions.cpu().numpy()):
        results.append({'id': image_name, 'prediction': prediction})
    df = pd.DataFrame(results)
    df.to_csv('DecisionTree.csv', index=False)
    print(f"Predictions saved to 'DecisionTree.csv'")


if __name__ == '__main__':
    main()
