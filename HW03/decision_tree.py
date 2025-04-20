import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from typing import List, Tuple

"""
Notice:
    1) You can't add any additional package
    2) You can add or remove any function "except" fit, _build_tree, predict
    3) You can ignore the suggested data type if you want
"""

class ConvNet(nn.Module): # Don't change this part!
    def __init__(self):
        super(ConvNet, self).__init__()
        self.model = timm.create_model('mobilenetv3_small_100', pretrained=True, num_classes=300)

    def forward(self, x):
        x = self.model(x)
        return x
    
class DecisionTree:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.data_size = X.shape[0]
        total_steps = 2 ** self.max_depth
        self.progress = tqdm(total=total_steps, desc="Growing tree", position=0, leave=True)
        self.tree = self._build_tree(X, y)
        self.progress.close()

    def _build_tree(self, X: pd.DataFrame, y: np.ndarray, depth: int):
        # (TODO) Grow the decision tree and return it
        """Implementation of the _build_tree function:
        1. Every node has to have feature_index, thershold, left, right, and class
        2. In the recursive, stop condition is:
            a. Reach max_depth
            b. All the data in the node are the same class
            c. No more feature to split
        3. Loop throuh all the features and calculate the best split use function _best_split
        4. Split the data into left and right node use function _split_data(Split the data basd on the best feature and threshold) 
        5. When feature's value<thershold, go to left node, else go to right node
        6. Keep recursively calling _build_tree to grow left and right tree until meet the stop condition
        7. Return the tree node that include feature_index, thershold, left, right and class(only for leaf node to get predicted class
        """

    def predict(self, X: pd.DataFrame)->np.ndarray:
        # (TODO) Call _predict_tree to traverse the decision tree to return the classes of the testing dataset
        raise NotImplementedError

    def _predict_tree(self, x, tree_node):
        # (TODO) Recursive function to traverse the decision tree
        raise NotImplementedError

    def _split_data(X: pd.DataFrame, y: np.ndarray, feature_index: int, threshold: float):
        # (TODO) split one node into left and right node 
        """Implementation of the _split_data function:
        1. Use thershold as the split point to split tge feature_index into left and right node
        2. If the feature's calue<thershold, go to left node, else go to tight node
        """
        # pd.DataFrame is like dict, and we can use [] to get the value of the key
        """Example of extract feature_index from X:
        X=np.array([[1,2,3],[4,5,6],[7,8,9]])
        X[:,1], :means choose all the rows, and 1 means choose the second column
        X[:,1] = [2,5,8]
        """
        # Use X[:feature_index]] to get the feature index column
        feature_values = X[:, feature_index]

        # Make a bool to check if the feature's value is less than the threshold
        left_mask = feature_values < threshold # Implementation of boolean mask, left_mask is a list of bool value
        """Example of boolean mask
        X = [
        [1.2, 3.4, 5.6],  # 第0行 → 保留 (mask=True)
        [2.1, 0.5, 7.8],  # 第1行 → 保留 (mask=True)
        [5.6, 2.3, 1.2]   # 第2行 → 捨棄 (mask=False)
        ]
        feature_values=X[:,0] # X[:,0] is the first column, and feature_values is [1.2, 2.1, 5.6]
        thershold=2.5
        left_mask=feature_values<threshold # left_mask is [True, True, False]
        X[left_mask] only remain the line when left_mask is True, so the result is [[1.2, 3.4, 5.6], [2.1, 0.5, 7.8]]
        """
        # Use the boolean mask to split the data into left and right node
        left_dataset_X, right_dataset_X = X[left_mask], X[~left_mask]
        left_dataset_y, right_dataset_y = y[left_mask], y[~left_mask]
        return left_dataset_X, left_dataset_y, right_dataset_X, right_dataset_y

    def _best_split(self, X: pd.DataFrame, y: np.ndarray):
        # (TODO) Use Information Gain to find the best split for a dataset
        """
        X is the feature of the dataset, y is the label of the dataset
        Implementation of the _best_split function:
        1. Loop through all the features and calculate the best split
        2. Best split means to maximize the information gain, gain=entropy(parent) - (weighted average of the entropy of the children)
        """
        best_gain = -1
        best_feature_index, best_treshold = None, None
        parent_entropy = self._entropy(y)

        n_features = X.shape[1] # Get the number of features in the dataset

        # Take all the only value of the feature, and take it as candidate threshold
        for feature_index in range(n_features):
            # Get the unique value of the feature
            thresholds = np.unique(X.iloc[:, feature_index])
            for threshold in thresholds:
                # Split the data to left node and right node
                left_x, left_y, right_x, right_y = self._split_data(X, y, feature_index, threshold)
                # Check if the left and right node is empty, if empty, continue the next feature 
                # Check y instead of X, because we want to check the label of the dataset
                if len(left_y)==0 or len(right_y)==0:
                    continue
                # Calculate the left and right node's entropy
                left_entropy = self._entropy(left_y)
                right_entropy = self._entropy(right_y)

                # Calculate the weighted average of the entropy of the children
                # len(left_y) and len(right_y) represent the number of samples in left and right node
                total_len = len(y)
                weighted_entropy = (len(left_y)/total_len)*left_entropy + (len(right_y)/total_len)*right_entropy
                # Calculate the information gain
                gain = parent_entropy-weighted_entropy

                # Check if the gain is better than the best gain, if so update the best gain and best feature index and best threshold
                if gain > best_gain:
                    best_gain, bes_feature_index, best_threshold = gain, feature_index, threshold

        return best_feature_index, best_threshold

    def _entropy(y: np.ndarray)->float:
        # (TODO) Return the entropy
        """Implementation of entropy function:
        1. np.ndarray is an array, and to calculate the entropy we need to use the formula: Entropy(S) = -Σ (p_i * log₂(p_i))
        2. Based on the formula, count every class's number in the dataset, and calculate the probability of each class
        3. Use the probability to calculate the entropy of the dataset
        4. Return the entropy dataset
        """
        # Use vals, counts=np.unique(y, return_count=true) to count the number of each class in data set 
        """Example of using np.unique(y, return_counts=True):
        y=[1,2,2,3,1,4]
        vals, counts = np.unique(y, return_counts=True)
        vals=[1,2,3,4] #vals is all the non-repeated values in y
        counts=[2,2,1,1]# count is the number of each y
        """
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_count=True)
        # Calculate the probability of each class, and use np.log2 to calculate the logarithm of the probability
        probs = counts/len(y)
        entropy = -np.sum(probs*np.log2(probs+1e-10)) # Add small value to prevent log(0)

        return entropy
"""Implementation of get_features_and_labels():Use CNN to extract the features from dataloaders 
1. Set the model to evaluation mode, and disable the gradient calculation
2. Loop over the dataloader, and get the featires and labels for images
3. Turn the torch tensor to numpy array, and return the features and labels
"""
def get_features_and_labels(model: ConvNet, dataloader: DataLoader, device)->Tuple[List, List]:
    # (TODO) Use the model to extract features from the dataloader, return the features and labels
    features, labels = [], []

    model.eval()
    with torch.no_grad():
        for images, label in dataloader:
            # Take the data to GPU. label is just number don;t need to transfer it to GPU
            images = images.to(device)
            # Foward pass the model, and get the output of the model
            outputs = model(images)# Output type is tensor, [B, 5], B means batch size, 5 means the number of classes

            # Append the result roeach list
            # Use .cpu() to move back to cpu, and numpy to turn the tensor to numpy array
            features.append(outputs.cpu().numpy())
            labels.append(label.numpy())
    # Know the features is like [(B, 5), (B, 5), (B, 5)], so we need to use np.concatenate to combine them together
    # Batch means every training, we have to sent B images to model, and epoch means we run through all the images

    # We have to stack all the features together, the system will automatically recognize each images's feature

    # Use if else is to prevent the error when the features is empty
    # np.vstack is to deal with 2D array's stacking, and np.concatenate is to deal with 1D array's stacking
    features = np.vstack(features) if features else np.array([])
    labels = np.concatenate(label) if label else np.array([]) 
    return features, labels
"""Implementation of get_features_and_paths(): Use CNN to extracct features from dataloaders
1. Set the model to evaluation mode, and disable the gradient calculation
2. Loop over the dataloader, and get the features and path of images
3. Turn the torch tensor value into numpy array, and return the features and path
"""
def get_features_and_paths(model: ConvNet, dataloader: DataLoader, device)->Tuple[List, List]:
    # (TODO) Use the model to extract features from the dataloader, return the features and path of the images
    features, paths = [], []

    model.eval()
    # Make sure model is on device
    model.to(device)

    with torch.no_grad():
        # path is the string from TestDataset's base name 
        for images, path in dataloader:
            # Take the images to GPU
            images = images.to(device)
            # Use forward pass the model, an嵌套結構d get the result of the model
            outputs = model(images)
            # Append the result to each list
            features.append(outputs.cpu().numpy())

            # Reason why we use extend instead of append is to prevent nested structure
            """For example:
            batch_paths=['images1.jpg', 'images2.jpg', 'images3.jpg']
            path=[]
            path.append(batch_paths) -> path=[['images1.jpg', 'images2.jpg', 'images3.jpg']]
            path.extend(batch_paths) -> path=['images1.jpg', 'images2.jpg', 'images3.jpg'
            """
            paths.extend(path) 
    # features is like[(B, 5), (B,5), (B,5)], so we stack them together to satrisfy the system's requirement
    # Use np.vstack to stack the 2D array.
    features = np.vstack(features) if features else np.array([])
    return features, paths