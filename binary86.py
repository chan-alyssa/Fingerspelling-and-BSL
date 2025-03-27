
#uses dictionary which is faster
#also plots loss compared to oversampled
#adding in IOU - before adding into bigger code
#with joints preprocessed

import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import matplotlib.pyplot as plt
import copy
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import cv2
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# class BinaryClassifier(nn.Module):
#     def __init__(self, input_size, hidden_size=64):
#         super(BinaryClassifier, self).__init__()
        
#         # First hidden layer (input to hidden)
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.relu1 = nn.ReLU()
        
#         # Output layer (hidden to output)
#         self.fc2 = nn.Linear(hidden_size, 1)  # Output layer (binary classification)

#     def forward(self, x):
#         # Pass through the first layer
#         x = self.fc1(x)
#         x = self.relu1(x)

#         # Output layer with no activation (we'll use Sigmoid outside the model)
#         x = self.fc2(x)
#         return x
class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout_prob=0.2):
        super(BinaryClassifier, self).__init__()
        
        # First hidden layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)  
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)
        
        # Second hidden layer (newly added)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.layer_norm2 = nn.LayerNorm(hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)

        # Third hidden layer (newly added)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.layer_norm3 = nn.LayerNorm(hidden_size // 4)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_prob)
        
        # Output layer
        self.fc4 = nn.Linear(hidden_size // 4, 1)  

    def forward(self, x):
        x = self.fc1(x)
        x = self.layer_norm1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.layer_norm2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.layer_norm3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.fc4(x)  # Output layer (no activation inside the model)
        return x


def calculate_f1_score(TP, FP, FN):
    """Calculate F1-score from TP, FP, FN."""
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1


def plot_metrics(train_losses, train_losses_oversample, test_losses, train_accuracies, test_accuracies, train_f1_scores, test_f1_scores,
                 train_class_0_accuracies, train_class_1_accuracies, test_class_0_accuracies, test_class_1_accuracies,
                 train_TP, train_TN, train_FP, train_FN, test_TP, test_TN, test_FP, test_FN):
    # Loss Plot
    plt.figure(figsize=(12, 8))
    # plt.plot(train_losses, label='Train Loss', linestyle='-', linewidth=2)
    plt.plot(train_losses_oversample, label='Train Loss Oversampled', linestyle='--', linewidth=2)
    plt.plot(test_losses, label='Test Loss', linestyle=':', linewidth=2)
    plt.title('Loss Curve', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.savefig('84loss_curve.png', bbox_inches='tight', dpi=300)

    # Class-wise Accuracy Plot
    plt.figure(figsize=(12, 8))
    plt.plot(test_accuracies, label='Test Average Accuracy', linestyle='-', linewidth=2)
    plt.plot(train_accuracies, label='Train Average Accuracy', linestyle='--', linewidth=2)
    plt.title('Average Accuracy Curve', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(fontsize=12)
    plt.savefig('84average_accuracy_curve.png', bbox_inches='tight', dpi=300)

    # TP, TN, FP, FN Plots
    plt.figure(figsize=(12, 8))
    plt.plot(train_TP, label='Train TP', linestyle='-', linewidth=2)
    plt.plot(train_TN, label='Train TN', linestyle='--', linewidth=2)
    plt.plot(train_FP, label='Train FP', linestyle=':', linewidth=2)
    plt.plot(train_FN, label='Train FN', linestyle='-.', linewidth=2)
    plt.title('Train TP, TN, FP, FN', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.legend(fontsize=12)
    plt.savefig('84train_tp_tn_fp_fn_curve.png', bbox_inches='tight', dpi=300)

    plt.figure(figsize=(12, 8))
    plt.plot(test_TP, label='Test TP', linestyle='-', linewidth=2)
    plt.plot(test_TN, label='Test TN', linestyle='--', linewidth=2)
    plt.plot(test_FP, label='Test FP', linestyle=':', linewidth=2)
    plt.plot(test_FN, label='Test FN', linestyle='-.', linewidth=2)
    plt.title('Test TP, TN, FP, FN', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.legend(fontsize=12)
    plt.savefig('84test_tp_tn_fp_fn_curve.png', bbox_inches='tight', dpi=300)
    
    plt.figure(figsize=(12, 8))
    plt.plot(train_class_0_accuracies, label='Train Class 0 Accuracy', linestyle='-', linewidth=2)
    plt.plot(train_class_1_accuracies, label='Train Class 1 Accuracy', linestyle='--', linewidth=2)
    plt.plot(test_class_0_accuracies, label='Test Class 0 Accuracy', linestyle=':', linewidth=2)
    plt.plot(test_class_1_accuracies, label='Test Class 1 Accuracy', linestyle='-.', linewidth=2)
    plt.title('Class-wise Accuracy Curve', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(fontsize=12)
    plt.savefig('84class_wise_accuracy_curve.png', bbox_inches='tight', dpi=300)



def create_video_indices_dict(video_names):
    # Using defaultdict to avoid key errors and streamline the process
    from collections import defaultdict
    video_indices_dict = defaultdict(list)
    for idx, video in enumerate(video_names):
        video_indices_dict[int(video.item())].append(idx)
    return video_indices_dict

def fingerspelling(predictions, window_size, threshold):
    step_size=1
    num_frames = len(predictions)

    fingerspelling_values = np.zeros(num_frames)

    for start in range(0, num_frames - window_size + 1, step_size):
        end = start + window_size
        window_predictions = predictions[start:end]

        # Check if the proportion of predicted frames > 0.5 in the window exceeds the threshold
        proportion_active = torch.sum(window_predictions)
        fingerspelling = 1 if proportion_active > threshold else 0

        if fingerspelling == 1:
            fingerspelling_values[start:end] = 1

    return fingerspelling_values


def find_events(binary_array):

    events = []
    in_event = False
    start = None

    for i, val in enumerate(binary_array):
        if val == 1 and not in_event:
            in_event = True
            start = i
        elif val == 0 and in_event:
            in_event = False
            events.append((start, i - 1))
    if in_event:
        events.append((start, len(binary_array) - 1))
    return events

def calculate_iou_and_metrics(predictions, ground_truth, video_list, iou_threshold=0.5):
    tp = 0
    fp = 0
    fn = 0

    ground_event = False  # Track if we are inside a ground truth event
    pred_event = False  # Track if we are inside a predicted event

    intersection = 0
    union = 0
    total_gt_events = 0

    previous_video = None  # Track the previous video to detect changes

    for i in range(len(predictions)):
        pred = predictions[i]
        gt = ground_truth[i]
        video_name = video_list[i]

        # Reset IoU tracking if the video has changed
        if video_name != previous_video:
            intersection = 0
            union = 0
            ground_event = False
            pred_event = False

        previous_video = video_name  # Update previous video tracker

        # Detect start of a new ground truth event
        if gt == 1 and not ground_event:
            ground_event = True
            total_gt_events += 1  # Count only when an event starts

        # Detect start of a new predicted event
        if pred == 1 and not pred_event:
            pred_event = True

        # Track intersection and union
        if ground_event or pred_event:
            union += 1
        if ground_event and pred_event:
            intersection += 1

        if pred == 0 and pred_event and not ground_event:
            if intersection==0:
                fp += 1 
                pred_event = False
                intersection = 0
                union = 0

        # Check if both events ended (a complete event window)
        if (pred == 0 and gt == 0) and (pred_event or ground_event):
            iou = intersection / union if union > 0 else 0
            if iou >= iou_threshold:
                tp += 1  # True Positive (correctly matched event)
            pred_event = False
            ground_event = False
            intersection = 0
            union = 0

        if gt == 0 and ground_event:
            ground_event = False  # Reset when the ground truth event ends
        
        if pred == 0 and pred_event:
            pred_event = False  # Reset when the predicted event ends

    # Final FN and FP calculations
    fn = total_gt_events - tp  # Missed ground truth events
 
    return tp, fn, fp



def oversample_random(X, y):
    """Random oversampling of the minority class."""
    # Separate majority and minority classes
    y = np.squeeze(y)
    X_majority = X[y == 0]
    X_minority = X[y == 1]

    # Oversample the minority class
    minority_oversampled = np.random.choice(len(X_minority), size=len(X_majority), replace=True)
    X_minority_oversampled = X_minority[minority_oversampled]

    # Combine the oversampled minority class with the majority class
    X_balanced = np.vstack([X_majority, X_minority_oversampled])
    y_balanced = np.hstack([np.zeros(len(X_majority)), np.ones(len(X_minority_oversampled))])

    # Shuffle the dataset
    indices = np.arange(len(y_balanced))
    np.random.shuffle(indices)
    return X_balanced[indices], y_balanced[indices]


def train_and_evaluate(model, X_train, y_train,  X_train_oversample, y_train_oversample,X_test, y_test, video_list, learning_rate, num_epochs):
    
    # pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()], dtype=torch.float32)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_epoch = -1
    best_accuracy = 0.0
    best_model = None
    best_preds = None  
    # Metrics tracking
    train_losses, train_losses_oversample, test_losses = [], [],[]
    train_accuracies, test_accuracies = [], []
    train_f1_scores, test_f1_scores = [], []
    train_class_0_accuracies, train_class_1_accuracies = [], []
    test_class_0_accuracies, test_class_1_accuracies = [], []
    train_TP, train_TN, train_FP, train_FN = [], [], [], []
    test_TP, test_TN, test_FP, test_FN = [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass on oversampled training data
        outputs_oversample = model(X_train_oversample)
        print(X_train_oversample.shape)
        # print(outputs_oversample.shape)
        # print(y_train_oversample.shape)
        loss_oversample = criterion(outputs_oversample, y_train_oversample)
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        train_losses.append(loss.item())
            
        # Backward pass
        loss_oversample.backward()  # Backpropagate loss from oversampled data
        optimizer.step()  # Update weights

        # Track training losses
        train_losses_oversample.append(loss_oversample.item())

        # Training metrics for original data
        train_preds = (torch.sigmoid(outputs_oversample) > 0.5).float()
        TP = ((train_preds == 1) & (y_train_oversample == 1)).sum().item()
        FN = ((train_preds == 0) & (y_train_oversample == 1)).sum().item()
        FP = ((train_preds == 1) & (y_train_oversample == 0)).sum().item()
        TN = ((train_preds == 0) & (y_train_oversample == 0)).sum().item()
        train_TP.append(TP)
        train_FN.append(FN)
        train_FP.append(FP)
        train_TN.append(TN)

        # train_accuracy = (train_preds == y_train).float().mean().item()
        # train_accuracies.append(train_accuracy)
        train_f1_scores.append(calculate_f1_score(TP, FP, FN))
        trainclass0= TN /((y_train_oversample == 0).sum().item() )
        # print(TN_test)
        # print(TP_test)
        train_class_0_accuracies.append(trainclass0)
        trainclass1= TP / (y_train_oversample == 1).sum().item()
        train_class_1_accuracies.append(trainclass1)
        train_accuracy = (trainclass1+trainclass0)/2
        # print(class0)
        # print(class1)
        train_accuracies.append(train_accuracy)
        
        # Testing metrics
        model.eval()
        with torch.no_grad():
            # Forward pass on original training data
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            test_losses.append(test_loss.item())
            test_preds = (torch.sigmoid(test_outputs) > 0.5).float()
            # test_preds=np.array(test_preds)
            # print('test_preds')
            # print(test_preds)

            # start_time=time.time()
            # results = fingerspelling(video_list, test_preds)
            # # end_time=time.time()
            # # elapsed_time = end_time - start_time

            # # print(f"Elapsed time: {elapsed_time:.2f} seconds")
            # results = results.unsqueeze(1)
            TP_test = ((test_preds == 1) & (y_test == 1)).sum().item()
            FN_test = ((test_preds == 0) & (y_test == 1)).sum().item()
            FP_test = ((test_preds == 1) & (y_test == 0)).sum().item()
            TN_test = ((test_preds == 0) & (y_test == 0)).sum().item()
            # TP_test = ((results == 1) & (y_test == 1)).sum().item()
            # FN_test = ((results == 0) & (y_test == 1)).sum().item()
            # FP_test = ((results == 1) & (y_test == 0)).sum().item()
            # TN_test = ((results == 0) & (y_test == 0)).sum().item()
            test_TP.append(TP_test)
            test_FN.append(FN_test)
            test_FP.append(FP_test)
            test_TN.append(TN_test)

            # test_accuracy = (results == y_test).float().mean().item()
            # test_accuracies.append(test_accuracy)
            test_f1_scores.append(calculate_f1_score(TP_test, FP_test, FN_test))
            class0= TN_test /((y_test == 0).sum().item() )
            # print(TN_test)
            # print(TP_test)
            test_class_0_accuracies.append(class0)
            class1= TP_test / (y_test == 1).sum().item()
            test_class_1_accuracies.append(class1)
            test_accuracy = (class1+class0)/2
            # print(class0)
            test_accuracies.append(test_accuracy)
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_epoch = epoch
                best_model = copy.deepcopy(model)  # Store a copy of the best model
                best_preds = test_preds.detach().clone()
                best_model = model  # Store the model at the best epoch
                best_preds = test_preds.detach().clone()
                torch.save(model.state_dict(), 'binary_classifierf86.pth')

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Train Loss: {loss_oversample:.4f}, Test Loss: {test_loss:.4f}, Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}")
    results = fingerspelling(test_preds, 5,3)
    best_model = model  # Reuse the model architecture
    best_model.load_state_dict(torch.load('binary_classifierf86.pth'))  

    tp ,fn, fp = calculate_iou_and_metrics(results, y_test,video_list)
    print(tp,fn,fp)
    print(TP_test,FN_test,FP_test)

    # train_best_accuracy = 0
    # for threshold in np.arange(0.5, 1.01, 0.01):  # Iterate over thresholds
    #     train_preds = (torch.sigmoid(outputs_oversample) > threshold).float()
    #     TP = ((train_preds == 1) & (y_train_oversample == 1)).sum().item()
    #     FN = ((train_preds == 0) & (y_train_oversample == 1)).sum().item()
    #     FP = ((train_preds == 1) & (y_train_oversample == 0)).sum().item()
    #     TN = ((train_preds == 0) & (y_train_oversample == 0)).sum().item()

    #     # Accuracy calculation
    #     train_accuracy = (TP + TN) / (TP + TN + FP + FN)
    #     if train_accuracy > train_best_accuracy:
    #         train_best_accuracy = train_accuracy
    #         best_threshold_train = threshold  # Save the best threshold

    # print(f"Best Training Threshold: {best_threshold_train:.2f}, Accuracy: {train_best_accuracy:.4f}")

    return train_losses, train_losses_oversample, test_losses, train_accuracies, test_accuracies, train_f1_scores, test_f1_scores, train_class_0_accuracies, train_class_1_accuracies, test_class_0_accuracies, test_class_1_accuracies, train_TP, train_TN, train_FP, train_FN, test_TP, test_TN, test_FP, test_FN


class BOBSLDataset(Dataset):
    def __init__(self, file_path, transform=None, target_transform=None):
        with h5py.File(file_path, 'r') as f:
            self.feature = f['video/features'][:]
            self.label = f['video/label'][:]
            self.frame_number = f['video/frame_number'][:]
            self.video_list = f['video/video_name'][:]


    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        feature = self.feature[idx]
        label = self.label[idx] 
        return feature, label

def main():
    

    train_file = "training1f86.h5"
    test_file = "testing1f86.h5"

    
    training_data = BOBSLDataset(train_file)
    testing_data = BOBSLDataset(test_file)

    # train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    # test_dataloader = DataLoader(testing_data, batch_size=64, shuffle=True)
    
    # for data in train_dataloader:
    #     X_train, y_train = data
    #     # print(y_train.shape)

    # for data in test_dataloader:
    #     X_test, y_test = data
    X_train = training_data.feature  # Convert to NumPy for processing
    y_train = training_data.label
    X_test = testing_data.feature
    y_test = testing_data.label
    video_list=testing_data.video_list

    # Perform random oversampling
    X_train_oversample, y_train_oversample = oversample_random(X_train, y_train)
    y_train_oversample = np.expand_dims(y_train_oversample, axis=-1)
    scaler = StandardScaler()
    X_train_oversample = scaler.fit_transform(X_train_oversample)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, 'scalerf86.pkl')


    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_train_tensor_oversample = torch.tensor(X_train_oversample, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_train_tensor_oversample = torch.tensor(y_train_oversample, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    model = BinaryClassifier(training_data.feature.shape[1]).to(device)
    results = train_and_evaluate(model,X_train_tensor, y_train_tensor, X_train_tensor_oversample, y_train_tensor_oversample, X_test_tensor, y_test_tensor,video_list,
                                 learning_rate=0.001, num_epochs=1000)

    plot_metrics(*results)


if __name__ == "__main__":
    main()
