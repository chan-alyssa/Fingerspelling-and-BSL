#oversampling

import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import cv2
import os

class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)


def calculate_f1_score(TP, FP, FN):
    """Calculate F1-score from TP, FP, FN."""
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1


def plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies, train_f1_scores, test_f1_scores,
                 train_class_0_accuracies, train_class_1_accuracies, test_class_0_accuracies, test_class_1_accuracies,
                 train_TP, train_TN, train_FP, train_FN, test_TP, test_TN, test_FP, test_FN):
    # Loss Plot
    plt.figure(figsize=(12, 8))
    plt.plot(train_losses, label='Train Loss', linestyle='-', linewidth=2)
    plt.plot(test_losses, label='Test Loss', linestyle=':', linewidth=2)
    plt.title('Loss Curve', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.savefig('3loss_curve.png', bbox_inches='tight', dpi=300)

    # Class-wise Accuracy Plot
    plt.figure(figsize=(12, 8))
    plt.plot(train_class_0_accuracies, label='Train Class 0 Accuracy', linestyle='-', linewidth=2)
    plt.plot(train_class_1_accuracies, label='Train Class 1 Accuracy', linestyle='--', linewidth=2)
    plt.plot(test_class_0_accuracies, label='Test Class 0 Accuracy', linestyle=':', linewidth=2)
    plt.plot(test_class_1_accuracies, label='Test Class 1 Accuracy', linestyle='-.', linewidth=2)
    plt.title('Class-wise Accuracy Curve', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(fontsize=12)
    plt.savefig('3class_wise_accuracy_curve.png', bbox_inches='tight', dpi=300)

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
    plt.savefig('3train_tp_tn_fp_fn_curve.png', bbox_inches='tight', dpi=300)

    plt.figure(figsize=(12, 8))
    plt.plot(test_TP, label='Test TP', linestyle='-', linewidth=2)
    plt.plot(test_TN, label='Test TN', linestyle='--', linewidth=2)
    plt.plot(test_FP, label='Test FP', linestyle=':', linewidth=2)
    plt.plot(test_FN, label='Test FN', linestyle='-.', linewidth=2)
    plt.title('Test TP, TN, FP, FN', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.legend(fontsize=12)
    plt.savefig('3test_tp_tn_fp_fn_curve.png', bbox_inches='tight', dpi=300)



def fingerspelling(video_names, predictions):
    # Convert predictions to a NumPy array for easier handling
    predictions = np.array(predictions)
    all_fingerspelling_values = np.zeros_like(predictions)  # Initialize with zeros, same length as predictions
    
    unique_video = np.unique(video_names)

    for video in unique_video:
        # Get video path and capture object
        video = str(int(video))  # Ensure video name is an integer in string format
        video_path = f'/scratch/local/hdd/alyssa/bobsl/bobsl/v1.4/original_data/videos/mp4/{video}.mp4'
        video=int(video)
        cap = cv2.VideoCapture(video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)

        # Indices of the frames belonging to the current video
        video_indices = np.atleast_1d(np.where(video_names == video)[0])
        video_predictions = predictions[video_indices]

        # Number of frames corresponds to the length of video_indices
        num_frames = len(video_indices)

        # Sliding window parameters (window_size is 1 second, step_size is 1 frame)
        window_size = int(frame_rate)  # 1-second window (assuming frame rate is frames per second)
        step_size = 1  # Moving by 1 frame at a time

        # Apply sliding window logic, frame-by-frame
        for start in range(num_frames):
            end = start + window_size

            # Get the window's predictions
            window_predictions = video_predictions[start:end]

            # Calculate the proportion of active frames in the window
            proportion_active = np.sum(window_predictions > 0.5) / window_size
            fingerspelling = 1 if proportion_active > 0.5 else 0

            # Assign the fingerspelling value for the current frame
            all_fingerspelling_values[video_indices[start]] = fingerspelling

    # Return as a torch tensor
    return torch.FloatTensor(all_fingerspelling_values)

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


def train_and_evaluate(model, X_train, y_train, X_test, y_test, video_list, learning_rate, num_epochs, lambda_reg=0.):
    
    # pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()], dtype=torch.float32)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Metrics tracking
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []
    train_f1_scores, test_f1_scores = [], []
    train_class_0_accuracies, train_class_1_accuracies = [], []
    test_class_0_accuracies, test_class_1_accuracies = [], []
    train_TP, train_TN, train_FP, train_FN = [], [], [], []
    test_TP, test_TN, test_FP, test_FN = [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        l2_reg = torch.tensor(0.)
        for param in model.parameters():
            if param.requires_grad and len(param.shape) > 1:
                l2_reg += torch.sum(param**2)
        total_loss = loss + lambda_reg * l2_reg
        total_loss.backward()
        optimizer.step()
        train_losses.append(total_loss.item())

        # Training metrics
        train_preds = (torch.sigmoid(outputs) > 0.5).float()
        TP = ((train_preds == 1) & (y_train == 1)).sum().item()
        FN = ((train_preds == 0) & (y_train == 1)).sum().item()
        FP = ((train_preds == 1) & (y_train == 0)).sum().item()
        TN = ((train_preds == 0) & (y_train == 0)).sum().item()
        train_TP.append(TP)
        train_FN.append(FN)
        train_FP.append(FP)
        train_TN.append(TN)
        train_accuracy = (train_preds == y_train).float().mean().item()
        train_accuracies.append(train_accuracy)
        train_f1_scores.append(calculate_f1_score(TP, FP, FN))
        train_class_0_accuracies.append(TN / (y_train==0).sum().item() if (TN + FP) > 0 else 0.0)
        train_class_1_accuracies.append(TP / (y_train==1).sum().item() if (TP + FN) > 0 else 0.0)

        # Testing metrics
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            test_losses.append(test_loss.item())
            test_preds = (torch.sigmoid(test_outputs) > 0.5).float()
            test_preds=np.array(test_preds)
            # print('test_preds')
            # print(test_preds)

            results = fingerspelling(video_list, test_preds)
            TP_test = ((results == 1) & (y_test == 1)).sum().item()
            FN_test = ((results == 0) & (y_test == 1)).sum().item()
            FP_test = ((results == 1) & (y_test == 0)).sum().item()
            TN_test = ((results == 0) & (y_test == 0)).sum().item()
            test_TP.append(TP_test)
            test_FN.append(FN_test)
            test_FP.append(FP_test)
            test_TN.append(TN_test)
            # print('results')
            # print(results)
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
            # print(class1)
            test_accuracies.append(test_accuracy)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Train Loss: {total_loss:.4f}, Test Loss: {test_loss:.4f}, Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}")

    return train_losses, test_losses, train_accuracies, test_accuracies, train_f1_scores, test_f1_scores, train_class_0_accuracies, train_class_1_accuracies, test_class_0_accuracies, test_class_1_accuracies, train_TP, train_TN, train_FP, train_FN, test_TP, test_TN, test_FP, test_FN


def main():
    train_file = "training1new.h5"
    test_file = "testing1new.h5"

    with h5py.File(train_file, 'r') as f:
        X_train = f['video/features'][:]
        y_train = f['video/label'][:]

    with h5py.File(test_file, 'r') as f:
        X_test = f['video/features'][:]
        y_test = f['video/label'][:]
        video_list = f['video/video_name'][:]

    # Perform random oversampling
    X_train, y_train = oversample_random(X_train, y_train)
    y_train = np.expand_dims(y_train, axis=-1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, 'scaler.pkl')
    video_list = np.array(video_list)

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    model = BinaryClassifier(X_train.shape[1])
    results = train_and_evaluate(model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor,video_list,
                                 learning_rate=0.001, num_epochs=500)
    
    torch.save(model.state_dict(), 'binary_classifier.pth')
    plot_metrics(*results)


if __name__ == "__main__":
    main()
