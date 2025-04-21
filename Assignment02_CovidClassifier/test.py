import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# -------- FocalLoss 정의 --------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, num_classes=3, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes
        self.reduction = reduction

        if alpha is None:
            self.alpha = torch.ones(num_classes)
        else:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha)
            else:
                self.alpha = torch.tensor([alpha] * num_classes)

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        alpha = self.alpha.to(inputs.device)[targets]
        focal_loss = alpha * (1 - pt) ** self.gamma * CE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# -------- 간단한 CNN 모델 정의 --------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112x112
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56x56
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# -------- 데이터 불러오기 함수 --------
def get_transform(img_size=(224, 224)):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

def get_datasets(data_dir, transform):
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)
    return train_dataset, val_dataset, test_dataset

# -------- 학습 함수 --------
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}")

        evaluate_model(model, val_loader, device, mode="Validation")

# -------- 평가 함수 --------
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(model, loader, device, mode="Test"):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

    print(f"\n{mode} Classification Report:\n")
    print(classification_report(all_targets, all_preds, digits=4))
    print(f"Confusion Matrix:\n{confusion_matrix(all_targets, all_preds)}\n")

# -------- 실행 부분 --------
if __name__ == "__main__":
    data_path = "/home/dibaeck/sketch/study_Data4Quality/task02_CovidClassifier/COVID19_1K"  # 여기를 실제 경로로 변경하세요
    batch_size = 32
    num_epochs = 10
    num_classes = 3

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    transform = get_transform()

    train_dataset, val_dataset, test_dataset = get_datasets(data_path, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleCNN(num_classes=num_classes)
    criterion = FocalLoss(gamma=2, alpha=[1, 1, 2], num_classes=num_classes)  # COVID class에 가중치 2 부여
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=num_epochs)

    print("Final Test Performance:")
    evaluate_model(model, test_loader, device, mode="Test")
