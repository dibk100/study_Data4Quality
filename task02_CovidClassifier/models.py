import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix
import torch
import torch.nn as nn

class BasicNN(nn.Module):
    def __init__(self, num_classes):
        super(BasicNN, self).__init__()
        self.num_classes = num_classes              # 1 : 이진 분류 (sigmoid + BCEWithLogitsLoss) / 2이상 : 다중 클래스 분류 (softmax + CrossEntropyLoss)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # GPU 1번 선택

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.pool = nn.AdaptiveAvgPool2d((8, 8))

        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 1 if num_classes == 1 else num_classes)

        self.to(self.device)
        

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def model_train(self, train_loader, criterion, optimizer, num_epochs=10):
        self.to(self.device)
        self.train()
        torch.cuda.empty_cache()

        for epoch in range(num_epochs):
            running_loss = 0.0
            correct_preds = 0
            total_preds = 0

            for inputs, labels in train_loader:
                print(inputs.shape)
                inputs = inputs.to(self.device).float()
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = self(inputs)

                if self.num_classes == 1:
                    labels = labels.float()
                    loss = criterion(outputs.squeeze(), labels)
                    probs = torch.sigmoid(outputs.squeeze())
                    preds = (probs >= 0.5).long()
                else:
                    labels = labels.long()
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                correct_preds += (preds == labels).sum().item()
                total_preds += labels.size(0)

            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = correct_preds / total_preds
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    def model_eval(self, test_loader):
        self.to(self.device)
        self.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device).float()
                labels = labels.to(self.device)
                outputs = self(inputs)

                if self.num_classes == 1:
                    probs = torch.sigmoid(outputs.squeeze())
                    preds = (probs >= 0.5).long()
                    all_probs.extend(probs.cpu().numpy())
                else:
                    probs = torch.softmax(outputs, dim=1)
                    _, preds = torch.max(probs, 1)
                    all_probs.extend(probs.cpu().numpy())

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=1)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=1)
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

        if self.num_classes == 1:
            auc = roc_auc_score(all_labels, all_probs)
            print(f"AUC-ROC: {auc:.4f}")

        cm = confusion_matrix(all_labels, all_preds)
        print(f"Confusion Matrix:\n{cm}")
