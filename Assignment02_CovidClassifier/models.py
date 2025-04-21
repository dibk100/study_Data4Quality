import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix
import torch
import torch.nn as nn
import torchvision.models as models

class CustomDenseNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, device=None):
        super(CustomDenseNet, self).__init__()
        
        # 사전학습된 DenseNet121 불러오기
        self.densenet = models.densenet121(pretrained=pretrained)
        
        # 마지막 분류 레이어 수정
        num_ftrs = self.densenet.classifier.in_features
        if num_classes == 2:
            # 이진 분류: 출력은 1개 (Sigmoid 사용)
            self.densenet.classifier = nn.Sequential(
                nn.Linear(num_ftrs, num_classes),  # 출력 1개
                nn.Sigmoid()  # 이진 분류의 경우 Sigmoid
            )
        else:
            # 다중 클래스 분류: 출력은 num_classes 개 (Softmax 사용)
            self.densenet.classifier = nn.Sequential(
                nn.Linear(num_ftrs, num_classes),  # 출력은 num_classes 개
                nn.Softmax(dim=1)  # 다중 분류의 경우 Softmax
            )
        
        # 장치 설정 (GPU 또는 CPU)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.densenet = self.densenet.to(self.device)

    def forward(self, x):
        return self.densenet(x)


    def model_train(self, train_loader, criterion, optimizer, num_epochs=10):
        self.to(self.device)
        self.train()
        torch.cuda.empty_cache() 
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct_preds = 0
            total_preds = 0

            for inputs, labels in train_loader:
                print(inputs.shape)                                         #### 32, 3 , 224, 224
                inputs = inputs.to(self.device).float()
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = self(inputs)
                 
                if self.num_classes == 2:       # 이진분류
                    labels = labels.float()
                    loss = criterion(outputs.squeeze(), labels)
                    probs = torch.sigmoid(outputs.squeeze())
                    preds = (probs >= 0.5).long()
                else:
                    labels = labels.long()          # 다중
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

                if self.num_classes == 2:
                    labels = labels.float()  # 이진 분류에서는 라벨을 float으로 변환
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

        if self.num_classes == 2:
            auc = roc_auc_score(all_labels, all_probs)
            print(f"AUC-ROC: {auc:.4f}")

        cm = confusion_matrix(all_labels, all_preds)
        print(f"Confusion Matrix:\n{cm}")
