import torch
import torch.nn as nn
import torch.optim as optim
from study_Data4Quality.Assignment02_CovidClassifier.models_sketch import CustomDenseNet
from utils import get_transform, get_customdatasets, get_dataloaders
from train_eval import train_one_epoch
from evaluation import evaluate_model

def run_experiment(target_classes, model_name, criterion, optimizer, device, data_dir, img_size, batch_size, epochs):
    print(f"\n🧪 Running experiment: {model_name}")
    
    transform = get_transform(img_size)
    train_dataset, val_dataset, test_dataset = get_customdatasets(data_dir, transform, target_classes)
    train_loader, val_loader, test_loader = get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size)

    num_classes = len(target_classes)
    model = CustomDenseNet(num_classes=num_classes).to(device)

    # 옵티마이저는 외부에서 전달된 걸로 교체
    optimizer.param_groups = []
    optimizer.add_param_group({'params': model.parameters()})

    # 학습 루프
    for epoch in range(epochs):
        print(f"Epoch [{epoch+1}/{epochs}]")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")

    print(f"\n📊 Validation Results ({model_name}):")
    evaluate_model(model, val_loader, device, target_classes)

    print(f"\n🧪 Test Results ({model_name}):")
    evaluate_model(model, test_loader, device, target_classes)


def main():
    # 공통 설정
    data_dir = "./data"
    img_size = (224, 224)
    batch_size = 32
    epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🧪 실험 1: PNEUMONIA vs NORMAL (Binary)
    run_experiment(
        target_classes=["PNEUMONIA", "NORMAL"],
        model_name="Binary_Pneumonia_vs_Normal",
        criterion=nn.BCEWithLogitsLoss(),
        optimizer=optim.Adam([], lr=0.001),  # 빈 param group으로 전달
        device=device, data_dir=data_dir, img_size=img_size,
        batch_size=batch_size, epochs=epochs
    )

    # 🧪 실험 2: COVID19 vs NORMAL (Binary)
    run_experiment(
        target_classes=["COVID19", "NORMAL"],
        model_name="Binary_Covid19_vs_Normal",
        criterion=nn.BCEWithLogitsLoss(),
        optimizer=optim.Adam([], lr=0.001),
        device=device, data_dir=data_dir, img_size=img_size,
        batch_size=batch_size, epochs=epochs
    )

    # 🧪 실험 3: COVID19 vs PNEUMONIA vs NORMAL (Multi-class)
    run_experiment(
        target_classes=["COVID19", "PNEUMONIA", "NORMAL"],
        model_name="MultiClass_All",
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.Adam([], lr=0.001),
        device=device, data_dir=data_dir, img_size=img_size,
        batch_size=batch_size, epochs=epochs
    )

if __name__ == "__main__":
    main()
