######################################## setting 
from dataprocessing import *
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

# 데이터 경로
data_path = '/home/dibaeck/sketch/study_Data4Quality/task02_CovidClassifier/COVID19_1K'

# 데이터 정규화 및 리사이즈, 텐서화
transform = get_transform()

train_dataset, val_dataset, test_dataset = get_datasets(data_path, transform)

# 데이터 로더 batch size 32
train_loader = DataLoader(train_dataset, shuffle=True)
val_loader = DataLoader(val_dataset, shuffle=False)
test_loader = DataLoader(test_dataset, shuffle=False)

from study_Data4Quality.task02_CovidClassifier.models import *

# 모델 불러오기
# Binary Classification (PNEUMONIA vs NORMAL)
binary_model_pneumonia_vs_normal = BasicNN(num_classes=2)

# Binary Classification (COVID19 vs NORMAL)
binary_model_covid19_vs_normal = BasicNN(num_classes=2)

# Multi-class Classification (COVID19 vs PNEUMONIA vs NORMAL)
multi_class_model = BasicNN(num_classes=3)

import torch.optim as optim

# 각각의 모델에 대해 이진 분류에서는 Binary Cross Entropy를, 다중 클래스 분류에서는 Cross Entropy를 사용

# 이진 분류 모델 (PNEUMONIA vs NORMAL, COVID19 vs NORMAL) 
criterion_binary = nn.BCEWithLogitsLoss()  # 이진 분류에서 사용
optimizer_binary = optim.Adam(binary_model_pneumonia_vs_normal.parameters(), lr=0.001)

# 다중 클래스 분류 모델 (COVID19 vs PNEUMONIA vs NORMAL)
criterion_multi_class = nn.CrossEntropyLoss()  # 다중 클래스 분류에서 사용
optimizer_multi_class = optim.Adam(multi_class_model.parameters(), lr=0.001)


# 모델 학습
binary_model_pneumonia_vs_normal.model_train_binary(train_loader, criterion_binary, optimizer_binary)

binary_model_covid19_vs_normal.model_train(train_loader, criterion_binary, optimizer_binary)

multi_class_model.model_train(train_loader,criterion_multi_class,optimizer_multi_class)

뭐가 문제인데ㅜㅠㅜㅠㅜㅠㅜ


# 모델 평가
binary_model_pneumonia_vs_normal.model_eval(test_loader)
binary_model_covid19_vs_normal.model_eval(test_loader)
multi_class_model.model_eval(test_loader)