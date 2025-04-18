# 활용할 함수들 모아둔 공관
import yaml
import torch
from sklearn.model_selection import train_test_split

def load_yqml(yaml_f):
    """
    config 파일을 wandb가 읽을 수 있게 변환
    """
    
    with open("./config/"+yaml_f+".yaml") as file:
        sweep_config = yaml.safe_load(file)
    return sweep_config


def load_dataset(dataset):
    """
    학습/확인/테스트 데이터셋으로 리턴하기
    """
    train_X, test_X = train_test_split(dataset, test_size=0.2, random_state=42)     # 학습 데이터와  테스트 데이터로 나눔.
    train_X, val_X = train_test_split(train_X, test_size=0.2, random_state=42)       # 학습 데이터를 학습과 검증 데이터로 다시 나눔

    train_loader = torch.utils.data.DataLoader(train_X, batch_size=64, shuffle=True)    # shuffle로 데이터 순서 섞음
    val_loader = torch.utils.data.DataLoader(val_X, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_X, batch_size=64)  
        
    return train_loader, val_loader, test_loader
