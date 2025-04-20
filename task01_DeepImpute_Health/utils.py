# 활용할 함수들 모아둔 공관
import yaml
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
    
def load_yqml(yaml_f):
    """
    config 파일을 wandb가 읽을 수 있게 변환
    """
    
    with open("./config/"+yaml_f+".yaml") as file:
        sweep_config = yaml.safe_load(file)
    return sweep_config

def dataset_split(dataset):
    """
    데이터셋 split : binary
    
    """
    X,Y = train_test_split(dataset, test_size=0.2, random_state=42)     # 학습 데이터와  테스트 데이터로 나눔.

    return X,Y

def loader_dataset(train_X,test_X,batch):
    """
    하이퍼파라미터 튜닝 단계 : 
        train_X : train_set
        test_X : val_set
        
    최종 모델 학습 단계 :
        train_X : train_set + val_set
        test_X : test_set
    """
    train_loader = torch.utils.data.DataLoader(train_X, batch_size=batch, shuffle=True)    # shuffle로 데이터 순서 섞음
    test_loader = torch.utils.data.DataLoader(test_X, batch_size=batch) 
    
    return train_loader, test_loader

def data_processing():
    """
    Q1_DataProcessing.ipynb
    참고하기
    
    info :
        scaler : 나중에 다시 스케일링사용해야해서 return으로 반환해야함.
    """
    data = pd.read_csv("./Data_01.csv")
    
    # 데이터 세팅
    drop_cols = ['HE_Ucot', 'HE_FVC', 'HE_Frtn','Target_DM', 'Target_HT']
    features = data.drop(columns=drop_cols)
    
    # 결측 마스크 저장
    mask = features.isnull()
    
    # 임시 결측치 대체 (컬럼 평균으로)
    features_filled = features.fillna(features.mean())

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_filled)
    
    return features_scaled, mask

def replace_missing_values(features_scaled,mask,predicted_values,scaler):
    """
    모델로 예측한 값으로 결측치 복원
    Args:
        features_scaled: (numpy.ndarray)학습할 때 입력했던 데이터
        mask : (dataframe)결측치였던 거 위치 기록한 데이터
        predicted_values: (numpy.ndarray) 모델 결과값
        scaler : 

    Returns:
        df_imputed: 결측치 다 채운 완성본
    """
    
    row_len = features_scaled.shape[0]
    col_len = features_scaled.shape[1]
    output = features_scaled.copy()             # 혹시 모르니까 copy
    
    # 마스크를 이용해 결측값만 대체
    for i in range(row_len):
        for j in range(col_len):
            if mask.iloc[i, j]:  # 결측치가 있는 부분만 대체
                output[i, j] = predicted_values[i, j]

    return output