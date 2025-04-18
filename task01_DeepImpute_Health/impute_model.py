import torch
import torch.nn as nn
import wandb
from utils import *
import time

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
        
    def model_train_fixed_hyperparameter(self,train_loader,optimizer,epochs):
        """
        Autoencoder 모델을 학습 : 파라미터 고정
        
        Args:
            train_loader (DataLoader): 학습 데이터

        Returns:
            None
        """

        ####################### SETTING #######################
        device = 'cuda' if torch.cuda.is_available() else 'cpu'     # gpu 
        if device!='cuda':
            raise RuntimeError("CUDA device not available. Please check if a GPU is available.")

        self.to(device)                                             # 모델을 gpu에 올리기
        self.train()                                                # 학습모드on
        
        # 손실 함수 설정
        criterion = nn.MSELoss()  # 평균제곱오차 고정

        ####################### Train! #######################
        print("####################### Train! #######################")
        start_time = time.time()  # 시작 시간 기록
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device).float()  # 배치 데이터를 장치에 올림
                optimizer.zero_grad()  # 기울기 초기화
                output = self(batch)  # 모델에 배치 데이터 넣고 출력값 얻음
                loss = criterion(output, batch)  # 손실 계산
                loss.backward()  # 역전파
                optimizer.step()  # 가중치 업데이트
                total_loss += loss.item()
            
            if epoch%10 == 0 :
                # 10epoch마다 손실 기록
                print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")
        else :
            print(f"> Final ::: Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}\n")
        
        end_time = time.time()  # 종료 시간 기록
        elapsed_time = end_time - start_time
        print(f"> 총 실행 시간: {elapsed_time:.2f}초")  # 소수점 2자리까지 출력
        
        return
    
    def evaluate_model(self,test_loader,mask,batch_size):
        """
        Autoencoder 모델을 평가
        Args:
            test_loader (DataLoader): 평가 데이터
            mask : 결측치였던 거 위치 기록한 데이터

        Returns:
            None
        """
        
        ####################### SETTING #######################
        device = 'cuda' if torch.cuda.is_available() else 'cpu'     # gpu 
        if device!='cuda':
            raise RuntimeError("CUDA device not available. Please check if a GPU is available.")
        
        self.to(device)
        self.eval()  # 평가 모드
        
        mask = mask.to_numpy()
        ####################### EVALUATE #######################
        total_loss = 0
        criterion = nn.MSELoss()                # 손실함수
        
        with torch.no_grad():
            for idx,batch in enumerate(test_loader):
                batch = batch.to(device).float()  # batch를 float로 변환
                
                # 여기서 배치 인덱스 추적을 위해
                start_idx = idx * batch_size
                end_idx = (idx + 1) * batch_size
                m = mask[start_idx:end_idx]  # 해당 배치에 맞는 마스크 추출
                
                output = self(batch)
                
                # 마스크를 이용하여 결측값을 제외한 부분만 손실을 계산
                loss = criterion(output[m == 1], batch[m == 1])  # mask가 1인 부분만 비교
                total_loss += loss.item()
        
        avg_loss = total_loss / len(test_loader)
        print(f"Average Test Loss: {avg_loss:.4f}")
        
        return    

    
    def model_train_wandb(self,train_loader,val_loader):
        """
        Autoencoder 모델 학습 : 최적의 하이퍼파라미터 찾는 실험
        
        Args:
            train_loader (DataLoader): 학습 데이터
            val_loader (DataLoader): 검증 데이터

        Returns:
            None
        """

        ####################### SETTING #######################
        device = 'cuda' if torch.cuda.is_available() else 'cpu'     # gpu 
        if device!='cuda':
            raise RuntimeError("CUDA device not available. Please check if a GPU is available.")

        self.to(device)                                             # 모델을 gpu에 올리기
        self.train()                                                # 학습모드on
        
        config = wandb.config  ######################## wandb 설정에서 하이퍼파라미터를 가져옴
        
        # 옵티마이저 설정
        if config.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
        elif config.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=config.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {config.optimizer}")
        
        # 손실 함수 설정
        criterion = nn.MSELoss()  # 평균제곱오차 고정

        ####################### step01 : Train! #######################
        for epoch in range(config.epochs):
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device).float()  # 배치 데이터를 장치에 올림
                optimizer.zero_grad()  # 기울기 초기화
                output = self(batch)  # 모델에 배치 데이터 넣고 출력값 얻음
                loss = criterion(output, batch)  # 손실 계산
                loss.backward()  # 역전파
                optimizer.step()  # 가중치 업데이트
                total_loss += loss.item()
            
            # 매 epoch마다 wandb로 손실 기록
            wandb.log({"train_loss": total_loss})
            print(f"Epoch {epoch+1}/{config.epochs} - Loss: {total_loss:.4f}")
            
            ####################### step02 : Validation! #######################
            self.eval()  # 평가모드로 변경
            val_loss = 0
            with torch.no_grad():  # 검증 시에는 기울기 계산이 필요 없으므로, torch.no_grad()로 메모리 절약
                for batch in val_loader:
                    batch = batch.to(device).float()
                    output = self(batch)  
                    loss = criterion(output, batch)  # 손실 계산
                    val_loss += loss.item()

            # 매 epoch마다 검증 데이터로 손실 기록
            print(f"Epoch {epoch+1}/{config.epochs} - Validation Loss: {val_loss:.4f}")
            
            ####################### step02 : END #######################
            self.train()          # step01 : 학습모드 on
        
        # 학습 종료
        return