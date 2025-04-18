import torch
import torch.nn as nn
import wandb
from utils import *

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
    
    def train_model(self,train_loader, config_file):
        """
        Autoencoder 모델을 학습
        
        Args:
            train_loader (DataLoader): 학습 데이터
            config_file : (str) 파일명

        Returns:
            None
        """
        # wandb setting
        wandb.login()
        sweep_config = load_yqml(config_file)
        sweep_id = wandb.sweep(sweep_config, project="Data4Quality_task01_impute")
        
        # wandb 실험 시작
        wandb.init(project="Data4Quality_task01_impute", tags=["autoencoder"], sweep_id=sweep_id)
        config = wandb.config
        
        # 학습 준비
        device = 'cuda' if torch.cuda.is_available() else 'cpu'     # 일단 cpu로만 학습..
        self.to(device)                                             # 모델을 cpu에 올리기
        self.train()                                                # 학습모드on
        
        # 옵티마이저 설정
        if config.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
        elif config.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=config.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {config.optimizer}")
        
        # 손실 함수 설정
        criterion = nn.MSELoss()  # 평균제곱오차 고정

        # 학습
        for epoch in range(config.epochs):
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device)  # 배치 데이터를 장치에 올림
                optimizer.zero_grad()  # 기울기 초기화
                output = self(batch)  # 모델에 배치 데이터 넣고 출력값 얻음
                loss = criterion(output, batch)  # 손실 계산
                loss.backward()  # 역전파
                optimizer.step()  # 가중치 업데이트
                total_loss += loss.item()
            
            # 매 epoch마다 wandb로 손실 기록
            wandb.log({"train_loss": total_loss})
            print(f"Epoch {epoch+1}/{config.epochs} - Loss: {total_loss:.4f}")
        
        # 학습 종료 후 wandb 종료
        wandb.finish()
        return