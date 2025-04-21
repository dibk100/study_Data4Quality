import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from utils import *
import time
from torch.utils.data import TensorDataset, DataLoader

class AgeRegressor(nn.Module):
    def __init__(self, input_dim):
        super(AgeRegressor, self).__init__()
        
        # 모델 구조
        self.dense1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.dense2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.output = nn.Linear(64, 1)  # 출력은 하나의 값
        
    def forward(self, x):
        x = torch.relu(self.dense1(x))
        x = self.dropout1(x)
        x = torch.relu(self.dense2(x))
        x = self.dropout2(x)
        
        output = self.output(x)  # 선형 회귀 출력
        return output
    
    def model_train(self, X_train, y_train, num_epochs=20, batch_size=32, learning_rate=0.001):
        # 학습기 설정
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)  # 모델을 GPU로 전송

        # 데이터 텐서 변환
        X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

        # 데이터로더 설정
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 손실 함수 및 옵티마이저 설정
        criterion = nn.MSELoss()  # 회귀 문제이므로 MSE 손실 함수
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # 학습 루프
        for epoch in range(num_epochs):
            for inputs, targets in train_loader:
                optimizer.zero_grad()  # 기울기 초기화
                
                # 모델 예측
                outputs = self(inputs)
                
                # 손실 계산
                loss = criterion(outputs, targets)
                
                # 역전파 및 최적화
                loss.backward()
                optimizer.step()

            if epoch%10==0 :
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        print("Training completed!")

class SingleLabelClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SingleLabelClassifier, self).__init__()
        
        self.dense1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.dense2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)

        # 출력층 하나만!
        self.output = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.dense1(x))
        x = self.dropout1(x)
        x = torch.relu(self.dense2(x))
        x = self.dropout2(x)
        
        output = torch.sigmoid(self.output(x))  # sigmoid 적용
        return output
    
    def model_train_fixed_hyperparameter(self, X_train, y_train):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'     # gpu 
        if device!='cuda':
            raise RuntimeError("CUDA device not available. Please check if a GPU is available.")

        # 데이터 준비
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32)
        
        if y_tensor.dim() == 1:
            y_tensor = y_tensor.unsqueeze(1)

        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        ####################### SETTING #######################
        self.to(device)                                             # 모델을 gpu에 올리기
        self.train()                                                # 학습모드on
        
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        num_epochs = 20
        
        # 학습 루프
        for epoch in range(num_epochs):
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        return
    
class MultiLabelClassifier(nn.Module):
    def __init__(self, input_dim):
        super(MultiLabelClassifier, self).__init__()
        
        # 모델 레이어 정의
        self.dense1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.dense2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        
        ################################# 두 개의 출력층 (sigmoid)  ####
        self.output_dm = nn.Linear(64, 1)
        self.output_ht = nn.Linear(64, 1)
    
    def forward(self, x):
        # 순차적으로 레이어를 호출
        x = torch.relu(self.dense1(x))
        x = self.dropout1(x)
        x = torch.relu(self.dense2(x))
        x = self.dropout2(x)
        
        # 두 개의 출력
        output_dm = torch.sigmoid(self.output_dm(x))
        output_ht = torch.sigmoid(self.output_ht(x))
        
        return output_dm, output_ht
    
    def model_train_fixed_hyperparameter(self,X_train,y_dm_train,y_ht_train):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'     # gpu 
        if device!='cuda':
            raise RuntimeError("CUDA device not available. Please check if a GPU is available.")

        # 데이터 준비
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_dm_tensor = torch.tensor(y_dm_train, dtype=torch.float32).unsqueeze(1)  # shape: (N, 1)
        y_ht_tensor = torch.tensor(y_ht_train, dtype=torch.float32).unsqueeze(1)

        train_dataset = TensorDataset(X_tensor, y_dm_tensor, y_ht_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        ####################### SETTING #######################
        self.to(device)                                             # 모델을 gpu에 올리기
        self.train()                                                # 학습모드on
        
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        num_epochs = 20
        
        # 학습 루프
        for epoch in range(num_epochs):
            for inputs, y_dm, y_ht in train_loader:
                inputs = inputs.to(device)
                y_dm = y_dm.to(device)
                y_ht = y_ht.to(device)
                
                # 기울기 초기화
                optimizer.zero_grad()

                # 모델 예측
                output_dm, output_ht = self(inputs)

                # 손실 계산
                loss_dm = criterion(output_dm, y_dm)
                loss_ht = criterion(output_ht, y_ht)
                loss = loss_dm + loss_ht  # 두 출력의 손실 합산

                # 역전파
                loss.backward()

                # 최적화
                optimizer.step()

            # 에폭마다 손실 출력
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        return
        
class GaussianNoise(nn.Module):
    """
    # GaussianNoise: 입력 데이터에 임의의 노이즈를 추가하여 모델이 결측치를 잘 복원하도록 유도
    """
    def __init__(self, noise_factor=0.1):
        super(GaussianNoise, self).__init__()
        self.noise_factor = noise_factor

    def forward(self, x):
        noise = torch.randn_like(x) * self.noise_factor
        return x + noise

class DenoisingAutoencoder(nn.Module):
    """
    after.
    - 학습이 느리거나, 과적합 (Dropout, BatchNorm, LeakyReLU) 추가해서 실험 or GaussianNoise layers 추가
    
    """
    def __init__(self, input_dim, noise_factor):
        super(DenoisingAutoencoder, self).__init__()
        
        # GaussianNoise : 노이즈 비율
        self.noise_layer  = GaussianNoise(noise_factor)
        
        # 인코더                                                
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # 디코더
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    

    def forward(self, x):
        # 노이즈 추가
        noisy_x = self.noise_layer(x)
        
        # 인코딩
        encoded = self.encoder(noisy_x)
        
        # 디코딩
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
            wandb.log({"val_loss": val_loss})
            print(f"Epoch {epoch+1}/{config.epochs} - Validation Loss: {val_loss:.4f}")
            ####################### step02 : END #######################
            self.train()          # step01 : 학습모드 on
        
        # 학습 종료
        return

########################### 
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