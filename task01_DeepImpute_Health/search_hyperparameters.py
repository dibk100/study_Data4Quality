from impute_model import *
from utils import *
import wandb
import torch

def sweep_train(input_dim, train_loader):
    """
    Sweep 실험을 실행하는 함수
    """
    wandb.init(tags=["Noise_autoencoder"])
    batch_size = wandb.config.batch_size
    noise_level = wandb.config.noise_level
    
    # 학습데이터를 train, val로 자르기, batch size고려해서 작성
    train_x, val_x = dataset_split(train_loader)
    train_loader, val_loader  = loader_dataset(train_x,val_x,batch_size)
    
    # 모델 생성
    model = DenoisingAutoencoder(input_dim=input_dim,noise_factor=noise_level)
    # 모델 학습
    model.model_train_wandb(train_loader,val_loader)
    wandb.finish()  # 실험이 끝났음을 알리기 위해 wandb.finish() 호출

def run():
    """
    하이퍼 파라미터는 메인 실행 함수.
    내가 직접 만들어서 함수들끼리 연결되는 게 이해 되는데 남들은 이걸 알아볼 수 있을까ㅠㅠ
    """

    # step01 : 데이터 전처리 파트 한번에 처리
    features_scaled, mask = data_processing()

    # 데이터를 텐서 형태로 수정
    X_tensor = torch.tensor(features_scaled, dtype=torch.float32)
    rows=X_tensor.shape[1]                                             # 입력 데이터 갯수
    
    # 학습/테스트 데이터셋으로 자르기
    train_X, test_X = dataset_split(features_scaled)                # test_X은 고정해야함. random_state=42
    
    # sweep 실험할 수 있게 설정
    sweep_config = load_yqml('dae')
    sweep_id = wandb.sweep(sweep_config, project="Data4Quality_task01_DAE")
    wandb.agent(sweep_id, function=lambda: sweep_train(rows, train_X))
    
if __name__ == "__main__":
    
    # wandb 로그인 시도 (force=True로 강제로 로그인)
    try:
        wandb.login(force=True)
    except Exception as e:
        print(f"Error logging in to wandb: {e}")
    
    run()