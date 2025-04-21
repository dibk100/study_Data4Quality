import os
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def count_images_in_dataset(data_path):
    """
    주어진 데이터 경로에서 train, val, test 폴더의 이미지 개수를 출력하고 리턴하는 함수
    """
    subfolders = ['train', 'val', 'test']
    
    def count_files_in_folder(folder_path):
        return sum(len(files) for _, _, files in os.walk(folder_path))
    
    # 결과를 딕셔너리로 저장
    image_counts = {}
    
    for subfolder in subfolders:
        path = os.path.join(data_path, subfolder)
        count = count_files_in_folder(path)
        image_counts[subfolder] = count
        print(f"{subfolder.capitalize()} 이미지 개수: {count}")
    
    # 이미지 개수 딕셔너리 리턴
    return image_counts
    
    

def count_and_visualize_images(data_path):
    """
    주어진 경로에서 train, val, test 폴더를 순회하여 각 클래스의 이미지 개수를 세고 시각적으로 표현하는 함수
    """
    # 폴더 이름과 클래스 이름
    categories = ['train', 'val', 'test']
    labels = ['NORMAL', 'PNEUMONIA', 'COVID19']
    counts = {'train': [0, 0, 0], 'val': [0, 0, 0], 'test': [0, 0, 0]}
    
    # 각 폴더 내 이미지 개수 집계
    for cat in categories:
        cat_path = os.path.join(data_path, cat)
        
        for i, label in enumerate(labels):
            label_path = os.path.join(cat_path, label, '*')
            counts[cat][i] = len(glob(label_path))
    
    # 시각적으로 표현
    # ---- Bar Plot ----
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # 막대그래프 (이미지 개수)
    ax[0].bar(labels, counts['train'], color='skyblue', label='train')
    ax[0].bar(labels, counts['val'], bottom=counts['train'], color='salmon', label='val')
    ax[0].bar(labels, counts['test'], bottom=[i+j for i,j in zip(counts['train'], counts['val'])], color='lightgreen', label='test')
    ax[0].set_title('Image Count per Class (Stacked)')
    ax[0].set_xlabel('Class')
    ax[0].set_ylabel('Number of Images')
    ax[0].legend()
    
    # 퍼센트 (원형 그래프)
    total_counts = [sum(counts[cat]) for cat in categories]
    ax[1].pie(total_counts, labels=categories, autopct='%1.1f%%', colors=['skyblue', 'salmon', 'lightgreen'], startangle=140)
    ax[1].set_title('Proportion of Images in Train/Val/Test')
    
    plt.tight_layout()
    plt.show()

def check_image_sizes(data_path):
    """
    주어진 경로에 있는 모든 이미지들의 크기를 확인하고 출력하는 함수
    """
    categories = ['train', 'val', 'test']
    labels = ['NORMAL', 'PNEUMONIA', 'COVID19']
    
    # 각 폴더 내 이미지 크기 확인
    for cat in categories:
        cat_path = os.path.join(data_path, cat)
        
        for label in labels:
            label_path = os.path.join(cat_path, label)
            
            # 해당 폴더 내 모든 이미지 경로 가져오기
            img_paths = glob(os.path.join(label_path, '*'))
            
            # 각 이미지의 크기 확인
            for idx,img_path in enumerate(img_paths):
                with Image.open(img_path) as img:
                    img_size = img.size  # (width, height)
                    if idx%100 == 0:
                        print(f"Image: {idx+1} - Size: {img_size}")
                        
def get_transform(img_size=(224, 224)):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # 흑백 이미지를 3채널로 확장
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

# def get_datasets(data_dir, transform):
#     train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
#     val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform)
#     test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)
#     return train_dataset, val_dataset, test_dataset

def get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def get_customdatasets(data_dir, transform, target_classes=None):
    def filter_dataset(dataset, class_to_idx, target_classes):
        # 선택된 클래스만 샘플로 유지
        if target_classes is None:
            return dataset

        # 선택된 클래스 인덱스
        target_class_indices = [class_to_idx[cls_name] for cls_name in target_classes]
        
        # samples와 targets 필터링
        filtered_samples = [(path, label) for path, label in dataset.samples if label in target_class_indices]
        dataset.samples = filtered_samples
        dataset.targets = [label for _, label in filtered_samples]

        # 라벨을 0, 1로 다시 매핑
        label_map = {class_to_idx[cls_name]: i for i, cls_name in enumerate(target_classes)}
        dataset.targets = [label_map[label] for label in dataset.targets]
        dataset.samples = [(path, label_map[label]) for path, label in dataset.samples]
        
        return dataset

    # ImageFolder 로딩
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)

    # 클래스 필터링
    if target_classes is not None:
        train_dataset = filter_dataset(train_dataset, train_dataset.class_to_idx, target_classes)
        val_dataset = filter_dataset(val_dataset, val_dataset.class_to_idx, target_classes)
        test_dataset = filter_dataset(test_dataset, test_dataset.class_to_idx, target_classes)

    return train_dataset, val_dataset, test_dataset