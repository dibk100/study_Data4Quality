# Deepimpute_Health 🔬🧬
건강검진 및 생체 정보 데이터(Data_01.csv)의 결측치를 대체할 딥러닝 기반 모델을 제시하고, 데이터 보간(Interpolation) 후 분류 모델을 통해 분석 결과를 도출.
   
### 01. **DataProcessing**
1. 결측치 대체 :   
   머신러닝/딥러닝 기반의 결측치 대체 모델을 활용하여 결측치를 효과적으로 보간한 후, 분류 모델을 개발.

2. 클래스 불균형 문제 해결 :   
   당뇨(Target_DM) 및 고혈압(Target_HT) 유병에 대한 각 분류 모델을 만들 때 **클래스 불균형 문제**를 고려하여 모델 성능을 개선


### 02. **Machine Learning / Deep Learning Analysis**
1. Multi-task Learning (Multi-label Classification)   
   당뇨 및 고혈압 유병 분류 문제에서 **single-label**을 고려하는 것이 아닌, 두 종류의 label을 함께 고려하는 **multi-task learning** (multi-label classification) 분석을 수행하기.   
   > **single-label**로 분석할 때와 비교.

2. 생체 나이 추정 모델  
   동일 데이터에 대해 나이(AGE 변수)를 예측하는 모델로 태스크를 변경하여 **생체 나이 추정 모형**을 만들기.   
   > 이 과정에서 데이터의 구조가 어떻게 변화하는지 설명하고, 나이 예측 모델에 맞는 성능 지표의 결과를 제시하기


