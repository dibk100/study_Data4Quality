# 함수 정리
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, log_loss, brier_score_loss
)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from IPython.display import display
import pandas as pd

# 성능 저장용 딕셔너리
def result(models,X_train_scaled,X_test_scaled,y_train,y_test):
    # 성능 저장용 딕셔너리
    results = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "ROC AUC": roc_auc_score(y_test, y_proba)
        }
    # 결과 출력
    results_df = pd.DataFrame(results).T.round(4)
    print("################## 성능 비교 : {name} #####################")
    display(results_df)
    
    return results

# 성능 지표
def calculate_metrics(y_DM_test,y_pred,y_proba):
    
    metrics = {
    "Accuracy": accuracy_score(y_DM_test, y_pred),
    "Precision": precision_score(y_DM_test, y_pred),
    "Recall": recall_score(y_DM_test, y_pred),
    "F1 Score": f1_score(y_DM_test, y_pred),
    "ROC AUC": roc_auc_score(y_DM_test, y_proba),
    "Log Loss": log_loss(y_DM_test, y_proba),
    "Brier Score": brier_score_loss(y_DM_test, y_proba)
    }
    
    return metrics

def roc(best_dt,x_DM_test_scaled,y_DM_test):
    # 예측 확률
    y_proba_best = best_dt.predict_proba(x_DM_test_scaled)[:, 1]

    #ROC Curve plot
    # ROC 커브 계산
    fpr, tpr, thresholds = roc_curve(y_DM_test, y_proba_best)
    auc_score = roc_auc_score(y_DM_test, y_proba_best)

    # 시각화
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve - Decision Tree")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

def show_feature_importance(best_dt,X) :
    # 변수 중요도 시각화
    # 변수 중요도 정리
    importances = best_dt.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    # 상위 15개 변수 시각화
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=importance_df.head(15))
    plt.title("Top 15 Feature Importances - Decision Tree")
    plt.tight_layout()
    plt.show()
    
def visualize_decision_tree(best_dt,X):
    # 시각화
    plt.figure(figsize=(20, 10))  # 크기는 필요에 따라 조절 가능
    plot_tree(
        best_dt,
        feature_names=X.columns,
        class_names=["No DM", "DM"],
        filled=True,
        rounded=True,
        max_depth=3,  # 트리 깊이 제한 없이 전체를 보고 싶다면 이 줄 제거 또는 None으로 설정
        fontsize=10
    )
    plt.title("Decision Tree Structure (Top Nodes)")
    plt.show()
    