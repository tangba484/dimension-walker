import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def HeatMap(left_df , target):
    target = target.reset_index()

    left_df = pd.concat([left_df, target['Adj Close']], axis=1)
    print(left_df)
    # 예시 데이터 생성
    # 상관관계 행렬 계산
    correlation_matrix = left_df.corr()

    # Heatmap으로 상관관계 시각화
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".1f")

    # 플롯 설정 (선택 사항)
    plt.title("상관관계 Heatmap")
    plt.show()
    return