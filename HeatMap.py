import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
def function(df):
    pca_df = Pca.pca(df)


    pca_df["Close"] = df["Close"]

    # 예시 데이터 생성
    # 상관관계 행렬 계산
    correlation_matrix = pca_df.corr()

    # Heatmap으로 상관관계 시각화
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".1f")

    # 플롯 설정 (선택 사항)
    plt.title("상관관계 Heatmap")
    plt.show()
    return
