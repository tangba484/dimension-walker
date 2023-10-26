
pca_df = Pca.pca(data)


pca_df["close"] = data["Close"]





import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 예시 데이터 생성
# 상관관계 행렬 계산
correlation_matrix = pca_df.corr()

# Heatmap으로 상관관계 시각화
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".1f")

# 플롯 설정 (선택 사항)
plt.title("상관관계 Heatmap")
plt.show()
import Pca

