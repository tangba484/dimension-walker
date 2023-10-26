from sklearn.decomposition import PCA

def 고윳값(data):
    result = pd.DataFrame({'설명가능한 분산 비율(고윳값)':pca.explained_varicance_,'기여율':pca.explained_variance_ratio}, index=np.array([f"pca{num+1}" for num in range(data.shape[1])]))
    
    result['누적기여율'] = result['기여율'].cumsum()
    
    return result
