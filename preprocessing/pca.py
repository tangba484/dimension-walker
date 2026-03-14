from sklearn.decomposition import PCA

def apply_pca(X, n_components):

    samples, time, features = X.shape

    X = X.reshape(-1, features)

    pca = PCA(n_components=n_components)

    X_pca = pca.fit_transform(X)

    X_pca = X_pca.reshape(samples, time, n_components)

    return X_pca, pca