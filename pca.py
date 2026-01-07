import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class PCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=6):
        self.mean = None
        self.components = None
        self.explained_variance_ratio = None
        self.n_components = n_components


    def fit(self, X, y=None):
        # standardize the data
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        # convariance matrix
        cov_matrix = np.cov(X, rowvar=False)
        # eigenvalues and eigenvectors from thwe matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        # sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        # select top n principal components
        self.components = eigenvectors[:, :self.n_components]
        explained_variance = eigenvalues[:self.n_components]
        # explained variance ratio
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio = explained_variance / total_variance

        return self


    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components)


    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
	