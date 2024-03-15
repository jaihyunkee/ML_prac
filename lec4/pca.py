import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import gc
import os

import matplotlib.pyplot as plt
import numpy as np


class PCA(object):
    """
    This class implements Principal Component Analysis (PCA).

    You may use any function, method or class from numpy.

    Shape (N, D) denotes a N x d matrix where N is the number of examples, and 
    D is the number of features.

    Attributes:
        n_components: The number of components to keep.

    """

    def __init__(self, n_components: int = None) -> None:
        """
        Class constructor. Initialize the class with the number of components to
        keep.

        Args:
            n_components: The number of components to keep, if None all components
                are kept.

        Returns:
            None
        """

        # >> YOUR CODE HERE

        self.n_components = n_components
        # << END OF YOUR CODE

    def cov(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the covariance matrix of the data.

        Args:
            X: The data to be decomposed into components of shape (N, D).

        Returns:
            cov_matrix: The covariance matrix of shape (D, D).

        Example:
            >>> X = np.array([[1, 2], [3, 4]])
            >>> pca = PCA(n_components=2)
            >>> cov_matrix = pca.cov(X)
            >>> print(cov_matrix)
            [[0.5 0.5]
             [0.5 0.5]]
        """

        # >> YOUR CODE HERE

        cov_matrix = np.cov(X, rowvar = False)
        return cov_matrix
        # << END OF YOUR CODE

    def eig(self, cov: np.ndarray) -> tuple:
        """
        Compute the eigenvalues and eigenvectors of a covariance matrix given a
        covariance matrix cov.

        Args:
            cov: The covariance matrix of shape (D, D).

        Returns:
            eigen_values: The eigenvalues of shape (D, ).
            eigen_vectors: The eigenvectors of shape (D, D).

        Example:
            >>> cov = np.array([[1, 0], [0, 1]])
            >>> pca = PCA(n_components=1)

            >>> eigen_values, eigen_vectors = pca.eig(cov)
            >>> print(eigen_values)
            [1. 1.]
            >>> print(eigen_vectors)
            [[1. 0.]
             [0. 1.]]
        """
        
        # >> YOUR CODE HERE
        eigen_values, eigen_vectors = np.linalg.eigh(cov)
        return eigen_values, eigen_vectors
        # << END OF YOUR CODE

    def fit(self, X) -> None:
        """
        Fit the data to the model. First, compute the covariance matrix of the
        data. Then, compute the eigenvalues and eigenvectors of the covariance
        matrix. Sort the eigenvalues and eigenvectors by the eigenvalues in
        descending order. Compute the components of the data. Finally, compute
        the explained variance ratio. Store components, explained_variance_ratio
        in the class.


        Args:
            X: The data to be decomposed into components of shape (N, D).

        Returns:
            None.

        Example:
            >>> X = np.array([[1, 2], [3, 4]])
            >>> pca = PCA(n_components=1)
            >>> pca.fit(X)
            >>> print(pca.components)
            [[0.70710678 0.70710678]]
            >>> print(pca.explained_variance_ratio)
            [1.]

        """
        # >> YOUR CODE HERE

        cov = self.cov(X)
        eigen_values, eigen_vectors = self.eig(cov)
        idx = eigen_values.argsort()[::-1]
        eigenvalues = eigen_values[idx]
        eigenvectors = eigen_vectors[:, idx]

        self.components = eigenvectors[:, :self.n_components].T
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio = eigenvalues[:self.n_components] / total_variance
        # << END OF YOUR CODE

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the data into the new space given by the components. Note that
        you should use the mean of the training data to subtract off the mean
        of the test data.

        Args:
            X: The data to be transformed of shape (N, D).

        Returns:
            The transformed data of shape (N, D).

        Example:
            >>> X = np.array([[1, 2], [3, 4]])
            >>> pca = PCA(n_components=1)
            >>> pca.fit(X)
            >>> print(pca.transform(X))
            [[-1.41421356]
             [ 1.41421356]]

        """
        X_transformed = np.dot( X - X.mean(), self.components.T)

        return X_transformed

    def inverse_transform(self, X: np.ndarray, X_pca: np.ndarray) -> np.ndarray:
        """
        Transform the data back into the original space given by the components.
        Note that you should use the mean of the training data to add back the
        mean of the test data.


        Args:
            X: The original data of shape (N, D).
            X_pca: The data that has been dimensionally reduced of shape (N, D).

        Returns:
            X_inv: The inverse transformed data of shape (N, D).

        Example:
            >>> X = np.array([[1, 2], [3, 4]])
            >>> pca = PCA(n_components=1)
            >>> pca.fit(X)
            >>> X_pca = pca.transform(X)
            >>> print(pca.inverse_transform(X, X_pca))
            [[1. 2.]
             [3. 4.]]
        """

        X_inv = np.dot(X_pca, self.components) + X.mean()

        return X_inv

# ------------------------- DON'T MODIFY THE CODE BELOW -----------------------------


def load_mnist_data(path='mnist_train_new.csv') -> tuple:
    df = pd.read_csv(path)

    target = 'label'
    X = df.drop(columns=target).values
    y = df[target].values

    print(
        f'Loaded data from {path}:\n\r X dimension: {X.shape}, y dimension: {y.shape}')

    # Garbage collection to save memory
    del df
    gc.collect()

    return np.array(X), np.array(y)


def plot_explained_variance_ratio(explained_variance_ratio, filename: str = 'explained_variance_ratio.png') -> None:
    """
    Plot explained variance ratio of the data. The x-axis should be the number
    of components and the y-axis should be the cumulative explained variance 
    ratio. Save the plot to a file given by filename.

    Args:
        filename: The filename to save the plot to.

    Returns:
        None.
    """

    plt.plot(explained_variance_ratio,
             label='Explained Variance Ratio')

    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.legend()

    plt.savefig(os.path.join(os.path.dirname(__file__), filename))


def compare_digit_images(filename, X, X_pca):
    fig, axs = plt.subplots(2, 5, figsize=(1.5*5, 2*2))

    for i in range(5):
        img = X[i].reshape((28, 28))
        axs[0, i].imshow(img, cmap='gray')

    for i in range(5):
        img = X_pca[i].reshape((28, 28))
        axs[1, i].imshow(img, cmap='gray')

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), filename))


if __name__ == '__main__':
    os.system('cls' if os.name == 'nt' else 'clear')
    X, y = load_mnist_data(os.path.join(
        os.path.dirname(__file__), 'mnist_train_sub.csv'))

    np.random.seed(12345)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, shuffle=True, test_size=0.3)

    knn = KNeighborsClassifier()

    knn.fit(X_train, y_train)

    start_time = time.time()  # start time

    p_train = knn.predict(X_train)
    train_acc = sum(y_train == p_train) / len(y_train)

    p_val = knn.predict(X_val)
    val_acc = sum(y_val == p_val) / len(y_val)

    end_time = time.time()  # end time

    print(
        f"KNN's Performance before PCA: train_acc={train_acc}, val_acc={val_acc}")
    print(f'\tTime: {end_time - start_time} seconds')

    pca = PCA(n_components=20)


    pca.fit(X)

    X_train_pca = pca.transform(X_train)
    X_val_pca = pca.transform(X_val)
    knn.fit(X_train_pca, y_train)
    start_time = time.time()  # start time

    p_train_pca = knn.predict(X_train_pca)
    train_acc_pca = sum(y_train == p_train_pca) / len(y_train)

    p_val_pca = knn.predict(X_val_pca)
    val_acc_pca = sum(y_val == p_val_pca) / len(y_val)

    end_time = time.time()  # end time

    print(
        f"KNN's Performance after PCA: train_acc={train_acc_pca}, val_acc={val_acc_pca}")
    print(f'\tTime: {end_time - start_time} seconds')

    plot_explained_variance_ratio(pca.explained_variance_ratio)

    X_train_inv = pca.inverse_transform(X, X_train_pca)
    compare_digit_images('compare.png', X_train, X_train_inv)

    print('Done!')
