import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import warnings

# Desactivar las advertencias de 'joblib'
warnings.filterwarnings("ignore", category=UserWarning, message="Could not find the number of physical cores")


class DimensionalityReduction:
    """
    This class implements dimensionality reduction using either PCA (Principal Component Analysis) or t-SNE (t-distributed Stochastic Neighbor Embedding).
    
    Attributes:
        method (str): The reduction method ('PCA' or 'TSNE').
        n_components (int): The number of components to reduce the data to.
        model: The model to be used (PCA or TSNE).
        logger: Logger instance to record execution logs.
        
    Methods:
        fit(X): Fits the model to the data X.
        transform(X): Transforms the data X using the fitted model.
        fit_transform(X): Fits the model and transforms the data in one step.
    """
    
    def __init__(self, method='PCA', n_components=2, logger=None):
        """
        Constructor to initialize the DimensionalityReduction class.
        
        :param method: Method of reduction ('PCA' or 'TSNE'). Default is 'PCA'.
        :param n_components: Number of components for dimensionality reduction. Default is 2.
        :param logger: Logger instance to log messages. Default is None.
        """
        self.method = method
        self.n_components = n_components
        self.model = None
        self.logger = logger  # Store the logger

        # Disable warnings related to CPU count in joblib
        warnings.filterwarnings("ignore", category=UserWarning, message="Could not find the number of physical cores")
        
        # Choose the model based on the selected method
        if method == 'PCA':
            self.model = PCA(n_components=n_components)
            if self.logger:
                self.logger.info(f"Initialized PCA model with {n_components} components.")
        elif method == 'TSNE':
            self.model = TSNE(n_components=n_components, n_jobs=1)  # Set n_jobs=1 to avoid parallelization issues
            if self.logger:
                self.logger.info(f"Initialized t-SNE model with {n_components} components.")
        else:
            raise ValueError("Unsupported method. Use 'PCA' or 'TSNE'.")
        
    def fit(self, X):
        """
        Fits the model to the input data X.
        
        :param X: Input data (feature matrix).
        """
        if self.logger:
            self.logger.info("Fitting the model to the data.")
        
        # It's important to standardize the data before applying PCA or t-SNE
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        if self.logger:
            self.logger.info("Data has been standardized.")
        
        self.model.fit(X)
        
        if self.logger:
            self.logger.info("Model fitting completed.")

    def transform(self, X):
        """
        Transforms the input data X using the fitted model.
        
        :param X: Input data (feature matrix).
        :return: Transformed data as a pandas DataFrame.
        """
        if self.model is None:
            raise ValueError("The model has not been fitted yet.")
        
        if self.logger:
            self.logger.info("Transforming the data using the fitted model.")
        
        # Standardize the data before applying PCA or t-SNE
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        if self.method == 'TSNE':
            # For t-SNE, we cannot use transform(), so we call fit_transform directly
            transformed_data = self.model.fit_transform(X)
        else:
            transformed_data = self.model.transform(X)

        # Convert the transformed data to a pandas DataFrame
        transformed_df = pd.DataFrame(transformed_data, columns=[f'{self.method}Component{i+1}' for i in range(self.n_components)])

        if self.logger:
            self.logger.info("Data transformation completed.")
        
        return transformed_df

    def fit_transform(self, X):
        """
        Fits the model and transforms the data in one step.
        
        :param X: Input data (feature matrix).
        :return: Transformed data as a pandas DataFrame.
        """
        if self.logger:
            self.logger.info("Starting fit_transform process.")
        
        # Fit and transform the data
        transformed_data = self.model.fit_transform(X)

        # Convert the transformed data to a pandas DataFrame
        transformed_df = pd.DataFrame(transformed_data, columns=[f'{self.method}Component{i+1}' for i in range(self.n_components)])

        if self.logger:
            self.logger.info("fit_transform process completed.")
        
        return transformed_df
