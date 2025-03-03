import os
import re
import numpy as np
import pandas as pd
from itertools import product
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import hdbscan
from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering, DBSCAN, OPTICS, MeanShift, AffinityPropagation, AgglomerativeClustering, Birch


class ClusteringAlgorithm:
    """
    Represents a clustering algorithm with its associated parameters.
    """
    def __init__(self, name, algorithm_class, optimum_k, params, logger=None):
        """
        Initializes a clustering algorithm.
        
        :param name: Name of the algorithm.
        :param algorithm_class: Class reference of the algorithm.
        :param optimum_k: Boolean indicating if the optimal number of clusters should be determined.
        :param params: Dictionary of algorithm parameters.
        """
        # Verificación de que algorithm_class es una referencia válida a una clase
        if not isinstance(algorithm_class, type):
            raise TypeError(f"Expected a class reference, but got {type(algorithm_class)}")
        
        self.logger = logger
        self.name = name
        self.algorithm_class = algorithm_class
        self.optimum_k = optimum_k
        self.params = params
        self.logger.info(f"Initialized algorithm: '{self.name}'")
    
    def to_dict(self):
        """
        Converts the algorithm's details to a dictionary.
        """
        return {
            'optimum_k': self.optimum_k,
            'class': self.algorithm_class,  # Pasa la referencia a la clase, no su nombre
            'params': self.params
        }

class ClusteringAlgorithms:
    """
    Manages a collection of clustering algorithms and helps with optimal k determination.
    """
    def __init__(self, max_k=10, logger=None, scaling_data=False):
        """
        Initializes the clustering algorithms manager.
        
        :param max_k: Maximum number of clusters for applicable algorithms.
        :param scaling_data: Boolean to indicate whether data should be scaled.
        """
        self.logger = logger
        self.MAX_K = max_k + 1
        self.scaling_data = scaling_data  # New parameter to control scaling
        self.algorithms = {}
        self.logger.info("Initializing default clustering algorithms...")
        self._initialize_default_algorithms()

    def _initialize_default_algorithms(self):
        """
        Initializes the default set of clustering algorithms.
        """
        default_algorithms = [
            ClusteringAlgorithm('KMeans', KMeans, True, {'n_clusters': list(range(2, self.MAX_K))}, self.logger),
            ClusteringAlgorithm('MiniBatchKMeans', MiniBatchKMeans, True, {'n_clusters': list(range(2, self.MAX_K))}, self.logger),
            ClusteringAlgorithm('GaussianMixture', GaussianMixture, True, {'n_components': list(range(2, self.MAX_K))}, self.logger),
            ClusteringAlgorithm('SpectralClustering', SpectralClustering, True, {'n_clusters': list(range(2, self.MAX_K)), 'affinity': ['nearest_neighbors', 'rbf']}, self.logger),
            ClusteringAlgorithm('DBSCAN', DBSCAN, False, {'eps': [0.1, 0.2, 0.3, 0.5, 1.0], 'min_samples': [5, 10, 15]}, self.logger),
            ClusteringAlgorithm('OPTICS', OPTICS, False, {'min_samples': [5, 10, 15, 20], 'max_eps': [1.0, 1.5, 2.0]}, self.logger),
            ClusteringAlgorithm('MeanShift', MeanShift, False, {}, self.logger),
            ClusteringAlgorithm('AffinityPropagation', AffinityPropagation, False, {'damping': [0.5, 0.7, 0.9]}, self.logger),
            ClusteringAlgorithm('AgglomerativeClustering', AgglomerativeClustering, True, {'n_clusters': list(range(2, self.MAX_K)), 'linkage': ['ward', 'complete', 'average', 'single']}, self.logger),
            ClusteringAlgorithm('BIRCH', Birch, True, {'n_clusters': list(range(2, self.MAX_K)), 'threshold': [0.1, 0.5, 1.0]}, self.logger),
            ClusteringAlgorithm('HDBSCAN', hdbscan.HDBSCAN, False, {'min_cluster_size': [5, 10, 15, 20]}, self.logger)
        ]
        
        for algo in default_algorithms:
            self.algorithms[algo.name] = algo
        self.logger.info("Default clustering algorithms initialized.")

    def add_algorithm(self, name, algorithm_class, optimum_k, params):
        """
        Adds a new clustering algorithm.
        """
        if name in self.algorithms:
            self.logger.error(f"The algorithm '{name}' already exists.")
            raise ValueError(f"The algorithm '{name}' already exists.")
        if not isinstance(algorithm_class, type):  # Verificación de tipo de clase
            raise TypeError(f"Expected a class reference, but got {type(algorithm_class)}")
        self.algorithms[name] = ClusteringAlgorithm(name, algorithm_class, optimum_k, params)
        self.logger.info(f"Added algorithm: {name}")
    
    def remove_algorithm(self, name):
        """
        Removes an existing clustering algorithm.
        """
        if name not in self.algorithms:
            self.logger.error(f"The algorithm '{name}' does not exist.")
            raise ValueError(f"The algorithm '{name}' does not exist.")
        del self.algorithms[name]
        self.logger.info(f"Removed algorithm: {name}")
    
    def modify_algorithm(self, name, optimum_k=None, params=None):
        """
        Modifies an existing clustering algorithm.
        """
        if name not in self.algorithms:
            self.logger.error(f"The algorithm '{name}' does not exist.")
            raise ValueError(f"The algorithm '{name}' does not exist.")
        if optimum_k is not None:
            self.algorithms[name].optimum_k = optimum_k
        if params is not None:
            self.algorithms[name].params = params
        self.logger.info(f"Modified algorithm: {name}")
    
    def get_algorithms(self):
        """
        Returns a dictionary representation of all clustering algorithms.
        """
        return {name: algo.to_dict() for name, algo in self.algorithms.items()}

    def tostring(self):
        """
        Returns a textual representation of the total number of clustering algorithms and their details.
        """
        num_algorithms = len(self.algorithms)
        result = "Clustering algorithms configuration\n"
        result += "-" * 40 + "\n"
        result += f"Algorithms: {num_algorithms}\n"
        result += "-" * 40 + "\n"
        
        for name, algo in self.algorithms.items():
            result += f"Algorithm: {name}\n"
            result += f"  Class: {algo.algorithm_class.__name__}\n"
            result += f"  Optimal k: {'Yes' if algo.optimum_k else 'No'}\n"
            result += f"  Parameters: {algo.params}\n"
            result += "-" * 40 + "\n"
        
        return result
    
    def scale_data(self, raw_data):
        """
        Scales the input data using StandardScaler if required.
        
        :param raw_data: The data to scale.
        :return: The scaled data (or original if not scaling).
        """
        if self.scaling_data:
            self.logger.warning("Scaling the data using StandardScaler...")
            scaler = StandardScaler()
            return scaler.fit_transform(raw_data)
        else:
            self.logger.warning("Data will not be scaled.")
            return raw_data
    
    def find_optimal_k(self, X, algorithm_info):
        """
        Determines the optimal number of clusters (k) for a given clustering algorithm.
        
        Parameters:
        - X: The dataset to be clustered.
        - algorithm_info: A dictionary containing the algorithm class and its respective parameter combinations.
        
        Returns:
        - optimal_k: The optimal number of clusters determined by silhouette score.
        - best_params: The parameter combination that produced the optimal clustering.
        """
        # Extract the algorithm and parameters
        algo_class = algorithm_info['class']
        algo_params = algorithm_info['params']
        
        # Check if algo_class is a valid class
        if not isinstance(algo_class, type):
            raise TypeError(f"Expected a class reference, but got {type(algo_class)}")

        best_params = None
        optimal_k = None
        best_score = -1
        scores = []
        
        # Iterate through all combinations of parameters
        for param_combination in product(*algo_params.values()):
            params = dict(zip(algo_params.keys(), param_combination))
            
            self.logger.info(f"Testing algorithm '{algo_class}' with parameters: {params}")
            
            try:
                # Create and fit the model with correct unpacking of params
                cluster_algo = algo_class(**params)
                cluster_algo.fit(X)
                
                # Determine labels based on the algorithm type
                labels = self._get_cluster_labels(cluster_algo, algo_class, X)
                
                # Calculate silhouette score
                score = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else -1
                scores.append(score)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    optimal_k = len(np.unique(labels))
                    
            except Exception as e:
                self.logger.error(f"Error with parameters {params} in algorithm '{algo_class}': {e}")
                continue
        
        if optimal_k is None:
            self.logger.error(f"Could not determine optimal k for '{algo_class}'.")
        else:
            self.logger.info(f"Best configuration for '{algo_class.__name__}': {best_params} -> optimal_k = {optimal_k}")
        
        return optimal_k, best_params

    def _get_cluster_labels(self, cluster_algo, algo_class, X):
        """
        Returns the cluster labels for the given clustering algorithm.
        """
        if isinstance(cluster_algo, (KMeans, GaussianMixture)):
            return cluster_algo.predict(X)
        elif isinstance(cluster_algo, SpectralClustering):
            return cluster_algo.labels_
        elif isinstance(cluster_algo, (DBSCAN, OPTICS)):
            return cluster_algo.labels_
        elif isinstance(cluster_algo, hdbscan.HDBSCAN):
            return cluster_algo.labels_
        return None
    
    def run_clustering(self, raw_data):
        """
        Runs the clustering evaluation for all algorithms and scales the data if necessary.
        """
        # Scale the data if needed
        X = self.scale_data(raw_data)
        
        optimal_clusters = {}
        
        for algorithm_name, algorithm_info in self.algorithms.items():
            self.logger.info(f"Evaluating algorithm '{algorithm_name}' for optimal values...")
            
            if algorithm_info.optimum_k:
                print(algorithm_info.to_dict())
                
                optimal_k, best_params = self.find_optimal_k(X, algorithm_info.to_dict())
                if optimal_k is not None:
                    optimal_clusters[algorithm_name] = {'optimal_k': optimal_k, 'params': best_params}
                    
        return optimal_clusters
