import pandas as pd
import abc


class ClusteringMethod(abc.ABC):
    """
    Interface for clustering methods
    """

    def __init__(self, method_name: str | None = None) -> None:
        super().__init__()
        self._method_name = method_name

    @abc.abstractmethod
    def run_clustering(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Clusterization method application"""
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Name of the clustering method"""
        return self._method_name


class KMeansMethod(ClusteringMethod):
    """
    K-means clustering method implementation
    """

    def __init__(self) -> None:
        super().__init__(method_name='k-means')
    
    def run_clustering(self, dataset: pd.DataFrame) -> pd.DataFrame:
        pass


class HierarchicalMethod(ClusteringMethod):
    """
    Hierarchical clustering method implementation
    """

    def __init__(self) -> None:
        super().__init__(method_name='hierarchical')

    def run_clustering(self, dataset: pd.DataFrame) -> pd.DataFrame:
        pass


class GaussianMixutreMethod(ClusteringMethod):
    """
    Gaussian mixture method implementation
    """

    def __init__(self) -> None:
        super().__init__(method_name='gaussian mixture')
    
    def run_clustering(self, dataset: pd.DataFrame) -> pd.DataFrame:
        pass


class DBScanMethod(ClusteringMethod):
    """
    db-scan method implementation
    """

    def __init__(self) -> None:
        super().__init__(method_name='db-scan')
    
    def run_clustering(self, dataset: pd.DataFrame) -> pd.DataFrame:
        pass
