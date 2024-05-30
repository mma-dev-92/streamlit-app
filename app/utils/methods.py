import pandas as pd
import abc
import matplotlib.pyplot as plt


class ClusteringMethod(abc.ABC):
    """
    Interface for clustering methods
    """

    def __init__(self, method_name: str | None = None) -> None:
        super().__init__()
        self._method_name = method_name

    @abc.abstractmethod
    def run(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Clusterization method application"""
        raise NotImplementedError

    @abc.abstractmethod
    def generate_plots(self, clustering_results: pd.DataFrame) -> list[plt.figure.Figure]:
        """Returns the list of plots to be displayed for results visualization"""
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Method name"""
        return self._method_name

    @property
    @abc.abstractmethod
    def short_description(self) -> str:
        """Short description of a method"""
        raise NotImplementedError


class KMeansMethod(ClusteringMethod):
    """
    K-means clustering method implementation
    """

    def __init__(self) -> None:
        super().__init__(method_name='k-means')
    
    def run(self, dataset: pd.DataFrame) -> pd.DataFrame:
        pass


class HierarchicalMethod(ClusteringMethod):
    """
    Hierarchical clustering method implementation
    """

    def __init__(self) -> None:
        super().__init__(method_name='hierarchical')

    def run(self, dataset: pd.DataFrame) -> pd.DataFrame:
        pass


class GaussianMixutreMethod(ClusteringMethod):
    """
    Gaussian mixture method implementation
    """

    def __init__(self) -> None:
        super().__init__(method_name='gaussian mixture')
    
    def run(self, dataset: pd.DataFrame) -> pd.DataFrame:
        pass


class DBScanMethod(ClusteringMethod):
    """
    db-scan method implementation
    """

    def __init__(self) -> None:
        super().__init__(method_name='db-scan')
    
    def run(self, dataset: pd.DataFrame) -> pd.DataFrame:
        pass
