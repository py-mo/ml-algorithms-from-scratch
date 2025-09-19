from .linearRegression import LinearRegression
from .logisticRegression import LogisticRegression
from .lassoRegression import LassoRegression
from .ridgeRegression import RidgeRegression
from .softmaxRegression import SoftmaxRegression
from .poissonRegression import PoissonRegression
from .elasticNet import ElasticNet
from .gam import GAModel


__all__ = ["LinearRegression", "LogisticRegression", "LassoRegression"
           , "RidgeRegression", "SoftmaxRegression", "PoissonRegression"
           , "ElasticNet", "GAModel"]
