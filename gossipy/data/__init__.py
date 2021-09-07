from typing import Any, Tuple, Union
import numpy as np
import pandas as pd
import torch
from sklearn import datasets
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler, LabelEncoder

from .. import LOG


# AUTHORSHIP
__version__ = "0.0.0dev"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2021, gossipy"
__license__ = "MIT"
__maintainer__ = "Mirko Polato, PhD"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#


__all__ = ["DataHandler", "DataDispatcher", "load_classification_dataset"]


class DataHandler():
    def __getitem__(self, idx: int) -> Any:
        raise NotImplementedError()

    def size(self, dim: int=0) -> int:
        raise NotImplementedError()

    def get_eval_set(self) -> Tuple[Any, Any]:
        raise NotImplementedError()
    
    def get_train_set(self) -> Tuple[Any, Any]:
        raise NotImplementedError()

    def eval_size(self) -> int:
        raise NotImplementedError()


class DataDispatcher():
    def __init__(self,
                 data_handler: DataHandler,
                 n: int=0, #number of clients
                 eval_on_user: bool=True):
        assert(data_handler.size() >= n)
        if n <= 1: n = data_handler.size()
        self.data_handler = data_handler
        self.n = n
        self.tr_assignments = [[] for _ in range(n)]
        self.te_assignments = [[] for _ in range(n)]
        for i in range(data_handler.size()):
            self.tr_assignments[i % n].append(i)
        self.eval_on_user = eval_on_user
        if eval_on_user:
            for i in range(data_handler.eval_size()):
                self.te_assignments[i % n].append(i)

    def __getitem__(self, idx: int) -> Any:
        assert(0 <= idx < self.n), "Index %d out of range." %idx
        return self.data_handler.at(self.tr_assignments[idx]), \
               self.data_handler.at(self.te_assignments[idx], True)
    
    def size(self) -> int:
        return self.n

    def get_eval_set(self) -> Tuple[Any, Any]:
        return self.data_handler.get_eval_set()
    
    def has_test(self) -> bool:
        return self.data_handler.eval_size() > 0


UCI_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/"

UCI_URL_AND_CLASS = {
    "spambase" : (UCI_BASE_URL + "spambase/spambase.data", 57),
    "sonar" : (UCI_BASE_URL + "undocumented/connectionist-bench/sonar/sonar.all-data", 60),
    "ionosphere" : (UCI_BASE_URL + "ionosphere/ionosphere.data", 34),
    "abalone" : (UCI_BASE_URL + "abalone/abalone.data", 0),
    "banknote" : (UCI_BASE_URL + "00267/data_banknote_authentication.txt", 4)
}


def load_classification_dataset(name_or_path: str,
                                normalize: bool=True,
                                as_tensor: bool=True) -> Union[Tuple[torch.Tensor, torch.Tensor],
                                                               Tuple[np.ndarray, np.ndarray]]:
    if name_or_path == "iris":
        dataset = datasets.load_iris()
        X, y = dataset.data, dataset.target
    elif name_or_path == "breast":
        dataset = datasets.load_breast_cancer()
        X, y = dataset.data, dataset.target
    elif name_or_path == "digits":
        dataset = datasets.load_digits()
        X, y = dataset.data, dataset.target
    elif name_or_path == "wine":
        dataset = datasets.load_wine()
        X, y = dataset.data, dataset.target
    elif name_or_path in {"sonar", "ionosphere", "abalone", "banknote", "spambase"}:
        url, label_id = UCI_URL_AND_CLASS[name_or_path]
        LOG.info("Downloading dataset %s from '%s'." %(name_or_path, url))
        data = pd.read_csv(url, header=None).to_numpy()
        y = LabelEncoder().fit_transform(data[:, label_id])
        X = np.delete(data, [label_id], axis=1).astype('float64')
    else:
        X, y = load_svmlight_file(name_or_path)
        X = X.toarray()

    if normalize:
        X = StandardScaler().fit_transform(X)

    if as_tensor:
        X = torch.tensor(X).float()
        y = torch.tensor(y).long()#.reshape(y.shape[0], 1)

    return X, y


#TODO: download
# def load_recsys_dataset(name: str,
#                         path: str) -> Dict[int, List[Tuple[int, float]]]:
#     ratings = {}
#     if name == "ml100k" or name == "ml1m":
#         with open(os.path.join(path, name + ".txt"), "r") as f:
#             for line in f.readlines():
#                 u, i, r = list(map(int, line.strip().split(",")))
#                 if u not in ratings:
#                     ratings[u] = []
#                 ratings[u].append((i, r))
#     else:
#         raise ValueError("Unknown dataset %s." %name)
#     return ratings