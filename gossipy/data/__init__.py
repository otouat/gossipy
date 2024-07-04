"""This module contains functions and classes to manage datasets loading and dispatching."""

import os
from abc import ABC, abstractmethod
from tkinter import Image
from typing import Any, Tuple, Union, Dict, List, Optional
import shutil
import numpy as np
from numpy.random import randint, shuffle, power, choice, dirichlet, permutation
import pandas as pd
from pathlib import Path
from pyparsing import ParseSyntaxException
import torch
import torchvision
from torch import Tensor, tensor
from sklearn import datasets
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import matplotlib.pyplot as plt
import numpy as np

from .. import LOG
from ..utils import download_and_unzip, download_and_untar
from torch.utils.data import Subset
from collections import Counter, defaultdict

# AUTHORSHIP
__version__ = "0.0.1"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2022, gossipy"
__license__ = "Apache License, Version 2.0"
__maintainer__ = "Mirko Polato, PhD"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#


__all__ = ["DataHandler",
           "DataDispatcher",
           "RecSysDataDispatcher",
           "load_classification_dataset",
           "load_recsys_dataset",
           "get_CIFAR10",
           "get_FashionMNIST",
           "get_FEMNIST"]

UCI_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/"

UCI_URL_AND_CLASS = {
    "spambase" : (UCI_BASE_URL + "spambase/spambase.data", 57),
    "sonar" : (UCI_BASE_URL + "undocumented/connectionist-bench/sonar/sonar.all-data", 60),
    "ionosphere" : (UCI_BASE_URL + "ionosphere/ionosphere.data", 34),
    "abalone" : (UCI_BASE_URL + "abalone/abalone.data", 0),
    "banknote" : (UCI_BASE_URL + "00267/data_banknote_authentication.txt", 4),
    #"dexter" : (UCI_BASE_URL + "dexter/DEXTER/", -1)
}


class DataHandler(ABC):
    def __init__(self):
        """Abstract class for data handlers.

        A :class:`DataHandler` class provides attributes and methods to manage a dataset.
        A subclass of :class:`DataHandler` must implement the following methods:

        - :meth:`__getitem__`
        - :meth:`at`
        - :meth:`size`
        - :meth:`get_eval_set`
        - :meth:`get_train_set`
        - :meth:`eval_size`
        """

        pass
    
    @abstractmethod
    def __getitem__(self, idx: Union[int, List[int]]) -> Any:
        """Get a sample (or samples) from the training set.
        
        Parameters
        ----------
        idx : int or list[int]
            The index or indices of the sample(s) to get.
        
        Returns
        -------
        Any
            The sample(s) at the given index(ices) in the training set.
        """

        pass
    
    @abstractmethod
    def at(self, 
           idx: Union[int, List[int]],
           eval_set: bool=False) -> Any:
        """Get a sample (or samples) from the training/test set.
        
        Parameters
        ----------
        idx : int or list[int]
            The index or indices of the sample(s) to get.
        eval_set : bool, default=False
            Whether to get the sample(s) from the training or the evaluation set.
        
        Returns
        -------
        Any
            The sample(s) at the given index(ices) in the training/evaluation set.
        """

        pass

    @abstractmethod
    def size(self, dim: int=0) -> int:
        """Get the size of the training set along a given dimension.

        Parameters
        ----------
        dim : int, default=0
            The dimension along which to get the size of the dataset.

        Returns
        -------
        int
            The size of the dataset along the given dimension.
        """

        pass

    @abstractmethod
    def get_eval_set(self) -> Tuple[Any, Any]:
        """Get the evaluation set of the dataset.

        Returns
        -------
        tuple[Any, Any]
            The evaluation set of the dataset.
        """

        ParseSyntaxException
    
    @abstractmethod
    def get_train_set(self) -> Tuple[Any, Any]:
        """Get the training set of the dataset.

        Returns
        -------
        tuple[Any, Any]
            The training set of the dataset.
        """

        pass

    @abstractmethod
    def eval_size(self) -> int:
        """Get the number of examples of the evaluation set.

        Returns
        -------
        int
            The size of the evaluation set of the dataset.
        """

        pass


class AssignmentHandler():

    def __init__(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    def uniform(self,
                y: Union[np.ndarray, torch.Tensor],
                n: int) -> List[np.ndarray]:
        """Distribute the examples uniformly across the users.

        Parameters
        ----------
        y: Union[np.ndarray, torch.Tensor]
            The labels. Not used.
        n: int
            The number of clients upon which the examples are distributed.

        Returns
        -------
        List[np.ndarray]
            The examples' ids assignment.
        """
        ex_client = y.shape[0] // n
        idx = permutation(y.shape[0])
        return [idx[range(ex_client*i, ex_client*(i+1))] for i in range(n)]

    def quantity_skew(self,
                      y: Union[np.ndarray, torch.Tensor],
                      n: int,
                      min_quantity: int=2,
                      alpha: float=4.) -> List[np.ndarray]:
        """
        Distribute the examples across the users according to the following probability density function:
        $P(x; a) = a x^{a-1}$
        where x is the id of a client (x in [0, n-1]), and a = `alpha` > 0 with
        - alpha = 1  => examples are equidistributed across clients;
        - alpha = 2  => the examples are "linearly" distributed across users;
        - alpha >= 3 => the examples are power law distributed;
        - alpha -> \infty => all users but one have `min_quantity` examples, and the remaining user all the rest.
        Each client is guaranteed to have at least `min_quantity` examples.

        Parameters
        ----------
        y: Union[np.ndarray, torch.Tensor]
            The labels. Not used.
        n: int
            The number of clients upon which the examples are distributed.
        min_quantity: int, default 2
            The minimum quantity of examples to assign to each user.
        alpha: float=4.
            Hyper-parameter of the power law density function  $P(x; a) = a x^{a-1}$.

        Returns
        -------
        List[np.ndarray]
            The examples' ids assignment.
        """
        assert min_quantity*n <= y.shape[0], "# of instances must be > than min_quantity*n"
        assert min_quantity > 0, "min_quantity must be >= 1"
        s = np.array(power(alpha, y.shape[0] - min_quantity*n) * n, dtype=int)
        m = np.array([[i] * min_quantity for i in range(n)]).flatten()
        assignment = np.concatenate([s, m])
        shuffle(assignment)
        return [np.where(assignment == i)[0] for i in range(n)]

    def classwise_quantity_skew(self,
                                y: Union[np.ndarray, torch.Tensor],
                                n: int,
                                min_quantity: int=2,
                                alpha: float=4.) -> List[np.ndarray]:
        assert min_quantity*n <= y.shape[0], "# of instances must be > than min_quantity*n"
        assert min_quantity > 0, "min_quantity must be >= 1"
        labels = list(range(len(torch.unique(y).numpy())))
        lens = [np.where(y == l)[0].shape[0] for l in labels]
        min_lbl = min(lens)
        assert min_lbl >= n, "Under represented class!"

        s = [np.array(power(alpha, lens[c] - n) * n, dtype=int) for c in labels]
        assignment = []
        for c in labels:
            ass = np.concatenate([s[c], list(range(n))])
            shuffle(ass)
            assignment.append(ass)

        res = [[] for _ in range(n)]
        for c in labels:
            idc = np.where(y == c)[0]
            for i in range(n):
                res[i] += list(idc[np.where(assignment[c] == i)[0]])

        return [np.array(r, dtype=int) for r in res]

    def label_quantity_skew(self,
                            y: Union[np.ndarray, torch.Tensor],
                            n: int,
                            class_per_client: int=2) -> List[np.ndarray]:
        """
        Suppose each party only has data samples of `class_per_client` (i.e., k) different labels.
        We first randomly assign k different label IDs to each party. Then, for the samples of each
        label, we randomly and equally divide them into the parties which own the label.
        In this way, the number of labels in each party is fixed, and there is no overlap between
        the samples of different parties.
        See: https://arxiv.org/pdf/2102.02079.pdf

        Parameters
        ----------
        y: Union[np.ndarray, torch.Tensor]
            The lables.
        n: int
            The number of clients upon which the examples are distributed.
        class_per_client: int, default 2
            The number of different labels in each client.

        Returns
        -------
        List[np.ndarray]
            The examples' ids assignment.
        """
        labels = set(torch.unique(y).numpy())
        assert 0 < class_per_client <= len(labels), "class_per_client must be > 0 and <= #classes"
        assert class_per_client * n >= len(labels), "class_per_client * n must be >= #classes"
        nlbl = [choice(len(labels), class_per_client, replace=False)  for u in range(n)]
        check = set().union(*[set(a) for a in nlbl])
        while len(check) < len(labels):
            missing = labels - check
            for m in missing:
                nlbl[randint(0, n)][randint(0, class_per_client)] = m
            check = set().union(*[set(a) for a in nlbl])
        class_map = {c:[u for u, lbl in enumerate(nlbl) if c in lbl] for c in labels}
        assignment = np.zeros(y.shape[0])
        for lbl, users in class_map.items():
            ids = np.where(y == lbl)[0]
            assignment[ids] = choice(users, len(ids))
        return [np.where(assignment == i)[0] for i in range(n)]

    def label_dirichlet_skew(self,
                             y: torch.Tensor,
                             n: int,
                             beta: float=.1) -> List[np.ndarray]:
        """
        The function samples p_k ~ Dir_n (beta) and allocate a p_{k,j} proportion of the instances of
        class k to party j. Here Dir(_) denotes the Dirichlet distribution and beta is a
        concentration parameter (beta > 0).
        See: https://arxiv.org/pdf/2102.02079.pdf

        Parameters
        ----------
        y: Union[np.ndarray, torch.Tensor]
            The lables.
        n: int
            The number of clients upon which the examples are distributed.
        beta: float, default .5
            The beta parameter of the Dirichlet distribution, i.e., Dir(beta).

        Returns
        -------
        List[np.ndarray]
            The examples' ids assignment.
        """
        assert beta > 0, "beta must be > 0"
        labels = set(torch.unique(y).numpy())
        pk = {c: dirichlet([beta]*n, size=1)[0] for c in labels}
        assignment = np.zeros(y.shape[0])
        for c in labels:
            ids = np.where(y == c)[0]
            shuffle(ids)
            shuffle(pk[c])
            assignment[ids[n:]] = choice(n, size=len(ids)-n, p=pk[c])
            assignment[ids[:n]] = list(range(n))

        return [np.where(assignment == i)[0] for i in range(n)]

    def label_pathological_skew(self,
                                y: Union[np.ndarray, torch.Tensor],
                                n: int,
                                shards_per_client: int=2) -> List[np.ndarray]:
        """
        The function first sort the data by label, divide it into `n * shards_per_client` shards, and
        assign each of n clients `shards_per_client` shards. This is a pathological non-IID partition
        of the data, as most clients will only have examples of a limited number of classes.
        See: http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf

        Parameters
        ----------
        y: Union[np.ndarray, torch.Tensor]
            The lables.
        n: int
            The number of clients upon which the examples are distributed.
        shards_per_client: int, default 2
            Number of shards per client.

        Returns
        -------
        List[np.ndarray]
            The examples' ids assignment.
        """
        sorted_ids = np.argsort(y)
        n_shards = int(shards_per_client * n)
        shard_size = int(np.ceil(len(y) / n_shards))
        assignments = np.zeros(y.shape[0])
        perm = permutation(n_shards)
        j = 0
        for i in range(n):
            for _ in range(shards_per_client):
                left = perm[j] * shard_size
                right = min((perm[j]+1) * shard_size, len(y))
                assignments[sorted_ids[left:right]] = i
                j += 1
        return [np.where(assignments == i)[0] for i in range(n)]


class DataDispatcher():
    def __init__(self,
                 data_handler: DataHandler,
                 n: int=0, #number of clients
                 eval_on_user: bool=True,
                 auto_assign: bool=True):
        """DataDispatcher is responsible for assigning data to clients.

        The assignment is done by shuffling the data and assigning it uniformly to the clients.
        If a specific assignment is required, use the :meth:`set_assignments` method.

        Parameters
        ----------
        data_handler : DataHandler
            The data handler that contains the data to be distributed.
        n : int, default=0
            The number of clients. If 0, the number of clients is set to the number of
            examples in the training set.
        eval_on_user : bool, default=True
            If True, a test set is assigned to each user.
        auto_assign : bool, default=True
            If True, the data is shuffled and assigned to the clients.
        """

        assert(data_handler.size() >= n)
        if n <= 1: n = data_handler.size()
        self.data_handler = data_handler
        self.n = n
        self.eval_on_user = eval_on_user
        self.tr_assignments = None
        self.te_assignments = None
        if auto_assign:
            self.assign()
    
    def set_assignments(self, tr_assignments: List[int],
                              te_assignments: Optional[List[int]]) -> None:
        """Set the specified assignments for the training and test sets.
        
        The assignment must be provided as a list of integers with the same length as the
        number of examples in the training/test set. Each integer is the index of the client
        that will receive the example.

        Parameters
        ----------
        tr_assignments : list of int
            The list of assignments for the training set.
        te_assignments : list of int, default=None
            The list of assignments for the test set. If None, the test set is not assigned.
        """

        assert len(tr_assignments) == self.n
        assert not te_assignments or len(te_assignments) == self.n
        self.tr_assignments = tr_assignments
        if te_assignments:
            self.te_assignments = te_assignments
        else:
            self.te_assignments = [[] for _ in range(self.n)]


    def assign(self, seed: Optional[int]=42) -> None:
        """Assign the data to the clients.

        The assignment is done by shuffling the data and assigning it uniformly to the clients.

        Parameters
        ----------
        seed : int, default=42
            The seed for the random number generator.
        """

        assign_handler = AssignmentHandler(seed)
        self.tr_assignments = assign_handler.uniform(self.data_handler.ytr, self.n)
        if self.eval_on_user:
            self.te_assignments = assign_handler.uniform(self.data_handler.yte, self.n)
        else:
            self.te_assignments = [[] for _ in range(self.n)]


    def __getitem__(self, idx: int) -> Any:
        """Return the data for the specified client.

        Parameters
        ----------
        idx : int
            The index of the client.
        
        Returns
        -------
        Any
            The data to assign to the specified client.
        """

        assert 0 <= idx < self.n, "Index %d out of range." %idx
        return self.data_handler.at(self.tr_assignments[idx]), \
               self.data_handler.at(self.te_assignments[idx], True)
    
    def size(self) -> int:
        """Returns the number of clients.

        Returns
        -------
        int
            The number of clients.
        """

        return self.n

    def get_eval_set(self) -> Tuple[Any, Any]:
        """Return the entire test set.

        Returns
        -------
        tuple[Any, Any]
            The test set.
        """

        return self.data_handler.get_eval_set()
    
    def has_test(self) -> bool:
        """Return True if there is a test set.

        Returns
        -------
        bool
            Whether there is a test set or not.
        """

        return self.data_handler.eval_size() > 0
    
    def __repr__(self) -> str:
        return str(self)
    
    def __str__(self) -> str:
        return "DataDispatcher(handler=%s, n=%d, eval_on_user=%s)" \
                %(self.data_handler, self.n, self.eval_on_user)

        
class OLDCustomDataDispatcher(DataDispatcher):
    def assign(self, seed: int = 42) -> None:
        self.tr_assignments = [[] for _ in range(self.n)]
        self.te_assignments = [[] for _ in range(self.n)]

        n_ex = self.data_handler.size()
        ex_x_user = n_ex // self.n  # Ensure equal partitioning

        for idx in range(self.n):
            start_index = idx * ex_x_user
            end_index = start_index + ex_x_user
            self.tr_assignments[idx] = list(range(start_index, min(end_index, n_ex)))

        if self.eval_on_user:
            n_eval_ex = self.data_handler.eval_size()
            eval_ex_x_user = n_eval_ex // self.n
            for idx in range(self.n):
                start_index = idx * eval_ex_x_user
                end_index = start_index + eval_ex_x_user
                self.te_assignments[idx] = list(range(start_index, min(end_index, n_eval_ex)))


class RecSysDataDispatcher(DataDispatcher):
    from .handler import RecSysDataHandler
    def __init__(self,
                 data_handler: RecSysDataHandler):
        """RecSysDataDispatcher is responsible for assigning recommendation data to clients.

        Differently from the DataDispatcher, the assignment is done by assigning a single
        example to each client. It is assumed that the :class:`RecSysDataHandler` contains the
        user-item rating matrix, thus a client is a row of the matrix.

        Parameters
        ----------
        data_handler : RecSysDataHandler
            The data handler that contains the data to be distributed.
        """

        self.data_handler = data_handler
        self.n = self.data_handler.n_users
        self.eval_on_user = True
    
    # docstr-coverage:inherited
    def assign(self, seed=42):
        torch.manual_seed(seed)
        self.assignments = torch.randperm(self.data_handler.size()).tolist()


    # docstr-coverage:inherited
    def __getitem__(self, idx: int) -> Any:
        assert(0 <= idx < self.n), "Index %d out of range." %idx
        return self.data_handler.at(self.assignments[idx]), \
               self.data_handler.at(self.assignments[idx], True)
    
    # docstr-coverage:inherited
    def size(self) -> int:
        return self.n

    # docstr-coverage:inherited
    def get_eval_set(self) -> Tuple[Any, Any]:
        return None
    
    # docstr-coverage:inherited
    def has_test(self) -> bool:
        return False
    
    def __str__(self) -> str:
        return f"RecSysDataDispatcher(handler={self.data_handler}, eval_on_user={self.eval_on_user})"


def load_classification_dataset(name_or_path: str,
                                normalize: bool=True,
                                as_tensor: bool=True) -> Union[Tuple[torch.Tensor, torch.Tensor],
                                                               Tuple[np.ndarray, np.ndarray]]:
    """Loads a classification dataset.

    A dataset can be loaded from *svmlight* file or can be one of the following:
    iris, breast, digits, wine, reuters, spambase, sonar, ionosphere, abalone, banknote.

    Parameters
    ----------
    name_or_path : str
        The name of the dataset or the path to the dataset.
    normalize : bool, default=True
        Whether to normalize (standard scaling) the data or not.
    as_tensor : bool, default=True
        Whether to return the data as a tensor or as a numpy array.
    
    Returns
    -------
    tuple[torch.Tensor, torch.Tensor] or tuple[np.ndarray, np.ndarray]
        A tuple containing the data and the labels with the specified type.
    """

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
    elif name_or_path == "reuters":
        url = "http://download.joachims.org/svm_light/examples/example1.tar.gz"
        folder = download_and_untar(url)[0]
        X_tr, y_tr = load_svmlight_file(folder + "/train.dat")
        X_te, y_te = load_svmlight_file(folder + "/test.dat")
        X_te = np.pad(X_te.toarray(), [(0, 0), (0, 17)], mode='constant', constant_values=0)
        X = np.vstack([X_tr.toarray(), X_te])
        y = np.concatenate([y_tr, y_te])
        y = LabelEncoder().fit_transform(y)
        shutil.rmtree(folder)
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


# TODO: add other recsys datasets
def load_recsys_dataset(name: str,
                        path: str=".") -> Tuple[Dict[int, List[Tuple[int, float]]], int, int]:
    """Load a recsys dataset.

    Currently, only the following datasets are supported: ml-100k, ml-1m, ml-10m and ml-20m.
    
    Parameters
    ----------
    name : str
        The name of the dataset.
    path : str, default="."
        The path in which to download the dataset.
    
    Returns
    -------
    tuple[dict[int, list[tuple[int, float]]], int, int]
        A tuple contining the ratings, the number of users and the number of items.
        Ratings are represented as a dictionary mapping user ids to a list of tuples (item id, rating).
    """

    ratings = {}
    if name in {"ml-100k", "ml-1m", "ml-10m", "ml-20m"}:
        folder = download_and_unzip("https://files.grouplens.org/datasets/movielens/%s.zip" %name)[0]
        if name == "ml-100k":
            filename = "u.data"
            sep = "\t"
        elif name == "ml-20m":
            filename = "ratings.csv"
            sep = ","
        else:
            filename = "ratings.dat"
            sep = "::"

        ucnt = 0
        icnt = 0
        with open(os.path.join(path, folder, filename), "r") as f:
            umap = {}
            imap = {}
            for line in f.readlines():
                u, i, r = list(line.strip().split(sep))[0:3]
                u, i, r = int(u), int(i), float(r)
                if u not in umap:
                    umap[u] = ucnt
                    ratings[umap[u]] = []
                    ucnt += 1
                if i not in imap:
                    imap[i] = icnt
                    icnt += 1
                ratings[umap[u]].append((imap[i], r))

        shutil.rmtree(folder)
    else:
        raise ValueError("Unknown dataset %s." %name)
    return ratings, ucnt, icnt


def get_CIFAR10(path: str="./data",
                as_tensor: bool=True) -> Union[Tuple[Tuple[np.ndarray, list], Tuple[np.ndarray, list]],
                                               Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]]:
    """Returns the CIFAR10 dataset.

    The method downloads the dataset if it is not already present in `path`.
    
    Parameters
    ----------
    path : str, default="./data"
        Path to save the dataset, by default "./data".
    as_tensor : bool, default=True
        If True, the dataset is returned as a tuple of pytorch tensors.
        Otherwise, the dataset is returned as a tuple of numpy arrays.
        By default, True.
    
    Returns
    -------
    tuple[tuple[np.ndarray, list], tuple[np.ndarray, list]] or tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]
        Tuple of training and test sets of the form :math:`(X_train, y_train), (X_test, y_test)`.
    """

    download = not Path(os.path.join(path, "/cifar-10-batches-py")).is_dir()
    train_set = torchvision.datasets.CIFAR10(root=path,
                                             train=True,
                                             download=download)
    test_set = torchvision.datasets.CIFAR10(root=path,
                                            train=False,
                                            download=download)
    if as_tensor:
        train_set = tensor(train_set.data).float().permute(0,3,1,2) / 255.,\
                    tensor(train_set.targets)
        test_set = tensor(test_set.data).float().permute(0,3,1,2) / 255.,\
                   tensor(test_set.targets)
    else:
        train_set = train_set.data, train_set.targets
        test_set = test_set.data, test_set.targets

    return train_set, test_set

import os
from pathlib import Path
from typing import Tuple, Union

import torchvision
import numpy as np
from torch import Tensor, tensor

def get_CIFAR100(path: str="./data",
                as_tensor: bool=True) -> Union[Tuple[Tuple[np.ndarray, list], Tuple[np.ndarray, list]],
                                               Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]]:
    """Returns the CIFAR100 dataset.

    The method downloads the dataset if it is not already present in `path`.
    
    Parameters
    ----------
    path : str, default="./data"
        Path to save the dataset, by default "./data".
    as_tensor : bool, default=True
        If True, the dataset is returned as a tuple of pytorch tensors.
        Otherwise, the dataset is returned as a tuple of numpy arrays.
        By default, True.
    
    Returns
    -------
    tuple[tuple[np.ndarray, list], tuple[np.ndarray, list]] or tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]
        Tuple of training and test sets of the form :math:`(X_train, y_train), (X_test, y_test)`.
    """

    download = not Path(os.path.join(path, "/cifar-100-python")).is_dir()
    train_set = torchvision.datasets.CIFAR100(root=path,
                                             train=True,
                                             download=download)
    test_set = torchvision.datasets.CIFAR100(root=path,
                                            train=False,
                                            download=download)
    if as_tensor:
        train_set = tensor(train_set.data).float().permute(0, 3, 1, 2) / 255.,\
                    tensor(train_set.targets)
        test_set = tensor(test_set.data).float().permute(0, 3, 1, 2) / 255.,\
                   tensor(test_set.targets)
    else:
        train_set = train_set.data, train_set.targets
        test_set = test_set.data, test_set.targets

    return train_set, test_set

def get_FashionMNIST(path: str="./data",
                     as_tensor: bool=True) -> Union[Tuple[Tuple[np.ndarray, list], Tuple[np.ndarray, list]],
                                                          Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]]:
    """Returns the FashionMNIST dataset.

    The method downloads the dataset if it is not already present in `path`.

    Parameters
    ----------
    path : str, default="./data"
        Path to save the dataset, by default "./data".
    as_tensor : bool, default=True
        If True, the dataset is returned as a tuple of pytorch tensors.
        Otherwise, the dataset is returned as a tuple of numpy arrays.
        By default, True.

    Returns
    -------
    Tuple[Tuple[np.ndarray, list], Tuple[np.ndarray, list]] or Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]
        Tuple of training and test sets of the form 
        :math:`(X_\text{train}, y_\text{train}), (X_\text{test}, y_\text{test})`.
    """

    download = not Path(os.path.join(path, "/FashionMNIST/raw/")).is_dir()
    train_set = torchvision.datasets.FashionMNIST(root=path,
                                                  train=True,
                                                  download=download)
    test_set = torchvision.datasets.FashionMNIST(root=path,
                                                 train=False,
                                                 download=download)
    if as_tensor:
        train_set = train_set.data / 255., train_set.targets
        test_set = test_set.data / 255., test_set.targets
    else:
        train_set = train_set.data.numpy() / 255., train_set.targets.numpy()
        test_set = test_set.data.numpy() / 255., test_set.targets.numpy()

    return train_set, test_set

#UNDOCUMENTED
def get_FEMNIST(path: str="./data") -> Tuple[Tuple[torch.Tensor, torch.Tensor, List[int]], \
                                             Tuple[torch.Tensor, torch.Tensor, List[int]]]:
    url = 'https://raw.githubusercontent.com/tao-shen/FEMNIST_pytorch/master/femnist.tar.gz'
    te_name, tr_name = download_and_untar(url, path)
    Xtr, ytr, ids_tr = torch.load(os.path.join(path, tr_name))
    Xte, yte, ids_te = torch.load(os.path.join(path, te_name))
    tr_assignment = []
    te_assignment = []
    sum_tr = sum_te = 0
    for i in range(len(ids_tr)):
        ntr, nte = ids_tr[i], ids_te[i]
        tr_assignment.append(list(range(sum_tr, sum_tr + ntr)))
        te_assignment.append(list(range(sum_te, sum_te + nte)))
    return (Xtr, ytr, tr_assignment), (Xte, yte, te_assignment)

from numpy.random import dirichlet

class NonIIDCustomDataDispatcher(DataDispatcher):
    def assign(self, seed: int = 42, alpha: float = 0.75) -> None:
        """
        Assigns data to clients in a non-IID manner using Dirichlet distribution.

        Parameters
        ----------
        seed : int, default=42
            The seed for the random number generator.
        alpha : float, default=0.5
            The parameter for the Dirichlet distribution.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        n_clients = self.n
        y = self.data_handler.ytr
        labels = torch.unique(y).numpy()
        n_classes = len(labels)

        # Dirichlet distribution
        proportions = dirichlet([alpha] * n_clients, n_classes)

        # Shuffle and assign samples
        self.tr_assignments = [[] for _ in range(n_clients)]
        for k in range(n_classes):
            idx_k = np.where(y == k)[0]
            np.random.shuffle(idx_k)
            split_idx = (np.cumsum(proportions[k]) * len(idx_k)).astype(int)[:-1]
            split_data = np.split(idx_k, split_idx)
            for client_idx, data in enumerate(split_data):
                self.tr_assignments[client_idx].extend(data)

        if self.eval_on_user:
            y_eval = self.data_handler.yte
            proportions_eval = dirichlet([alpha] * n_clients, n_classes)
            self.te_assignments = [[] for _ in range(n_clients)]
            for k in range(n_classes):
                idx_k = np.where(y_eval == k)[0]
                np.random.shuffle(idx_k)
                split_idx = (np.cumsum(proportions_eval[k]) * len(idx_k)).astype(int)[:-1]
                split_data = np.split(idx_k, split_idx)
                for client_idx, data in enumerate(split_data):
                    self.te_assignments[client_idx].extend(data)
        else:
            self.te_assignments = [[] for _ in range(n_clients)]

    def print_distribution(self) -> None:
        """
        Prints the distribution of training and evaluation data among clients.
        """
        def count_classes(assignments):
            y = self.data_handler.ytr
            counts = [np.bincount(y[client_data], minlength=len(torch.unique(y))) for client_data in assignments]
            return counts

        train_counts = count_classes(self.tr_assignments)
        eval_counts = count_classes(self.te_assignments)

        print("Training Data Distribution:")
        for i, counts in enumerate(train_counts):
            print(f"Client {i}: {counts}")

        if self.eval_on_user:
            print("\nEvaluation Data Distribution:")
            for i, counts in enumerate(eval_counts):
                print(f"Client {i}: {counts}")

# Define the classes and contexts
CLASSES = {
    "car": 0, "flower": 1, "penguin": 2, "camel": 3, "chair": 4,
    "monitor": 5, "truck": 6, "wheat": 7, "sword": 8, "seal": 9,
    "lion": 10, "fish": 11, "dolphin": 12, "lifeboat": 13, "tank": 14,
    "fishing rod": 15, "sunflower": 16, "cow": 17, "bird": 18, "airplane": 19,
    "shark": 20, "rabbit": 21, "snake": 22, "hot air balloon": 23, "hat": 24,
    "spider": 25, "motorcycle": 26, "tortoise": 27, "dog": 28, "elephant": 29,
    "chicken": 30, "bee": 31, "gun": 32, "fox": 33, "phone": 34, "bus": 35,
    "cat": 36, "sailboat": 37, "cactus": 38, "pumpkin": 39, "train": 40,
    "dragonfly": 41, "ship": 42, "helicopter": 43, "bicycle": 44, "squirrel": 45,
    "bear": 46, "mailbox": 47, "horse": 48, "banana": 49, "mushroom": 50,
    "cauliflower": 51, "whale": 52, "camera": 53, "beetle": 54, "monkey": 55,
    "lemon": 56, "pepper": 57, "sheep": 58, "umbrella": 59
}

# Define the contexts
CONTEXTS = ["autumn", "dim", "grass", "outdoor", "rock", "water"]

def load_images_from_folder(folder_path, class_mapping, img_size=(224, 224), fraction=1.0):
    images = []
    labels = []
    contexts = []
    total_files = sum([len(files) for r, d, files in os.walk(folder_path)])
    processed_files = 0

    for root, dirs, files in os.walk(folder_path):
        for class_name in dirs:
            class_path = os.path.join(root, class_name)
            if class_name.lower() in class_mapping:
                class_label = class_mapping[class_name.lower()]
                print(f"Processing class: {class_name} with label: {class_label}")
                class_files = os.listdir(class_path)
                num_files_to_load = int(len(class_files) * fraction)
                selected_files = np.random.choice(class_files, num_files_to_load, replace=False)
                for filename in selected_files:
                    img_path = os.path.join(class_path, filename)
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize(img_size)  # Resize image
                        images.append(np.array(img))
                        labels.append(class_label)
                        contexts.append("")  # No context for test data
                    except Exception as e:
                        print(f"Error loading image: {img_path}, {e}")
                    processed_files += 1
                    if processed_files % 100 == 0:
                        print(f"Processed {processed_files}/{total_files} files.")

    return images, labels, contexts

def get_NICO(path: str = "./data", as_tensor: bool = True, train_fraction: float = 1.0, test_fraction: float = 1.0) -> Union[Tuple[Tuple[np.ndarray, list, list], Tuple[np.ndarray, list]], Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]:
    """Returns the NICO++ dataset.

    Parameters
    ----------
    path : str, default="./data"
        Path to the root folder of NICO++ dataset.
    as_tensor : bool, default=True
        If True, returns data as PyTorch tensors, otherwise as numpy arrays.
    train_fraction : float, default=1.0
        Fraction of training data to load (1.0 means all data).
    test_fraction : float, default=1.0
        Fraction of test data to load (1.0 means all data).

    Returns
    -------
    Tuple[Tuple[np.ndarray, list, list], Tuple[np.ndarray, list]] or Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
        Tuple of training and test sets: (X_train, y_train, c_train), (X_test, y_test).
        Here, c_train denotes the context for each image in the training set.
    """
    # Paths to training and test data
    train_folder = os.path.join(path, "NICO++", "track_1", "track_1", "public_dg_0416", "train")
    test_folder = os.path.join(path, "NICO++", "track_2", "track_2", "public_ood_0412_nodomainlabel", "train")

    # Load training data
    print("Loading training data...")
    X_train, y_train, c_train = load_images_from_folder(train_folder, CLASSES, fraction=train_fraction)

    # Load test data (without contexts since they are not provided in the test set)
    print("Loading test data...")
    X_test, y_test, _ = load_images_from_folder(test_folder, CLASSES, fraction=test_fraction)

    # Convert lists to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    if as_tensor:
        # Convert numpy arrays to PyTorch tensors
        X_train = torch.tensor(X_train).float().permute(0, 3, 1, 2) / 255.
        y_train = torch.tensor(y_train)
        c_train = torch.tensor([CONTEXTS.index(c) if c in CONTEXTS else -1 for c in c_train])
        
        X_test = torch.tensor(X_test).float().permute(0, 3, 1, 2) / 255.
        y_test = torch.tensor(y_test)
    else:
        # Leave as numpy arrays
        c_train = [CONTEXTS.index(c) if c in CONTEXTS else -1 for c in c_train]

    return (X_train, y_train, c_train), (X_test, y_test)


class OLDCustomDataDispatcher2:
    def __init__(self, data_handler, n: int, eval_on_user=True, auto_assign=True):
        self.data_handler = data_handler
        self.n = n
        self.eval_on_user = eval_on_user
        self.auto_assign = auto_assign
        self.tr_assignments = []
        self.te_assignments = []

    def assign(self, seed: int = 42) -> None:
        self.tr_assignments = [[] for _ in range(self.n)]
        self.te_assignments = [[] for _ in range(self.n)]

        n_ex = self.data_handler.size()
        ex_x_user = n_ex // self.n  # Ensure equal partitioning

        for idx in range(self.n):
            start_index = idx * ex_x_user
            end_index = start_index + ex_x_user
            self.tr_assignments[idx] = list(range(start_index, min(end_index, n_ex)))

        if self.eval_on_user:
            n_eval_ex = self.data_handler.eval_size()
            eval_ex_x_user = n_eval_ex // self.n
            for idx in range(self.n):
                start_index = idx * eval_ex_x_user
                end_index = start_index + eval_ex_x_user
                self.te_assignments[idx] = list(range(start_index, min(end_index, n_eval_ex)))

from sklearn.model_selection import train_test_split

class ClassificationDataHandler2:
    def __init__(self,
                 X: Union[np.ndarray, torch.Tensor],
                 y: Union[np.ndarray, torch.Tensor],
                 X_te: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 y_te: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 test_size: float = 0.2,
                 seed: int = 42):
        """Handler for classification data.

        The handler provides methods to access the data and to split it into
        training and evaluation sets.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            The data set examples matrix. If ``X_te`` is not None, then the
            data set is assumed to be already split into training and evaluation set
            (``test_size`` will be ignored).
        y : np.ndarray or torch.Tensor
            The data set labels.
        X_te : np.ndarray or torch.Tensor, default=None
            The evaluation set examples matrix.
        y_te : np.ndarray or torch.Tensor, default=None
            The evaluation set labels.
        test_size : float, default=0.2
            The size of the evaluation set as a fraction of the data set.
        seed : int, default=42
            The seed used to split the data set into training and evaluation set.
        """

        assert 0 <= test_size < 1
        assert isinstance(X, (torch.Tensor, np.ndarray))

        if test_size > 0 and (X_te is None or y_te is None):
            if isinstance(X, torch.Tensor):
                n = X.shape[0]
                te = round(n * test_size)
                torch.manual_seed(seed)
                perm = torch.randperm(n)
                split = perm[:n - te], perm[n - te:]
                self.Xtr, self.ytr = X[split[0], :], y[split[0]]
                self.Xte, self.yte = X[split[1], :], y[split[1]]
            else:
                self.Xtr, self.Xte, self.ytr, self.yte = train_test_split(X, y,
                                                                          test_size=test_size,
                                                                          random_state=seed,
                                                                          shuffle=True)
        else:
            self.Xtr, self.ytr = X, y
            self.Xte, self.yte = X_te, y_te

        self.n_classes = len(np.unique(self.ytr))

    def __getitem__(self, idx: Union[int, List[int]]) -> \
            Union[Tuple[np.ndarray, Union[int, List[int]]], Tuple[torch.Tensor, Union[int, List[int]]]]:
        return self.Xtr[idx, :], self.ytr[idx]

    def at(self,
           idx: Union[int, List[int]],
           eval_set=False) -> Tuple[Union[np.ndarray, torch.Tensor], int]:
        """Get the data set example and label at the given index or list of indices.

        Parameters
        ----------
        idx : int or list of int
            The index or list of indices of the data set examples to get.
        eval_set : bool, default=False
            If True, the data set example and label are retrieved from the evaluation set.
            Otherwise, they are retrieved from the training set.

        Returns
        -------
        X : np.ndarray or torch.Tensor
            The data set example.
        y : int
            The data set label.
        """

        if eval_set:
            if isinstance(self.Xte, torch.Tensor):
                return self.Xte[idx, :], self.yte[idx]
            else:
                return self.Xte[idx, :], self.yte[idx]
        else:
            return self[idx]

    def size(self, dim: int = 0) -> int:
        return self.Xtr.shape[dim]

    def get_train_set(self) -> Tuple[Any, Any]:
        return self.Xtr, self.ytr

    def get_eval_set(self) -> Tuple[Any, Any]:
        return self.Xte, self.yte

    def eval_size(self) -> int:
        return self.Xte.shape[0] if self.Xte is not None else 0

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        res = f"{self.__class__.__name__}(size_tr={self.size()}, size_te={self.eval_size()}"
        res += f", n_feats={self.size(1)}, n_classes={self.n_classes})"
        return res