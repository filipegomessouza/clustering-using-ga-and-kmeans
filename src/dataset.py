from typing import List
import numpy as np
from sklearn.utils import Bunch

class Dataset:
    def __init__(self, data: np.ndarray, name: str, class_names: List[str]):
        self.data = data
        self.name = name
        self.class_names = class_names
        self.number_of_groups = len(class_names)

    @classmethod
    def from_bunch(cls, bunch: Bunch, name: str) -> 'Dataset':
        return cls(bunch.data, name, bunch.target_names)
