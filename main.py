from typing import List, Dict, Union
from sklearn.datasets import load_iris, load_wine
from src.dataset import Dataset
from src.runner import Runner
import pandas as pd

RUN_TIMES = 10

datasets: List[Dataset] = [
    Dataset.from_bunch(load_iris(), 'iris'),
    Dataset.from_bunch(load_wine(), 'wine'),
]

dataframe_rows: List[Dict[str, Union[str, float]]] = []

for dataset in datasets:
    print(f'Running GA on dataset {dataset.name}')
    runner = Runner(dataset, f'clusters/{dataset.name}.png', RUN_TIMES)

    runner.run()
    runner.plot()
    dataframe_rows.append(runner.get_dataframe_row())

dataframe = pd.DataFrame(dataframe_rows)

print(dataframe)
