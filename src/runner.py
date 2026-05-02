from typing import List, Optional, Dict, Union
import numpy as np
import pandas as pd
from src.dataset import Dataset
from src.genetic_algorithm import GeneticAlgorithm


class Runner:
    def __init__(self, dataset: Dataset, filename: str, run_times: int):
        self.dataset = dataset
        self.filename = filename
        self.run_times = run_times

        self._best_fitnesses: List[float] = []
        self._best_solutions: List[np.ndarray] = []
        self._genetic_algorithms: List[GeneticAlgorithm] = []
        self._best_idx: Optional[int] = None

    def run(self) -> None:
        for _ in range(self.run_times):
            genetic_algorithm = GeneticAlgorithm(self.dataset, self.dataset.number_of_groups)
            best_solution, best_fitness = genetic_algorithm.run()

            self._best_fitnesses.append(best_fitness[0])
            self._best_solutions.append(best_solution)
            self._genetic_algorithms.append(genetic_algorithm)

        self._best_idx = int(np.argmin(self._best_fitnesses))

    def plot(self) -> None:
        if self._best_idx is None:
            return

        self._genetic_algorithms[self._best_idx].plot(self.filename)

    def get_dataframe_row(self) -> Dict[str, Union[str, float]]:
        if self._best_idx is None:
            return {}

        return {
            'dataset': self.dataset.name,
            'mean_fitness': np.mean(self._best_fitnesses),
            'std_fitness': np.std(self._best_fitnesses),
            'min_fitness': self._best_fitnesses[self._best_idx],
        }
