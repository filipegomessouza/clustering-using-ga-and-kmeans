from typing import List, Optional
import numpy as np
import pandas as pd
from src.genetic_algorithm import GeneticAlgorithm


class Runner:
    def __init__(self, dataset: np.ndarray, number_of_groups: int, filename: str, class_names: List[str], run_times: int):
        self.dataset = dataset
        self.number_of_groups = number_of_groups
        self.filename = filename
        self.class_names = class_names
        self.run_times = run_times

        self._best_fitnesses: List[float] = []
        self._best_solutions: List[np.ndarray] = []
        self._genetic_algorithms: List[GeneticAlgorithm] = []
        self._best_idx: Optional[int] = None

    def run(self) -> None:
        for _ in range(self.run_times):
            genetic_algorithm = GeneticAlgorithm(self.dataset, number_of_groups=self.number_of_groups)
            best_solution, best_fitness = genetic_algorithm.run()

            self._best_fitnesses.append(best_fitness[0])
            self._best_solutions.append(best_solution)
            self._genetic_algorithms.append(genetic_algorithm)

        self._best_idx = int(np.argmin(self._best_fitnesses))

    def plot(self) -> None:
        if self._best_idx is None:
            return

        self._genetic_algorithms[self._best_idx].plot(self.filename, self.class_names)

    def stats(self) -> pd.DataFrame:
        if self._best_idx is None:
            return pd.DataFrame()

        return pd.DataFrame([{
            'mean_fitness': np.mean(self._best_fitnesses),
            'std_fitness': np.std(self._best_fitnesses),
            'min_fitness': self._best_fitnesses[self._best_idx],
        }])
