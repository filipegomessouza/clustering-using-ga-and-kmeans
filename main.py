from sklearn.datasets import load_iris
from sklearn.utils import Bunch
from src.runner import Runner

iris_dataset: Bunch = load_iris()

runner = Runner(
    dataset=iris_dataset.data,
    number_of_groups=3,
    filename='test.png',
    class_names=iris_dataset.target_names,
    run_times=10,
)

runner.run()
runner.plot()
print(runner.stats())
