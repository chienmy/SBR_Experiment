# SBR_Experiment

The  purpose of this experiment is to find unlabelled security bugs among thousands of bug reports via **active learning** and **ranking**. This experiment simulates the review process under the specified SBR(**S**ecurity **B**ugs **R**eport) recall rate.

## Structure

- `encoder/`: Different method to transform words into vectors
- `model/`: Training different models and predict the probability of SBR
- `experiment/`: Experiments to run
- `RQ*.py`: Runner

## Configure

All configurations are in [`BaseExperiment`](./experiment/BaseExperiment.py) class, and we use [`ExperimentFactory`](./experiment/ExperimentFactory.py) to build its instance from a list of configurations dictionary.

## Running

```bash
pip install numpy pandas sklearn alive_progress
python3 RQ*.py
```



