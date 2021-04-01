# SBR_Experiment

The  purpose of this experiment is to find unlabelled security bugs among thousands of bug reports via **active learning** and **ranking**. This experiment simulates the review process under the specified SBR(**S**ecurity **B**ugs **R**eport) recall rate.

## Structure

- `encoder/`: Different method to transform words into vectors
- `model/`: Training different models and predict the probability of SBR
- `Experiment.py`: Main process
- `main.py`: Runner

## Configure

All configurations are in [`Experiment`](./Experiment.py) class.

