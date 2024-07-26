# Socially efficient mechanism on the minimum budget

## Scope

This repository is the official implementation of [Socially efficient mechanism on the minimum budget]().


## Requirements

Here, we set up the environment with poetry.  To install requirements:

```setup
poetry install
```

## Running experiments

The figures in the body of the paper can be reproduced by running the following commands:
```
poetry run python neurips2024.py --hist --fix n --n_max 16 --reps 1000
poetry run python neurips2024.py --vs_n --reps 1000
poetry run python neurips2024.py --vs_m --n_max 16 --reps 1000
poetry run python neurips2024.py --vs_d --n_max 16 --reps 1000
```

The figures in the appendix can be reproduced analogously by varying the values of the parameters that can be specified with arguments.  The details can be found in `README-function.md` and `neurips2024.py`.


## Authors

- Author: Hirota Kinoshita, Takayuki Osogami

