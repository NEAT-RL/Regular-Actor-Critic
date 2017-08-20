# Regular Actor-Critic

RAC implementation for benchmarking. This code is based on implementation from
```
PENG, Y., CHEN, G., ZHANG, M., AND MEI, Y. Effective policy gradient search for reinforcement learning through neat based feature extraction.
```
# Setup
First install python (v 3.5 is recommended for OpenAI libraries) and install conda or [miniconda](https://conda.io/docs/install/quick.html). Miniconda is recommended as its smaller and useful if you have space requirements.

Set up a conda environment by following the Project dependencies [repo](https://github.com/NEAT-RL/Project-Dependencies)

Install the organisation's custom [gym](https://github.com/NEAT-RL/gym) [gym-ple](https://github.com/NEAT-RL/gym-ple) libraries into the conda environment. These libraries extend openai gym. If you are looking on running gym-ple games, then you will need to install the PyGame-Learning-Environment from [here](https://github.com/NEAT-RL/PyGame-Learning-Environment).

# Running algorithm
The main file is *AC.py*.
Experiment logs are written into log directory.

Run this file and if you want to save the std outputs for clarity use following command:

```
python AC.py [Environment Id] [Display] > output.log 2&>1
```

## Current Environments Configured
| Enviornment id  | Info  |
|---|---|
| CartPole-v0  | Standard implementation of gym cartpole   |
| CartPole-v1  | Standard implementation of gym cartpole   |
| MountainCar-v0  | Standard implementation of gym mountain car problem  |
| MountainCarExtraLong-v0  | Custom implementation of gym mountain car problem where the episode length is 999.  |


Use any of these environment id as an argument.

## Adding your own environment
The code uses the environment id to extract properties from the *properties* directory.
The arrangement of the properties directory is
```
properties/<Environment id>/Config
properties/<Environemt id>/neatem_properties.ini
```
Adding your environment involves creating a new directory of your environment id and add the two properties files used by the algorithm
