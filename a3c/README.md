# A3C-Pytorch

## What's this repository?

- Implementation of Asynchronous Advantage Actor-Critic (A3C) on Pytorch on variety of environments:
    - OpenAI Gym
    - VizDoom
    - DeepMind Lab
- Base directory has the A3C codes for OpenAI Gym environments
    - Most of the codes were referenced from [@dgriff777](https://github.com/dgriff777/rl_a3c_pytorch) and [@ikostrikov](https://github.com/ikostrikov/pytorch-a3c)
    - Working!
- The directory `3D` contains codes that works with 3D environments (VizDoom and DeepMind Lab)
    - Could not train... Still work in progress

## TODO

- [ ] Debug the 3D codes and understand why it's failing


## Minimum Requirements:

- Python 3.6
- Pytorch 0.4.0

For 3D (WIP):

- OpenCV
- VizDoom
- DeepMind Lab


## Usage

### OpenAI Gym Environments (Base Directory)

- Training:

```bash
python main.py --env Pong-v0 --workers 16
```

You can change the environment to any of the working environments in OpenAI Gym.

```bash
python main.py --env Pong-v0 --workers 16 --gpu-ids 0 1 --amsgrad True
```

- Testing:

```bash
python gym_eval.py --env Pong-v0 --num-episodes 100
```

### 3D Environments 

**Make sure you are in the `3d` directory**

- Training:

```bash
python main.py 
```

`--use-lab` can be passed to train in DeepMind Lab environments.

For more information check out the `argparse` lines in `main.py`.

- Testing:

```bash
python main.py --test --load
```
