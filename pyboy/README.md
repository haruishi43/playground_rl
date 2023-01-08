# Introduction

Training time:
- Super Mario Land: 26 hours on a 4-core CPU 
- Kirby's Dream Land: 20 hours on CUDA GPU
- Kirby's Dream Land Bossfight: 12 hours on CUDA GPU

# Requirements
- Python3.7 on Windows
- Python3+ on Linux
- PyBoy (https://github.com/Baekalfen/PyBoy)
- SDL2
    - Ubuntu: __`sudo apt install libsdl2-dev`__
    - Fedora: __`sudo dnf install SDL2-devel`__
    - macOS: __`brew install sdl2`__
    - Windows: PyBoy guide https://github.com/Baekalfen/PyBoy/wiki/Installation#windows-10-64-bit Download link https://www.libsdl.org/download-2.0.php
- For package requirements see requirements.txt
- GameBoy ROM files for Super Mario Land or Kirby's Dream Land (place these in /games) 
  - Filename must be "Kirby_Dream_Land" and "Super_Mario_Land" respectively

# Run with Python
To run from source, first install dependencies:
- __`pip3 install -r requirements.txt`__

Then, run:
- __`python3 main.py`__

# Docker
Build command: __`docker build --tag pyboy-rl .`__ 

Once inside the image run __`python3 main.py`__ to start the program. The docker container only supports headless mode, and the game emulator ui will not show up.

# Based on
DDQN - https://arxiv.org/abs/1509.06461

PyTorch RL Super Mario Bros Example - https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
