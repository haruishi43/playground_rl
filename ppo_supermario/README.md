# [PYTORCH] Proximal Policy Optimization (PPO) for playing Super Mario Bros

## Introduction

Here is my python source code for training an agent to play super mario bros. By using Proximal Policy Optimization (PPO) algorithm introduced in the paper **Proximal Policy Optimization Algorithms** [paper](https://arxiv.org/abs/1707.06347).

<p align="left">
  <img src=".readme/demo/video-1-2.gif" width="200">
  <img src=".readme/demo/video-1-3.gif" width="200">
  <img src=".readme/demo/video-1-4.gif" width="200"><br/>
  <img src=".readme/demo/video-2-1.gif" width="200">
  <img src=".readme/demo/video-2-2.gif" width="200">
  <img src=".readme/demo/video-2-3.gif" width="200">
  <img src=".readme/demo/video-2-4.gif" width="200"><br/>
  <img src=".readme/demo/video-3-1.gif" width="200">
  <img src=".readme/demo/video-3-2.gif" width="200">
  <img src=".readme/demo/video-3-3.gif" width="200">
  <img src=".readme/demo/video-3-4.gif" width="200"><br/>
  <img src=".readme/demo/video-4-1.gif" width="200">
  <img src=".readme/demo/video-4-2.gif" width="200">
  <img src=".readme/demo/video-4-3.gif" width="200">
  <img src=".readme/demo/video-4-4.gif" width="200"><br/>
  <img src=".readme/demo/video-5-1.gif" width="200">
  <img src=".readme/demo/video-5-2.gif" width="200">
  <img src=".readme/demo/video-5-3.gif" width="200">
  <img src=".readme/demo/video-5-4.gif" width="200"><br/>
  <img src=".readme/demo/video-6-1.gif" width="200">
  <img src=".readme/demo/video-6-2.gif" width="200">
  <img src=".readme/demo/video-6-3.gif" width="200">
  <img src=".readme/demo/video-6-4.gif" width="200"><br/>
  <img src=".readme/demo/video-7-1.gif" width="200">
  <img src=".readme/demo/video-7-2.gif" width="200">
  <img src=".readme/demo/video-7-3.gif" width="200">
  <img src=".readme/demo/video-7-4.gif" width="200"><br/>
  <img src=".readme/demo/video-8-1.gif" width="200">
  <img src=".readme/demo/video-8-2.gif" width="200">
  <img src=".readme/demo/video-8-3.gif" width="200"><br/>
  <img src=".readme/demo/video-1-1.gif" width="200">
  <i>Sample results</i>
</p>

## Running the code

* **Train your model** by running `python train.py`. For example: `python train.py --world 5 --stage 2 --lr 1e-4`
* **Test your trained model** by running `python test.py`. For example: `python test.py --world 5 --stage 2`

**Note**: If you got stuck at any level, try training again with different **learning rates**. You could conquer 31/32 levels like what I did, by changing only **learning rate**. Normally I set **learning rate** as **1e-3**, **1e-4** or **1e-5**. However, there are some difficult levels, including level **1-3**, in which I finally trained successfully with **learning rate** of **7e-5** after failed for 70 times.

## Why there is still level 8-4 missing?

In world 4-4, 7-4 and 8-4, map consists of puzzles where the player must choose the correct the path in order to move forward. If you choose a wrong path, you have to go through path you visited again. With some hardcore setting for the environment, the first 2 levels are solved. But the last level has not been solved yet.


## Dependencies:

- Doesn't run on newer versions of `gym` (only runs on `gym==0.19.0`)
  - `gym-super-mario-bros` reward function is not `float` which raises errors