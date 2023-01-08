from __future__ import print_function

import os
os.environ["OMP_NUM_THREADS"] = "1"  # make sure numpy doesn't use more than 1 thread

import sys
import time
import logging

import numpy as np
import cv2
# import tensorflow as tf
# import threading
import torch
import torch.multiprocessing as mp
from shared_optim import SharedRMSprop, SharedAdam

from arguments import create_args
from utils import make_dir, setup_logger
from model import A3Clstm
from agent import Agent
from train import train
from test import test


if __name__=='__main__':
    
    parser = create_args()
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    
    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
        mp.set_start_method('spawn')
    
    if args.use_lab:
        from env_lab import EnvLab
        model_path = "model_lab_a3c/"
        env = EnvLab(80, 80, 60, "seekavoid_arena_01")
    else:
        from env_vizdoom import EnvVizDoom
        model_path = "model_vizdoom_a3c/"
        env = EnvVizDoom(os.path.join(args.vizdoom_path, args.vizdoom_scenarios))

    make_dir(model_path)
    model_name = model_path + args.model_name  # use this to save model
    
    shared_model = A3Clstm(env.channels, env.num_actions)
    if args.load:
        saved_state = torch.load(
            '{}.dat'.format(model_name),
            map_location=lambda storage, loc: storage)
        shared_model.load_state_dict(saved_state)
    shared_model.share_memory() 

    if args.shared_optimizer:
        if args.optimizer == 'RMSprop':
            optimizer = SharedRMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = SharedAdam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
        optimizer.share_memory()
    else:
        print('not using optimizer may cause issues')
        optimizer = None

    if not args.test:
        # start multiprocessing:

        processes = []

        p = mp.Process(target=test, args=(args, shared_model, model_name))
        p.start()
        processes.append(p)
        time.sleep(0.1)

        for rank in range(0, args.workers):
            p = mp.Process(
                target=train, args=(rank, args, shared_model, optimizer))
            p.start()
            processes.append(p)
            time.sleep(0.1)
        for p in processes:
            time.sleep(0.1)
            p.join()

    else:
        gpu_ids = args.gpu_ids
        gpu_id = gpu_ids[0]

        torch.manual_seed(args.seed)
        if gpu_id >= 0:
            torch.cuda.manual_seed(args.seed)

        saved_state = torch.load(
            '{}.dat'.format(model_name),
            map_location=lambda storage, loc: storage)

        log = {}
        make_dir(args.log_dir)
        setup_logger('{}_mon'.format(args.env), r'{0}{1}_mon'.format(
            args.log_dir, args.env))
        log['{}_mon'.format(args.env)] = logging.getLogger('{}_mon'.format(
            args.env))

        d_args = vars(args)
        for k in d_args.keys():
            log['{}_mon'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

        if args.use_lab:
            from env_lab import EnvLab
            env = EnvLab(80, 80, 60, "seekavoid_arena_01")
        else:
            from env_vizdoom import EnvVizDoom
            env = EnvVizDoom(os.path.join(args.vizdoom_path,args.vizdoom_scenarios))
        
        # Initialize player
        player = Agent(None, env, args, None)
        player.model = A3Clstm(player.env.channels,
                            player.env.num_actions)
        player.gpu_id = gpu_id
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                player.model = player.model.cuda()

        # Load saved state
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                player.model.load_state_dict(saved_state)
        else:
            player.model.load_state_dict(saved_state)
        player.model.eval()

        num_tests = 0
        start_time = time.time()
        reward_total_sum = 0

        for i_episode in range(args.num_test_episodes):
            # initialize state
            player.restart_env()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()
            
            player.eps_len = 0
            reward_sum = 0
            while True:
                player.eps_len += 1
                if args.render:
                    if i_episode % args.render_freq == 0:
                        frame = player.env.raw_observation()
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        cv2.imshow("render", frame)
                        cv2.waitKey(30)
                
                player.action_test()
                reward_sum += player.reward

                if player.done:               
                    num_tests += 1
                    reward_total_sum += reward_sum
                    reward_mean = reward_total_sum / num_tests
                    log['{}_mon'.format(args.env)].info(
                        "Time {0}, episode reward {1}, episode length {2}, reward mean {3:.4f}".
                        format(
                            time.strftime("%Hh %Mm %Ss",
                                        time.gmtime(time.time() - start_time)),
                            reward_sum, player.eps_len, reward_mean))
                    break
