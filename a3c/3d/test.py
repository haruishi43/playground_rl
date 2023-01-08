from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
from utils import setup_logger, make_dir
from model import A3Clstm
from agent import Agent
import os
import time
import logging


def test(args, shared_model, model_path):
    ptitle('Test Agent')
    gpu_id = args.gpu_ids[-1]
    log = {}
    make_dir(args.log_dir)
    setup_logger('{}'.format(args.env), r'{0}{1}'.format(
        args.log_dir, args.env))
    log['{}'.format(args.env)] = logging.getLogger('{}'.format(
        args.env))
    d_args = vars(args)
    for k in d_args.keys():
        log['{}'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

    torch.manual_seed(args.seed)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed)

    # Initialize Environment
    if args.use_lab:
        from env_lab import EnvLab
        env = EnvLab(80, 80, 60, "seekavoid_arena_01")
    else:
        from env_vizdoom import EnvVizDoom
        env = EnvVizDoom(os.path.join(args.vizdoom_path,args.vizdoom_scenarios))

    # Initialize Player:
    player = Agent(None, env, args, None)
    player.gpu_id = gpu_id
    player.model = A3Clstm(player.env.channels,
                           player.env.num_actions)
    player.env.reset()
    state = player.env.observation()
    player.state = torch.from_numpy(state).float()

    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.model = player.model.cuda()
            player.state = player.state.cuda()

    reward_sum = 0
    start_time = time.time()
    num_tests = 0
    reward_total_sum = 0
    max_score = -1000  # change this parameter later
    
    while True:
        player.eps_len += 1

        # sync with shared model
        if player.done:
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.model.load_state_dict(shared_model.state_dict())
            else:
                player.model.load_state_dict(shared_model.state_dict())
            player.model.eval()
        
        # step through according to current model
        player.action_test()
        # update reward
        reward_sum += player.reward
        # change player's done state
        player.done = player.done or player.eps_len >= args.max_episode_length
        
        if player.done:
            num_tests += 1
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
            log['{}'.format(args.env)].info(
                "Time {0}, episode reward {1}, episode length {2}, reward mean {3:.4f}".
                format(
                    time.strftime("%Hh %Mm %Ss",
                                    time.gmtime(time.time() - start_time)),
                    reward_sum, player.eps_len, reward_mean))

            # print(list(player.model.parameters())[0])
            if args.save_max and reward_sum >= max_score:
                max_score = reward_sum
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        state_to_save = player.model.state_dict()
                        torch.save(state_to_save, '{}.dat'.format(
                            model_path))
                else:
                    state_to_save = player.model.state_dict()
                    torch.save(state_to_save, '{}.dat'.format(
                        model_path))

            reward_sum = 0
            player.eps_len = 0
            time.sleep(10)
            player.restart_env()
