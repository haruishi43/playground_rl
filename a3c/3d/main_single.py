#########################################################################
# A2C?
#########################################################################

import os
os.environ["OMP_NUM_THREADS"] = "1"  # make sure numpy doesn't use more than 1 thread

import sys
import time
import logging

import numpy as np
import cv2
import torch
import torch.optim as optim
from shared_optim import SharedRMSprop, SharedAdam

from arguments import create_args
from utils import make_dir, setup_logger
from model import A3Clstm
from agent_debug import Agent

from logger import Logger

# from tensorboardX import SummaryWriter


if __name__=='__main__':
    #########################################################################
    # PREP
    #########################################################################

    parser = create_args()
    args = parser.parse_args()

    device = 'cpu'
    torch.manual_seed(args.seed)
    gpu_id = -1
    if args.gpu_ids != -1:
        gpu_id = 0
        device = 'cuda:' + str(gpu_id)  # set device id to somthing different 
        torch.cuda.manual_seed(args.seed)

    if args.use_lab:
        from env_lab import EnvLab
        model_path = "model_lab_a3c/"
        env = EnvLab(80, 80, 60, "seekavoid_arena_01", args.seed)
    else:
        from env_vizdoom import EnvVizDoom
        model_path = "model_vizdoom_a3c/"
        env = EnvVizDoom(os.path.join(args.vizdoom_path, args.vizdoom_scenarios), args.seed)
    
    make_dir(model_path)
    model_name = model_path + args.model_name  # use this to save model

    model = A3Clstm(env.channels, env.num_actions)
    if args.load:
        saved_state = torch.load(
            '{}.dat'.format(model_name),
            map_location=lambda storage, loc: storage)
        model.load_state_dict(saved_state)

    # if args.optimizer == 'RMSprop':
    #     optimizer = SharedRMSprop(model.parameters(), lr=args.lr)
    # elif args.optimizer == 'Adam':
    #     optimizer = SharedAdam(
    #         model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
    if args.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
    else:
        print('optimizer not set')

    #########################################################################
    # Algorithm
    #########################################################################

    max_train = 1000

    player = Agent(model, env, args, None)
    player.gpu_id = gpu_id
    player.device = device
    player.env.reset()
    state = player.env.observation()
    player.state = torch.from_numpy(state).float()

    player.state = player.state.to(device)
    player.model = player.model.to(device)


    ## Params 
    entropy_loss_coef = 00.1
    start_time = time.time()
    reward_sum = 0
    num_tests = 0
    reward_total_sum = 0
    max_score = -1000  # change this parameter later

    log = {}
    make_dir(args.log_dir)
    setup_logger('{}'.format(args.env), r'{0}{1}'.format(
        args.log_dir, args.env))
    log['{}'.format(args.env)] = logging.getLogger('{}'.format(
        args.env))
    d_args = vars(args)
    for k in d_args.keys():
        log['{}'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

    ## logger
    # my_writer = SummaryWriter(log_dir='tensorboard_log')
    logger = Logger('./logs')


    ## Loop
    counter = 0
    while True:

        ######################################################################
        # Train
        ######################################################################
        num_train = 0
        player.model.train()
        while num_train < max_train:
            # first, agent is in a done state
            player.update_lstm()

             # play environment for number of steps
            for step in range(args.num_steps):
                player.eps_len += 1
                player.action_train()
                # if player finishes, 
                if player.done:
                    player.eps_len = 0
                    player.state = player.state.to(device)
                    break

            ## Updates:

            R = torch.zeros(1, 1).data
            if not player.done:
                value, _, _ = player.model((player.state.unsqueeze(0),
                                            (player.hx, player.cx)))
                R = value.data

            R = R.to(device)

            player.values.append(R)
            policy_loss = 0
            value_loss = 0
            gae = torch.zeros(1, 1).to(device)

            for i in reversed(range(len(player.rewards))):
                R = args.gamma * R + player.rewards[i]
                advantage = R - player.values[i]
                value_loss = value_loss + 0.5 * advantage.pow(2)

                # TD
                delta_t = player.rewards[i] + args.gamma * \
                    player.values[i + 1].data - player.values[i].data
                # Generalized Advantage Estimataion
                gae = gae * args.gamma * args.tau + delta_t

                # entropy loss (atari 0.01)
                policy_loss = policy_loss - \
                    player.log_probs[i] * \
                    gae - entropy_loss_coef * player.entropies[i]

            
            player.model.zero_grad()
            # time.sleep(1)
            loss = (policy_loss + 0.5 * value_loss)
            loss.backward()
            optimizer.step()

            loss = None
            player.clear_actions()

            num_train += 1
            counter += 1


        # print("finished training")
        ######################################################################
        # Test
        ######################################################################
        
        player.eps_len = 0
        reward_sum = 0
        player.model.eval()
        
        while True:
            player.eps_len += 1
            
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
                # my_writer.add_scalar('episode_reward', reward_sum, counter)
                # my_writer.add_scalar('episode_length', player.eps_len, counter)
                # my_writer.add_scalar('FPS', counter / (time.time() - start_time), counter)
                # print(list(player.model.parameters())[0])

                # logging
                # scalar
                info = {'Reward': reward_mean, 'Episode Length': player.eps_len}
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, counter)

                # histogram
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, value.data.cpu().numpy(), counter)
                    logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), counter)

                # # weight visualization
                # for tag, images in info.items():
                #     logger.image_summary(tag, images, counter)


                if args.save_max and reward_sum >= max_score:
                    max_score = reward_sum
                    if gpu_id >= 0:
                        with torch.cuda.device(gpu_id):
                            state_to_save = player.model.state_dict()
                            torch.save(state_to_save, '{}.dat'.format(
                                model_name))
                    else:
                        state_to_save = player.model.state_dict()
                        torch.save(state_to_save, '{}.dat'.format(
                            model_name))

                # time.sleep(10)
                player.restart_env()
                break

