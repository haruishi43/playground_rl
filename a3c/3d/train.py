from __future__ import division
from setproctitle import setproctitle as ptitle
import os
import torch
import torch.optim as optim
from utils import ensure_shared_grads
from model import A3Clstm
from agent import Agent
import time

def train(rank, args, shared_model, optimizer):
    ptitle('Training Agent: {}'.format(rank))
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    
    device = 'cpu'
    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        device = 'gpu:'+str(gpu_id)
        torch.cuda.manual_seed(args.seed + rank)

    # Initialize Environment
    if args.use_lab:
        from env_lab import EnvLab
        env = EnvLab(80, 80, 60, "seekavoid_arena_01", args.seed + rank)
    else:
        from env_vizdoom import EnvVizDoom
        env = EnvVizDoom(os.path.join(args.vizdoom_path,args.vizdoom_scenarios), args.seed + rank)

    if optimizer is None:
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
    
    # Initialize Player
    player = Agent(None, env, args, None)  # init agent
    player.gpu_id = gpu_id
    player.model = A3Clstm(player.env.channels,
                           player.env.num_actions)
    player.env.reset()
    state = player.env.observation()
    player.state = torch.from_numpy(state).float()

    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.state = player.state.cuda()
            player.model = player.model.cuda()
    
    player.model.train()

    while True:
        # Sync with the shared model:
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                player.model.load_state_dict(shared_model.state_dict())
        else:
            player.model.load_state_dict(shared_model.state_dict())

        # first, agent is in a done state
        player.update_lstm()

        # play environment for number of steps
        for step in range(args.num_steps):
            player.eps_len += 1
            player.action_train()
            # if player finishes, 
            if player.done:
                player.eps_len = 0
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        player.state = player.state.cuda()
                break

        R = torch.zeros(1, 1).data
        if not player.done:
            value, _, _ = player.model((player.state.unsqueeze(0),
                                        (player.hx, player.cx)))
            R = value.data

        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                R = R.cuda()

        player.values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                # policy_loss = policy_loss.cuda()
                # value_loss = value_loss.cuda()
                gae = gae.cuda()
        
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
                gae - 0.0001 * player.entropies[i]
                        
            # print(player.entropies[i])

        # torch.nn.utils.clip_grad_norm_(player.model.parameters(), 40)  # clip if gradient exceed
        player.model.zero_grad()
        # time.sleep(1)
        loss = (policy_loss + 0.5 * value_loss)
        loss.backward()
        ensure_shared_grads(player.model, shared_model, gpu=gpu_id >= 0)
        optimizer.step()
        
        loss = None
        player.clear_actions()
