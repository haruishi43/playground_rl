from __future__ import division
import torch
import torch.nn.functional as F
from torch.autograd import Variable

# Workers:
# Each process trains an agent

class Agent(object):
    def __init__(self, model, env, args, state):
        self.model = model
        self.env = env  
        self.state = state
        self.hx = None
        self.cx = None
        self.eps_len = 0
        self.args = args
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.done = True
        self.reward = 0
        self.gpu_id = -1
        self.frame_repeat = 1
        self.hidden_state_num = 512

    def update_lstm(self):
        if self.done:
            if self.gpu_id >= 0:
                with torch.cuda.device(self.gpu_id):
                    self.cx = torch.zeros(1, self.hidden_state_num).cuda()
                    self.hx = torch.zeros(1, self.hidden_state_num).cuda()
            else:
                self.cx = torch.zeros(1, self.hidden_state_num)
                self.hx = torch.zeros(1, self.hidden_state_num)
        else:
            self.cx = Variable(self.cx.data)
            self.hx = Variable(self.hx.data)

    def action_train(self):
        state = self.env.observation()  # new state
        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()

        value, logit, (self.hx, self.cx) = self.model(
            (self.state.unsqueeze(0), (self.hx, self.cx)))
        prob = F.softmax(logit, dim=1)

        # print(prob)
        # print(prob.cpu())
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1)
        action = prob.multinomial(1).data  # sample action
        # print(action.cpu())
        log_prob = log_prob.gather(1, action)

        # store values:
        self.entropies.append(entropy)
        self.values.append(value)
        self.log_probs.append(log_prob)
        
        # print(action.cpu().numpy())
        action = action.cpu().numpy()[0][0]
        # print(action)
        reward, self.done = self.env.act(action, self.frame_repeat)
        if self.done:
            self.env.reset()

        self.reward = max(min(reward, 1), -1)
        # print(self.reward)
        self.rewards.append(self.reward)
    
    def restart_env(self):
        self.env.reset()

    def action_test(self):
        state = self.env.observation()
        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()

        with torch.no_grad():
            self.update_lstm()
            value, logit, (self.hx, self.cx) = self.model((
                self.state.unsqueeze(0), (self.hx, self.cx)))

        prob = F.softmax(logit, dim=1)
        action = prob.max(1)[1].data.cpu().numpy()  # get best action
        action = action[0]
        # action = prob.multinomial(1).data.cpu().numpy()[0][0]
        
        # print(action)

        self.reward, self.done = self.env.act(action, self.frame_repeat)
        if self.done:
            self.env.reset()


    def clear_actions(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        return self
