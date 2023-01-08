from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import norm_col_init, weights_init


class A3Clstm(torch.nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(A3Clstm, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)
        # self.conv5 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        # self.maxp5 = nn.MaxPool2d(2, 2)

        # self.conv1 = nn.Conv2d(num_inputs, 32, 8, stride=1, padding=2)
        # self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        # self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        lstm_size = 512
        num_outputs = num_actions
        self.lstm = nn.LSTMCell(1024, lstm_size)
        self.critic_linear = nn.Linear(lstm_size, 1)
        self.actor_linear = nn.Linear(lstm_size, num_outputs)

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)
        # self.conv5.weight.data.mul_(relu_gain)
        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        # register backward hook
        # for m in self.children():
        #     m.register_backward_hook(self.printgradnorm)

        self.train()


    def forward(self, inputs):
        x, (hx, cx) = inputs
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        x = F.relu(self.maxp1(self.conv1(x)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))
        # x = F.relu(self.maxp5(self.conv5(x)))

        x = x.view(x.size(0), -1)  # flatten
        # print(x.shape)
        hx, cx = self.lstm(x, (hx, cx))

        x = hx

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)

    def printgradnorm(self, module, grad_input, grad_output):
        print('Inside ' + module.__class__.__name__ + ' backward')
        print('Inside class:' + module.__class__.__name__)

        # print('grad_input: ', type(grad_input))
        # print('grad_input[0]: ', type(grad_input[0]))
        # print('grad_output: ', type(grad_output))
        # print('grad_output[0]: ', type(grad_output[0]))
        # print('grad_input size:', grad_input[0].size())
        print('grad_output size:', grad_output[0].size())
        # print('grad_input norm:', grad_input[0].norm())
        print('-'*20)


class A3Clstm_debug(torch.nn.Module):
    '''A3C lstm for visualization'''
    def __init__(self, num_inputs, num_actions):
        super(A3Clstm_debug, self).__init__()

        # self.conv1 = nn.Conv2d(num_inputs, 128, 8, stride=4, padding=1)
        # self.conv2 = nn.Conv2d(128, 64, 4, stride=2, padding=1)
        # self.conv3 = nn.Conv2d(64, 64, 4, stride=2, padding=1)

        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)

        lstm_size = 512
        num_outputs = num_actions
        self.lstm = nn.LSTMCell(1024, lstm_size)
        self.critic_linear = nn.Linear(lstm_size, 1)
        self.actor_linear = nn.Linear(lstm_size, num_outputs)

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)
        # self.conv5.weight.data.mul_(relu_gain)
        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        # register backward hook
        # for m in self.children():
        #     m.register_backward_hook(self.printgradnorm)

        self.train()


    def forward(self, inputs):
        x, (hx, cx) = inputs
        conv1 = self.conv1(x)
        max1 = self.maxp1(conv1)
        z1 = F.relu(max1)

        conv2 = self.conv2(z1)
        max2 = self.maxp2(conv2)
        z2 = F.relu(max2)

        conv3 = self.conv3(z2)
        max3 = self.maxp3(conv3)
        z3 = F.relu(max3)

        conv4 = self.conv4(z3)
        max4 = self.maxp4(conv4)
        z4 = F.relu(max4)

        x = z4.view(z4.size(0), -1)  # flatten
        # print(x.shape)
        hx, cx = self.lstm(x, (hx, cx))

        x = hx

        return conv1, z1, conv2, z2, conv3, z3, conv4, z4,\
            self.critic_linear(x), self.actor_linear(x), (hx, cx)