from __future__ import division
from __future__ import print_function

import itertools as it
import cv2
import numpy as np

import random

from vizdoom import *

class EnvVizDoom(object):
    def __init__(self, scenario_path, seed=None):
        self.channels = 1
        self.resolution = (self.channels, 80, 80)
        self.game = DoomGame()
        self.game.set_doom_scenario_path(scenario_path)

        #FIXME: load config file
        #self.game.set_screen_format(ScreenFormat.GRAY8)
        self.game.set_screen_format(ScreenFormat.RGB24)
        #self.game.set_screen_resolution(ScreenResolution.RES_160X120)
        self.game.set_render_hud(True) # False
        self.game.set_render_crosshair(False)
        self.game.set_render_weapon(True)
        self.game.set_render_decals(False)
        self.game.set_render_particles(False)
        self.game.set_episode_start_time(10) # 10 20
        self.game.set_window_visible(False)
        self.game.set_sound_enabled(False)
        self.game.set_mode(Mode.PLAYER)


        # BASIC
        self.game.set_doom_map("map01")
        self.game.set_screen_resolution(ScreenResolution.RES_640X480)
        self.game.add_available_button(Button.MOVE_LEFT)
        self.game.add_available_button(Button.MOVE_RIGHT)
        self.game.add_available_button(Button.ATTACK)
        self.game.set_episode_timeout(300)
        self.game.set_living_reward(-1)

        # # Deadly Corridor
        # self.game.set_screen_resolution(ScreenResolution.RES_320X240)
        # self.game.add_available_button(Button.MOVE_LEFT)
        # self.game.add_available_button(Button.MOVE_RIGHT)
        # self.game.add_available_button(Button.ATTACK)
        # self.game.add_available_button(Button.MOVE_FORWARD)
        # self.game.add_available_button(Button.MOVE_BACKWARD)
        # self.game.add_available_button(Button.TURN_LEFT)
        # self.game.add_available_button(Button.TURN_RIGHT)
        # self.game.add_available_game_variable(GameVariable.HEALTH)
        # self.game.set_episode_timeout(300)
        # self.game.set_death_penalty(100)
        # self.game.set_doom_skill(5)
        
        # Other
        # self.game.add_available_game_variable(GameVariable.AMMO2)
        # self.game.add_available_game_variable(GameVariable.POSITION_X)
        # self.game.add_available_game_variable(GameVariable.POSITION_Y)
        if seed is not None:
            self.game.set_seed(seed)
        self.game.init()
        print("Doom initialized.")

        n = self.game.get_available_buttons_size()
        self.actions = [list(a) for a in it.product([0, 1], repeat=n)]
        # self.actions = np.identity(3, dtype=bool).tolist()
        self.num_actions = len(self.actions)
        # print(self.actions)
        print(self.num_actions)

    def reset(self):
        self.game.new_episode()

    def act(self, action, frame_repeat=1):
        # print(action)
        action = self.map_actions(action)
        # action = random.choice(self.actions)
        # action = self.actions[57]
        # print(action)
        
        reward = self.game.make_action(action, frame_repeat)
        done = self.game.is_episode_finished()
        return reward, done
        
        # return self.game.make_action(action, frame_repeat)

    def is_finished(self):
        return self.game.is_episode_finished()

    def observation(self):
        return self.preprocess(self.game.get_state().screen_buffer)

    def raw_observation(self):
        return self.game.get_state().screen_buffer

    def map_actions(self, action_raw):
        return self.actions[action_raw]

    def preprocess(self, frame):
        """
        Change the resolution size and return numpy array
        """
        if self.channels == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (self.resolution[2], self.resolution[1]))
        return np.reshape(frame, self.resolution)
