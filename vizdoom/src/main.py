import os
os.environ['OMP_NUM_THREADS'] = '1'

from doom.game import play
# from doom.runners import run_basic, run_deadly_corridor
from utils.args import parse_arguments
import argparse


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='AI runner')
    # parser.add_argument("--scenario", type=str, help="Scenario")
    # args, remaining = parser.parse_known_args()
    #
    # if args.scenario is not None:
    #     run_scenario(args.scenario)

    game_args = parse_arguments()
    play(game_args)