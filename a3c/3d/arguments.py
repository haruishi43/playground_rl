
import argparse

def create_args():
    ### Simulator Control Params

    parser = argparse.ArgumentParser(description='A3C')
    parser.add_argument(
        '--lr',
        type=float,
        default=0.00025,
        metavar='LR',
        help='learning rate (default: 0.00025)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        metavar='G',
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--tau',
        type=float,
        default=1.00,
        metavar='T',
        help='parameter for GAE (default: 1.00)')
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        metavar='S',
        help='random seed (default: 1)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=20,
        metavar='NS',
        help='number of forward steps in A3C (default: 20)')
    parser.add_argument(
        '--max-episode-length',
        type=int,
        default=1000,
        metavar='M',
        help='maximum length of an episode (default: 10000)')
    parser.add_argument(
        '--save-max',
        default=True,
        metavar='SM',
        help='Save model on every test run high score matched or bested')
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        metavar='W',
        help='how many training processes to use (default: 4)')
    parser.add_argument(
        '--shared-optimizer',
        default=True,
        metavar='SO',
        help='use an optimizer without shared statistics.')
    parser.add_argument(
        '--use-lab',
        help='Use lab or not',
        action='store_true')
    parser.add_argument(
        '--optimizer',
        default='Adam',
        metavar='OPT',
        help='shares optimizer choice of Adam or RMSprop')
    parser.add_argument(
        '--amsgrad',
        default=True,
        metavar='AM',
        help='Adam optimizer amsgrad parameter')
    parser.add_argument(
        '--vizdoom-path',
        default='/home/haruishi/pkg/ViZDoom/',
        metavar='VIZPATH',
        help='Where VizDoom is located for get maps and controls')
    parser.add_argument(
        '--vizdoom-scenarios',
        # default= 'scenarios/deadly_corridor.wad' ,
        default= 'scenarios/basic.wad', # 'scenarios/deadly_corridor.wad',
        metavar='VIZSCEN',
        help='Where VizDoom is located for get maps and controls')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        default=-1,
        nargs='+',
        help='GPUs to use [-1 CPU only] (default: -1)')
    parser.add_argument(
        '--env',
        default='VizDoom',
        metavar='ENV',
        help='environment to train on (default: VizDoom)')

    parser.add_argument(
        '--load',
        help='load trained model',
        action='store_true')
    parser.add_argument(
        '--log-dir',
        default='logs/', 
        metavar='LG', 
        help='folder to save logs')
    parser.add_argument(
        '--test',
        help='Test the data',
        action='store_true')
    parser.add_argument(
        '--num-test-episodes',
        type=int,
        default=100,
        metavar='NE',
        help='how many episodes in evaluation (default: 100)')
    parser.add_argument(
        '--render',
        help='Show visual',
        action='store_false')
    parser.add_argument(
        '--render-freq',
        type=int,
        default=1,
        metavar='RF',
        help='Frequency to watch rendered game play')
    parser.add_argument(
        '--write-video',
        help='write video',
        action='store_true')
    parser.add_argument(
        '--model-name',
        default='a3c',
        help='name of the save model')

    return parser