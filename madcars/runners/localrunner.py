#!/usr/bin/env python3
from itertools import product
from asyncio import events
from mechanic.game import Game
from mechanic.strategy import KeyboardClient, FileClient
import numpy as np
import os
import sys
import argparse

parser = argparse.ArgumentParser(description='LocalRunner for MadCars')
parser.add_argument('-f', '--fp', type=str, nargs='?',
    help='Path to executable with strategy for first player',
    default=None)
parser.add_argument('-s', '--sp', type=str, nargs='?',
    help='Path to executable with strategy for second player',
    default=None)
parser.add_argument('-g', '--games-num', type=int,
    help='Number of games to play', default=100)
parser.add_argument('--full', action='store_true',
    help='Run full games/cars', default=False)
parser.add_argument('--no-eps', action='store_true', default=False)
args = parser.parse_args()

maps = ['PillMap']
cars = ['Buggy']
if args.full:
    maps = ['PillMap','PillHubbleMap', 'PillHillMap', 'PillCarcassMap',
        'IslandMap', 'IslandHoleMap']
    cars = cars = ['Buggy', 'Bus', 'SquareWheelsBuggy']

games = [','.join(t) for t in product(maps, cars)]*args.games_num
cur_dir = os.path.dirname(__file__)
rel_path = '../players'.split('/')
python_path = os.path.abspath(os.path.join(cur_dir, *rel_path))
python_interpreter = 'python{major}'.format(
    major=sys.version_info.major)
fp = [python_interpreter, '-u', os.path.join(python_path, 'pytorch_main.py'),
    '--train', '--even']
if args.fp is not None:
    fp = args.fp.split()
sp = [python_interpreter, '-u', os.path.join(python_path, 'pytorch_main.py'),
    '--train']
if args.sp is not None:
    sp = args.sp.split()
if args.no_eps:
    fp.append('--no-eps')
    sp.append('--no-eps')
fc = FileClient(fp, None)
sc = FileClient(sp, None)
game = None
r = np.random.choice(2, p=[0.5, 0.5])
if r == 1:
    print('Usual session')
    game = Game([fc, sc], games, extended_save=False)
else:
    print('Session with swapped players')
    game = Game([sc, fc], games, extended_save=False)

loop = events.new_event_loop()
events.set_event_loop(loop)

game.tick()
while not game.game_complete:
    future_message = loop.run_until_complete(game.tick())
    game.tick()

print('Winner:', game.get_winner().id)
