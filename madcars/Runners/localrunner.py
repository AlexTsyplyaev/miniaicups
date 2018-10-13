#!/usr/bin/env python3
from itertools import product
from asyncio import events
from mechanic.game import Game
from mechanic.strategy import KeyboardClient, FileClient
import numpy as np
import os
import sys

maps = ['PillMap']#,'PillHubbleMap', 'PillHillMap', 'PillCarcassMap', 'IslandMap', 'IslandHoleMap']
cars = ['Buggy']#, 'Bus', 'SquareWheelsBuggy']
games = [','.join(t) for t in product(maps, cars)]*101
cur_dir = os.path.dirname(os.path.basename(__file__))
rel_path = '../players'.split('/')
python_path = os.path.join(cur_dir, *rel_path)
python_interpreter = 'python'.format(
    major=sys.version_info.major, minor=sys.version_info.minor)
fc = FileClient([python_interpreter, '-u', os.path.join(python_path, 'pytorch_main.py'), '--train'], None)
sc = FileClient([python_interpreter, '-u', os.path.join(python_path, 'r.py')], None)
game = None
r = np.random.choice(2, p=[0.9, 0.1])
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
