from itertools import product
from asyncio import events
from mechanic.game import Game
from mechanic.strategy import KeyboardClient, FileClient
import numpy as np

maps = ['PillMap','PillHubbleMap', 'PillHillMap', 'PillCarcassMap', 'IslandMap', 'IslandHoleMap']
cars = ['Buggy', 'Bus', 'SquareWheelsBuggy']
games = [','.join(t) for t in product(maps, cars)]*5
fc =FileClient(['python', '..\\examples\\python2(3)\\main.py'], None)
sc = FileClient(['python', '..\\examples\\python2(3)\\r.py'], None)
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

print(game.get_winner().id)
