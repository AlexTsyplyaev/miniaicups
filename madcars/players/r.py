import json
import random

input()
try:
    while True:
        z = input()
        commands = ['left', 'right', 'stop']
        cmd = random.choice(commands)
        print(json.dumps({"command": cmd}))
except EOFError:
    exit(0)
