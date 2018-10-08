import json
import os
import datetime
import numpy as np
import pickle
from pathlib import Path
from sklearn.neural_network import MLPRegressor
agentPath=''

filename='agent.p'
def find_net(directory):
    global agentPath
    for root, dirs, files in os.walk(directory):
        for name in files:
            if name == filename:
                agentPath = Path(os.path.abspath(os.path.join(root, name)))
                return
        for d in dirs:
            if not agentPath:
                find_net(d)


find_net(r'/home/')

numpy_2d_arrays = [0]*9
ticknum,ticksum=0,0
gamecount=-1
gamecountscheduler=-1
wines=0
loses=0
states, qValues, rewards = [], [], []
gamma=0.998
lives,previouslives=0,0
prevCoords = ()
isTrained = False
commands = ('left', 'right', 'stop')
if agentPath.is_file():
    F = open(agentPath, "rb")
    agent = pickle.load(F)
    F.close()
else:
    exit(0)
    agent = MLPRegressor(hidden_layer_sizes=(256, 256), max_iter=1, warm_start=True, activation='tanh', solver='sgd')
    agent.fit(
        np.array([1., 0.,0.,1.,0.,0.,0.,0.,0.,300.,300.,329.,295.,0.,422.,295.,0.,0.,1.,900.,300.,871.,295.,0.,778.,295.,0.,0.,-1.,10.] + [0] * 14 + [0] + [0,1,0]).reshape(1, -1),
        np.array([0]).reshape(1, -1))
    F = open(agentPath, "wb")
    pickle.dump(agent, F)
    F.close()

while True:
    try:
        z = input()
        dict = json.loads(z)
        if dict["type"] == "new_match":
            gamecount+=1
            gamecountscheduler += 1
            lives=dict["params"]["my_lives"]
            prevCoords = ()
            if gamecount==0:
                previouslives=dict["params"]["my_lives"]
            else:
                if lives==previouslives:
                    rewards.append(10)
                    wines+=1
                else:
                    rewards.append(-10)
                    loses+=1
                previouslives=lives
                if (not isTrained) and (gamecount == 50):
                    isTrained = True
                    extendedRewards = []
                    for matchI in range(len(rewards)):
                        if len(states[matchI]) != len(qValues[matchI]):
                            print('len(states) != len(qValues)')
                        matchLength = len(states[matchI])
                        extendedRewards.append([[]] * matchLength)
                        if len(extendedRewards) != matchI + 1:
                            print('len(extendedRewards) != matchI + 1')
                        extendedRewards[-1][matchLength - 1] = rewards[matchI]
                        for tickI in reversed(range(matchLength - 1)): # already made update of the last reward
                            extendedRewards[matchI][tickI] = gamma * qValues[matchI][tickI + 1] # we have zero rewards on all other iterations

                        plainStates = []
                        for match in states:
                            for tick in match:
                                plainStates.append(tick)
                        plainRewards = []
                        for match in extendedRewards:
                            for tick in match:
                                plainRewards.append(tick)

                    #agent.fit(plainStates, plainRewards)
                    states, rewards,qValues = [], [],[]
                    gamecount = 0
                    wines = 0
                    loses = 0
            states.append([])
            qValues.append([])
            ticknum,numpy_2d_arrays = 0,[0]*9
            numpy_2d_arrays[dict["params"]["proto_car"]["external_id"] - 1] = 1
            numpy_2d_arrays[dict["params"]["proto_map"]["external_id"] + 2] = 1

        elif dict["type"] == "tick":

            mycar_data = list(
                [[num for elem in dict["params"]["my_car"][:1]+dict["params"]["my_car"][3:4]+dict["params"]["my_car"][4:] for num in
                 elem]+
                [dict["params"]["my_car"][1]]+[dict["params"]["my_car"][2]]][0])
            enemycar_data = list(
                [[num for elem in
                  dict["params"]["enemy_car"][:1] + dict["params"]["enemy_car"][3:4] + dict["params"]["enemy_car"][4:] for num in
                  elem] +
                 [dict["params"]["enemy_car"][1]] + [dict["params"]["enemy_car"][2]]][0])

            coords = dict["params"]["my_car"][0] + [dict["params"]["my_car"][1]]\
                     + dict["params"]["my_car"][3][:2] + dict["params"]["my_car"][4][:2]\
                     + dict["params"]["enemy_car"][0] + [dict["params"]["enemy_car"][1]]\
                     + dict["params"]["enemy_car"][3][:2] + dict["params"]["enemy_car"][4][:2]
            if not prevCoords:
                speeds = [0] * len(coords)
            else:
                speeds = []
                for i in range(len(coords)):
                    speeds.append(coords[i] - prevCoords[i])
            prevCoords = coords

            ticknum+=1
            ticksum+=1
            state=\
            [np.array(\
                numpy_2d_arrays + mycar_data + enemycar_data + [dict["params"]["deadline_position"]]
                 + speeds + [ticknum] + [1,0,0]),
            np.array(\
                numpy_2d_arrays + mycar_data + enemycar_data + [dict["params"]["deadline_position"]]
                 + speeds + [ticknum] + [0,1,0]),
            np.array(\
                numpy_2d_arrays + mycar_data + enemycar_data + [dict["params"]["deadline_position"]]
                 + speeds + [ticknum] + [0,0,1])]
            qValue=np.array([agent.predict(state[0].reshape(1, -1))[0]]+[agent.predict(state[1].reshape(1, -1))[0]]+[agent.predict(state[2].reshape(1, -1))[0]])
            qValueIdx = np.argmax(qValue)
            states[gamecount].append(state[qValueIdx].tolist())
            qValues[gamecount].append(qValue[qValueIdx])
            print(json.dumps({"command": commands[qValueIdx], 'debug': commands[qValueIdx]}))

    except EOFError:
        exit(0)
