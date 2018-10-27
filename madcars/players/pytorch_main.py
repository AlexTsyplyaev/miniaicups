import json
import random
import datetime
import numpy as np
import os
import argparse
from filelock import FileLock

import numpy as np

import torch, torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def to_one_hot(y, n_dims=None):
    """ helper #1: take an integer vector (tensor of variable) and convert it to 1-hot matrix. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    return Variable(y_one_hot)


def where(cond, x_1, x_2):
    """ helper #2: like np.where but in PyTorch. """
    return (cond * x_1) + ((1-cond) * x_2)


# < YOUR CODE HERE >
class Model(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(n_states, n_states * 3)
        self.fc2 = nn.Linear(n_states * 3, n_states * 2)
        self.fc3 = nn.Linear(n_states * 2, int(n_states * 1.5))
        self.fc_out1 = nn.Linear(int(n_states * 1.5), n_actions)
        self.fc_out2 = nn.Linear(int(n_states * 1.5), n_states)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        o1 = self.fc_out1(x)
        o2 = self.fc_out2(x)
        return o1, o2


def define_network(state_dim, n_actions):
    return Model(state_dim, n_actions)


# < YOUR CODE HERE >
n_actions = 3
def get_action(state, epsilon):
    """
    sample actions with epsilon-greedy policy
    recap: with probability = epsilon pick random action, else pick action with highest Q(s,a)
    """
    state = Variable(torch.FloatTensor(state))
    q_values, _ = agent(state)
    q_values = q_values.data.numpy()

    r = np.random.choice(2, p=[epsilon, 1-epsilon])
    if r == 1:
        return int(np.argmax(q_values))
    else:
        return random.choice([0,1,2])


# < YOUR CODE HERE >
def compute_td_loss(states, actions, rewards, next_states, is_done, gamma=0.99, check_shapes=False):
    """ Compute td loss using torch operations only."""
    states = Variable(torch.FloatTensor(states))  # shape: [batch_size, state_size]
    actions = Variable(torch.IntTensor(actions))  # shape: [batch_size]
    rewards = Variable(torch.FloatTensor(rewards))  # shape: [batch_size]
    next_states = Variable(torch.FloatTensor(next_states))  # shape: [batch_size, state_size]
    is_done = Variable(torch.FloatTensor(is_done))  # shape: [batch_size]

    # get q-values for all actions in current states
    predicted_qvalues, predictes_states = agent(states)  # < YOUR CODE HERE >
    # select q-values for chosen actions
    predicted_qvalues_for_actions = torch.sum(
        predicted_qvalues.cpu() * to_one_hot(actions, n_actions), dim=1)
    # compute q-values for all actions in next states
    predicted_next_qvalues, _ = agent(next_states)  # < YOUR CODE HERE >
    # compute V*(next_states) using predicted next q-values
    next_state_values, _ = torch.max(predicted_next_qvalues, dim=1)

    assert isinstance(next_state_values.data, torch.FloatTensor)

    # compute 'target q-values' for loss
    target_qvalues_for_actions = rewards + gamma * next_state_values  # < YOUR CODE HERE >
    # at the last state we shall use simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
    target_qvalues_for_actions = where(is_done, rewards, target_qvalues_for_actions).cpu()
    # Mean Squared Error loss to minimize
    loss = torch.mean((predicted_qvalues_for_actions - target_qvalues_for_actions.detach()) ** 2)
    loss_states = torch.norm(states - predictes_states, 2)

    if check_shapes:
        assert predicted_next_qvalues.data.dim() == 2, \
            'make sure you predicted q-values for all actions in next state'
        assert next_state_values.data.dim() == 1, \
            'make sure you computed V(s-prime) as maximum over just the actions axis and not all axes'
        assert target_qvalues_for_actions.data.dim() == 1, \
            'there is something wrong with target q-values, they must be a vector'

    return loss + loss_states


parser = argparse.ArgumentParser()
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--no-eps', action='store_true', default=False)
parser.add_argument('--even', action='store_true', default=False)
args = parser.parse_args()

def modulo2(value, even):
    value = value % 2
    if even:
        return value == 0
    else:
        return value != 0

def BASE_EPSILON():
    if not args.train:
        return 0
    if args.no_eps:
        return 0.1
    return 0.5

VERBOSE = True  # used for logging

numpy_2d_arrays = [0]*9
ticknum,ticksum=0,0
gamecount=-1
gamecountscheduler=-1
wins=0
loses=0
states, actions, rewards, dones, next_states = [], [], [], [], []
gamma=0.95
lives,prev_lives=0,0
prevCoords = ()
commands = ('left', 'right', 'stop')
epsilon = BASE_EPSILON()
epsilon_decay = 0.97

agentPath = os.path.abspath(os.path.join(os.path.dirname(__file__), "agent.pth"))
agentLock = os.path.abspath(os.path.join(os.path.dirname(__file__), "agent.pth.lock"))
agent = define_network(45, 3)
if os.path.isfile(agentPath):
    # load weights
    agent.load_state_dict(torch.load(agentPath))
else:
    agent_lock = FileLock(agentLock)
    with agent_lock:
        torch.save(agent.state_dict(), agentPath)

if VERBOSE:
    FI = open("zz.txt", "a", buffering=1)
    FI.write("\nWorks\n")
    FI.write(str(datetime.datetime.now().time()))
    FI.write("\n")

opt = torch.optim.Adam(agent.parameters(), lr=1e-4)

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
                prev_lives=dict["params"]["my_lives"]
            else:
                next_states.extend([states[len(states) - i - 1] for i in reversed(range(ticknum - 1))] + states[-1])
                dones[-1] = True
                if lives==prev_lives:  # not dead yet
                    rewards[-1] = 200
                    wins+=1
                else:
                    rewards[-1] = -100
                    loses+=1
                prev_lives=lives
                if VERBOSE and gamecount%20==0:
                    FI.write("\n25 games passed in:")
                    FI.write(str(datetime.datetime.now().time()))
                    FI.write("\nTotal wins:")
                    FI.write(str(wins))
                    FI.write("\nTotal loses:")
                    FI.write(str(loses))
                    FI.write("\nTotal matches:")
                    FI.write(str(gamecountscheduler))
                    FI.write("\n")
                    FI.write("\nSum of ticks:")
                    FI.write(str(ticksum))
                    FI.write("\n")
                if args.train and modulo2(gamecount, args.even):
                    agent_lock = FileLock(agentLock)
                    with agent_lock:
                        # 1. read new agent weights
                        agent.load_state_dict(torch.load(agentPath))
                        # 2. compute loss + do backprop
                        opt.zero_grad()
                        loss = compute_td_loss(
                            states,
                            actions,
                            rewards,
                            states[1:] + [states[-1]],
                            dones)
                        loss.backward()
                        opt.step()
                        # 3. save new weights to agent.pth
                        torch.save(agent.state_dict(), agentPath)
                    if VERBOSE:
                        FI.write("TRAINED\n")
                    states, rewards, qValues, actions, dones = [], [], [], [], []
                    gamecount = 0
                    wins = 0
                    loses = 0
                    epsilon = BASE_EPSILON()
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
            state = np.array(\
                numpy_2d_arrays + mycar_data + enemycar_data + [dict["params"]["deadline_position"]]
                 + speeds + [ticknum])
            choice=get_action(state, epsilon)
            states.append(state)
            actions.append(choice)
            rewards.append(-1)
            dones.append(False)
            # update epsilon:
            if args.train:
                epsilon = max(epsilon * epsilon_decay, 0.1)
            FI.write(commands[choice])
            FI.write("\n")
            print(json.dumps({"command": commands[choice]}))
    except EOFError:
        if args.train:
            # save weights
            agent_lock = FileLock(agentLock)
            with agent_lock:
                torch.save(agent.state_dict(), agentPath)
        if VERBOSE:
            FI.write("\n")
            FI.write("BAD WOLF")
            FI.write("\n")
            FI.close()
        exit(0)
