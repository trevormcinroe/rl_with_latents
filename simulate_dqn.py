import argparse
import torch
import matplotlib.pyplot as plt
import pickle
import numpy as np
from collections import namedtuple
from environments.kuka import KukaEnv
from models import DQNCnn
from agents.dqn_agent import DQNAgent
from tqdm import tqdm
from memories import ReplayBuffer
import time

parser = argparse.ArgumentParser(description='Training simulation for various deep RL environments.')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
parser.add_argument('--seed', type=int, default=19, help='random seed')
parser.add_argument('--repeat', type=int, default=25, help='number of times to repeat a given action')
parser.add_argument('--max-ep-len', type=int, default=1000, help='maximum number of steps per episode')
parser.add_argument('--render', action='store_true', default=False, help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how often to print progress')
parser.add_argument('--episodes', type=int, default=10000, help='number of training episodes to run')
parser.add_argument('--images', action='store_true', default=False, help='Image-based states')
parser.add_argument('--name', help='experiment name')
args = parser.parse_args()

env = KukaEnv(
    renders=args.render,
    is_discrete=True,
    max_steps=args.max_ep_len,
    action_repeat=args.repeat,
    images=args.images,
    static_all=True,
    static_obj_rnd_pos=False,
    rnd_obj_rnd_pos=False,
    full_color=False
)

env.seed(args.seed)
torch.manual_seed(args.seed)

saved_action = namedtuple('saved_action', ['log_prob', 'value'])

policy_net = DQNCnn(7)
target_net = DQNCnn(7)

memory = ReplayBuffer(100000, 4)

agent = DQNAgent(memory, 64, 1., 1e-6, 0.05, 7, policy_net, target_net, 0.99, 0.001)

e = 0
N_EXPLORE = 10

for i in tqdm(range(N_EXPLORE)):
    done = False
    s = env.reset()

    # To make this compatible with the ReplayBuffer, we need to expand the 3rd channel...
    s = np.expand_dims(s, 2)

    while not done:
        last_stored_frame_idx = agent.memory.store_frame(s)
        obs = agent.memory.encode_recent_observation()
        a = np.random.choice([x for x in range(7)])

        s_, r, picked_up, d, _ = env.step(a)
        s_ = np.expand_dims(s_, 2)

        if d:
            t = 1
        else:
            t = 0

        agent.memory.store_effect(last_stored_frame_idx, a, r, t)

        if d:
            done = True

        s = s_

successful_eps = []
start = time.time()

for i in range(args.episodes):
    done = False
    s = env.reset()
    # To make this compatible with the ReplayBuffer, we need to expand the 3rd channel...
    s = np.expand_dims(s, 2)
    inner_success = []
    env_steps = 0
    n_param_steps = 0

    while not done:
        last_stored_frame_idx = agent.memory.store_frame(s)
        obs = agent.memory.encode_recent_observation()
        a = agent.choose_action(obs)
        s_, r, picked_up, d, _ = env.step(a.item())
        s_ = np.expand_dims(s_, 2)

        env_steps += 1

        if d:
            t = 1
        else:
            t = 0

            agent.memory.store_effect(last_stored_frame_idx, a, r, t)

        inner_success.append(picked_up)

        if d:
            done = True

        # 42 steps per episode it seems
        if env_steps % 5 == 0:
            for _ in range(20):
                agent.learn()
            n_param_steps += 1

        # if n_param_steps % 100 == 0:
        #     agent.update_target_net()

        s = s_

    if np.sum(inner_success) > 0:
        successful_eps.append(1)
    else:
        successful_eps.append(0)

    if i % 10 == 0:
        agent.update_target_net()

    if i % args.log_interval == 0:
        print(f'Episode: {i}, Pct: {np.mean(successful_eps[-100:])}, Hours time {(time.time() - start) / 3600}, Eps: {agent.epsilon}')




filename_net = 'results/model_' + str(args.name) + '.pth'
torch.save(policy_net.state_dict(), filename_net)

# Plotting
with open(f'./results/results_{args.name}.data', 'wb') as file:
    pickle.dump(successful_eps, file)

smoothed = []
b_idx = 0
e_idx = 100
while e_idx < len(successful_eps):
    smoothed.append(np.mean(successful_eps[b_idx:e_idx]))
    b_idx += 1
    e_idx += 1
fig = plt.figure(dpi=400)
plt.plot(smoothed)
plt.title('Success Rate in Kuka Environment (Pick Up)')
plt.xlabel('Episode')
plt.ylabel('Success Rate')
plt.savefig(f'../results/results_{args.name}.png')

