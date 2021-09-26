import time
import gym
import math
import random
import numpy as np
import matplotlib
from celluloid import Camera
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from IPython import display

#The DeepQNetwork
class DQN(nn.Module):
    def __init__(self, img_h, img_w):
        super().__init__()
        self.lay1 = nn.Linear(img_h*img_w*3, 20)
        self.lay2 = nn.Linear(20,30)
        self.lay3 = nn.Linear(30,2)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.lay1(x))
        x = F.relu(self.lay2(x))
        x = self.lay3(x)

        return x

#Experience Tuple
Experience = namedtuple(
    'Experience',
    ('state','action','reward','next_state')
)

#RelayMemory Class
class RelayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.count = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.count % self.capacity] = experience
        self.count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide(self, batch_size):
        return len(self.memory) > batch_size

#Epsilon-Greedy Stratedy
class EpsilonGreedy():
    def __init__(self, max, min, decay):
        self.max = max
        self.min = min
        self.decay = decay

    def get_exploration_rate(self, cur_step):
        return self.min + (self.max - self.min)*math.exp(-1*cur_step*self.decay)

#Agent Class
class Agent():
    def __init__(self, strategy, num_action):
        self.strategy = strategy 
        self.num_action = num_action
        self.cur_step = 0 

    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.cur_step)
        self.cur_step += 1 

        if rate > random.random():
            return torch.tensor([random.randrange(self.num_action)])
        else:
            with torch.no_grad():
                return policy_net(state).argmax(dim=1)

#Environment Manager Class
class EnvManager():
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.env.reset()
        self.cur_screen = None
        self.done = False 

    def reset(self):
        self.env.reset()
        self.cur_screen = None

    def close(self):
        self.env.close()

    def render(self, mode = "human"):
        return self.env.render(mode)

    def num_action(self):
        return self.env.action_space.n 
    
    def take_action(self, action):
        _, reward, self.done, _ = self.env.step(action.item())
        return torch.tensor([reward])

    def just_starting(self):
        return self.cur_screen is None 

    def get_state(self):
        if self.just_starting() or self.done:
            self.cur_screen = self.get_processed_screen()
            blank_screen = torch.zeros_like(self.cur_screen)
            return blank_screen
        else:
            s1 = self.cur_screen
            s2 = self.get_processed_screen()
            self.cur_screen = s2
            return s2 - s1

    def get_height(self):
        screen = self.get_processed_screen()
        return screen.shape[2]

    def get_width(self):
        screen = self.get_processed_screen()
        return screen.shape[3]

    def get_processed_screen(self):
        screen = self.render('rgb_array').transpose((2, 0, 1))
        screen = self.cropped(screen)
        return self.transformed_screen(screen)

    def cropped(self, screen):
        screen_height = screen.shape[1]
        top = int(screen_height * 0.4)
        bottom = int(screen_height * 0.8)
        screen = screen[:, top:bottom, :]
        return screen

    def transformed_screen(self, screen):
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        resize = T.Compose([
            T.ToPILImage()
            ,T.Resize((40,90))
            ,T.ToTensor()
        ])
        return resize(screen).unsqueeze(0)
    
def plot(values, episode):
    plt.figure(2)
    plt.clf()        
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(episode,values)
    plt.pause(0.001)
    display.clear_output(wait=True)

#Hyper-Parameters of DQN 
batch_size = 256
gamma = 0.999
eps_max = 1
eps_min = 0.01
eps_decay = 0.001
target_update = 10
memory_size = 100000
lr = 0.001
num_episodes = 1000


def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return (t1,t2,t3,t4)

#Get QValues   
class QValues():
    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))
    @staticmethod        
    def get_next(target_net, next_states):                
        final_state_locations = next_states.flatten(start_dim=1) \
            .max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        return values

#Initialization......... 
em = EnvManager()
strategy = EpsilonGreedy(eps_max, eps_min, eps_decay)
agent = Agent(strategy, em.num_action())
memory = RelayMemory(memory_size)

policy_net = DQN(em.get_height(), em.get_width())
target_net = DQN(em.get_height(), em.get_width())

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)
episode_duration = []
episode = []

for e in range(num_episodes):
    em.reset() 
    state = em.get_state()

    for timestep in count():
        action = agent.select_action(state, policy_net)
        reward = em.take_action(action)
        next_state = em.get_state()
        memory.push(Experience(state, action, reward, next_state))
        state = next_state

        if memory.can_provide(batch_size):
            experiences = memory.sample(batch_size)
            states, actions, rewards, next_states = extract_tensors(experiences)

            current_q_values = QValues.get_current(policy_net, states, actions)
            next_q_values = QValues.get_next(target_net, next_states)
            target_q_values = (next_q_values * gamma) + rewards

            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if em.done:
            episode_duration.append(timestep)
            episode.append(e)
            plot(episode_duration, episode)
            break
    
    if e % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

em.close()