# # Imports
from gym import Env
from gym.spaces import Discrete, Box
from gym import spaces
import numpy as np
import random
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
import time
import matplotlib as plt
from rl.agents import SARSAAgent
from rl.policy import EpsGreedyQPolicy

def start(distanz, speed):  
    
    class CustomEnv(Env):
        
        Ausgabe = []
        
        def __init__(self, distanz, speed):
            self.action_space = spaces.Discrete(3) 
            self.observation_space = spaces.Box(
                low=0, high=999, shape=(3,))
            self.state = [40, 40, 40]
            obj_touched = False
            self.prev_reward = 0.0
            self.distanz = distanz
            self.speed = speed

        def _transform_action(self, action):
            if action == 0: action = [ 0, 0, 0.0] # Nothing
            if action == 1: action = [-1, 0, 0.0] # Left
            if action == 2: action = [+1, 0, 0.0] # Right
            return action    

        def distanzfunktion(self, distanz):
                return distanz#[random.randint(1, 40) for i in range(3)]

        def speedfunktion(self, speed):
                return speed

        def step(self, action):
            action = self._transform_action(action) # diskreter actionspace
            self.state = self.distanzfunktion(self.distanz)
            self.true_speed = self.speedfunktion(self.speed)

            #print("Abstände: ", self.state)
            #print("Aktion: ", action)

            global Ausgabe
            Ausgabe = action
            
            
            if 0 in self.state:
                obj_touched = True

            self.reward += 2*self.true_speed
            step_reward = 0
            self.done = False
            self.done = True
            if action is not None:  # First step without action, called from reset()
                if action[0] > 0 or action[0] < 0: # bestrafung fürs lenken
                    self.reward -= 5
                if self.true_speed <= 0.05: # Bestrafung fürs stehenbleiben
                    self.reward -= 30
                else: 
                    self.reward += 40
                if self.reward >= 100000 or self.reward <= -20000 or self.obj_touched == True: # Abbruch wenn objekt berührt
                    self.done = True
                    print("Reward: ", self.reward)
                step_reward = self.reward - self.prev_reward
                self.prev_reward = self.reward
            return self.state, step_reward, self.done, {}


        def reset(self):
            self.distance = [40, 40, 40]
            self.reward = 0
            self.prev_reward = 0.0
            self.obj_touched = False
            return self.state

    env = CustomEnv(distanz, speed)
    states = env.observation_space.shape[0]
    # print('States:', states)
    actions = env.action_space.n
    # print('Actions:', actions)

    def agent(states, actions):
        model = Sequential()
        model.add(Flatten(input_shape = (1, states)))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(actions, activation='linear'))
        return model

    model = agent(env.observation_space.shape[0], env.action_space.n)

    policy = EpsGreedyQPolicy()

    sarsa = SARSAAgent(model = model, policy = policy, nb_actions = env.action_space.n, gamma=0.95)

    sarsa.compile('adam', metrics = ['mse'])

    sarsa.load_weights('.\sarsa_weights_Rewards_Abstand_konstSpeed_5.h5f')
    
    scores = sarsa.test(env, nb_episodes = 1, visualize= False)

    return Ausgabe
