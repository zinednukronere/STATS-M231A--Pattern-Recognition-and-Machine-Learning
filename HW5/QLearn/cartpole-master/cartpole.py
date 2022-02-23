#%%
import random
import gym
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from collections import deque
from tensorflow.compat.v1.keras.models import Sequential
from tensorflow.compat.v1.keras.layers import Dense
from tensorflow.compat.v1.keras.optimizers import Adam
import pickle


from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995


class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))
        self.losses=[]
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return 
        batch = random.sample(self.memory, BATCH_SIZE)
        losses = []
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            loss = (q_values[0][action] - q_update)**2 
            losses.append(loss)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)
        return np.mean(losses)

def cartpole():
    env = gym.make(ENV_NAME)
    score_logger = ScoreLogger(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0
    runLosses = []
    while True:
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        meanLosses = []
        while True:
            step += 1
            #env.render()
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                if len(meanLosses) > 0:
                    runLosses.append(np.mean(meanLosses))
                else:
                    runLosses.append(0)
                pickle.dump(runLosses, open("runLosses.p", "wb" ) )              
                score_logger.add_score(step, run)
                break
            meanLoss = dqn_solver.experience_replay()
            if meanLoss is not None:
                meanLosses.append(meanLoss)


if __name__ == "__main__":
    cartpole()
#%%
import pickle
file = open("runLosses.p",'rb')
runLossesLoaded = pickle.load(file)
#%%
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.plot(runLossesLoaded)
plt.xlabel("runs")
plt.ylabel("loss")
plt.title("Change in loss through runs for Cartpole-v1")

plt.show()