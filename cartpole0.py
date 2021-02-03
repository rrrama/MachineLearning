import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import random
random.seed(123)
import gym
from collections import deque
import matplotlib.pyplot as plt
import time

class CartPoleAgent:

    def __init__(self, iterations = 10000):
        self.exploration_rate = 1
        self.exploration_delta = 1/iterations
        self.batch_size = 10
        self.memory = deque(maxlen=1000)
        self.gamma = 0.95

        self.model = Sequential()
        self.model.add(Dense(24,input_shape=(4,),activation="relu"))
        self.model.add(Dense(24,activation="relu"))
        self.model.add(Dense(2,activation="linear"))
        self.model.compile(loss="mse", optimizer="adam")

    def act(self,state,explore=True):
        if self.exploration_rate > random.random() and explore:
            return random.randint(0,1)
        return np.argmax(self.model.predict(state))

    def store(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory,self.batch_size)
        """
        reward_vec = np.array([a[2] for a in batch]).transpose()
        next_state_matrix = np.array([a[3] for a in batch])
        update = reward_vec+np.amax(self.model.predict(next_state_matrix,batch_size=self.batch_size),axis=1)

        """
        for state,action,reward,next_state,done in batch:
            if done:
                update = reward
            else:
                update = reward + self.gamma*np.amax(self.model.predict(next_state)[0])

            #print("test "+str(state.shape))
            values = self.model.predict(state)
            #print(values)
            #print(values.shape)
            values[0][action] = update
            self.model.fit(state,values,verbose=0)



def main():
    env = gym.make("CartPole-v0")
    iterations=200
    agent = CartPoleAgent(iterations)
    scores=[]
    for run in range(iterations):
        state = env.reset()
        #print(state.shape)
        state = np.reshape(state, (1,4))
        move=0
        done = False
        while not done:
            action = agent.act(state)
            next_state,reward,done,info = env.step(action)
            #print(next_state.shape)
            next_state = np.reshape(next_state, (1,4))
            if done:
                reward *= -1
            agent.store(state,action,reward,next_state,done)
            state = next_state
            agent.learn()
            move+=1
        scores.append(move)
        agent.exploration_rate -= agent.exploration_delta
        agent.exploration_rate = max(0.01, agent.exploration_rate)
        if run%1==0:
            print("Run {} finished with score {}, exp_rate = {}".format(run,move,agent.exploration_rate))
    return scores

if __name__=="__main__":
    start = time.time()
    data = main()
    end = time.time()
    print("time taken: {}".format(end-start))
    plt.plot(data)
    plt.show()
