

import numpy as np
import random 
import torch
import torch.nn as nn
from torch.optim import Adam
import time 
from model import Actor
from memory import SequentialMemory
from random_process import OrnsteinUhlenbeckProcess
from util import *


class NeurWIN(object):

    def __init__(self, stateSize, lr, env, seed, numEpisodes, hidden,
                 batchSize, discountFactor, saveDir, episodeSaveInterval):
        #-------------constants-------------
        self.seed = seed
        torch.manual_seed(self.seed)
        self.myRandomPRNG = random.Random(self.seed)
        self.G = np.random.RandomState(self.seed) 
        
        self.numEpisodes = numEpisodes
        self.episodeRanges = np.arange(0, self.numEpisodes+episodeSaveInterval, episodeSaveInterval)
        self.stateSize = stateSize
        self.batchSize = batchSize
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.sigmoidParam = 5.
        self.num_states = 100
        self.episodeTimeLimit = 300

        self.beta = discountFactor
        self.env = env
        self.actor = Actor(self.stateSize, 1, hidden)

        self.num_layers = len(self.actor.fc)
        self.weight_grads = np.zeros((self.num_layers, self.batchSize)).tolist()
        self.bias_grads = np.zeros((self.num_layers, self.batchSize)).tolist()
        

        self.numOfActions = 2
        self.directory = saveDir

        self.LearningRate = lr 
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.LearningRate)
        if self.device == torch.device('cuda'): 
            self.cuda()

        #-------------counters-------------
        self.currentMiniBatch = 0
        self.batchCounter = 0
        self.episodeRewards = []
        self.discountRewards = []
     

    def newMiniBatchReset(self, index, state):
        '''Function for new mini-batch procedures. For recovering bandits, the actviation cost is chosen for a random state.'''
        random_state = np.random.randint(low=0, high=self.num_states)
        random_state = torch.FloatTensor([random_state]).to(self.device)
        self.cost = self.actor.forward(random_state).cpu().detach().numpy()[0]
    
    def cuda(self):
        torch.cuda.set_device(0) # specify which gpu to train on
        self.actor.cuda()

    def takeAction(self, state):
        '''Function for taking action based on the sigmoid function's generated probability distribution.'''
        
        index = self.actor.forward(torch.FloatTensor(state)).to(self.device)

        if (self.episodeTimeStep == 0) and (self.currentEpisode % self.batchSize == 0):
            self.newMiniBatchReset(index, state)
        
        sigmoidProb = torch.sigmoid(self.sigmoidParam*(index - self.cost))
        probOne = sigmoidProb.detach().numpy()[0]
        probs = [probOne, 1-probOne]
        probs = np.array(probs)
        probs /= probs.sum()


        action = self.G.choice([1,0], 1, p = probs)
        if action == 1:
            logProb = torch.log(sigmoidProb)   
            
            logProb.backward()
        
        elif action == 0:
            logProb = torch.log(1 - sigmoidProb) 
            
            logProb.backward()

        return action[0]

    def _saveEpisodeGradients(self):
        '''Function for saving the gradients of each episode in one mini-batch'''

        for layer in range(self.num_layers):
            self.weight_grads[layer][self.batchCounter-1] = self.actor.fc[layer].weight.grad.clone()
            self.bias_grads[layer][self.batchCounter-1] = self.actor.fc[layer].bias.grad.clone()

        self.optimizer.zero_grad(set_to_none=False)
        
    def _performBatchStep(self):
        '''Function for performing the gradient ascent step on accumelated mini-batch gradients.'''
        print('performing batch gradient step')
        
        meanBatchReward = sum(self.discountRewards) / len(self.discountRewards)
        for i in range(len(self.discountRewards)):
            self.discountRewards[i] = self.discountRewards[i] - meanBatchReward


        for layer in range(self.num_layers):
            for x in range(self.batchSize):
                self.actor.fc[layer].weight.grad += self.discountRewards[x]*self.weight_grads[layer][x]
                self.actor.fc[layer].bias.grad += self.discountRewards[x]*self.bias_grads[layer][x]
            

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=False)

        self.weight_grads = np.zeros((self.num_layers, self.batchSize)).tolist()
        self.bias_grads = np.zeros((self.num_layers, self.batchSize)).tolist()
        self.discountRewards = []  
        

    def _discountRewards(self, rewards):
        '''Function for discounting an episode's reward based on set discount factor.'''
        for i in range(len(rewards)):
            rewards[i] = (self.beta**i) * rewards[i]
        return -1*sum(rewards) 

    def learn(self):
        print(f'starting training')
        self.start = time.time()
        self.currentEpisode = 0
        self.totalTimestep = 0
        self.episodeTimeStep = 0
        self.episodeTimeList = []

        while self.currentEpisode < self.numEpisodes:
            if self.currentEpisode in self.episodeRanges:
                self.close(self.currentEpisode)
            episodeRewards = []
            s_0 = self.env.reset()

            done = False

            while done == False:
                
                action = self.takeAction(s_0)
                s_1, reward, done, info = self.env.step(action)

                if action == 1:
                    reward -= self.cost  
                episodeRewards.append(reward)
                s_0 = s_1

                self.totalTimestep += 1
                self.episodeTimeStep += 1
                
                if self.episodeTimeStep == (self.episodeTimeLimit):
                    done = True
                if done:
                    print(f'finished episode: {self.currentEpisode+1}')
                    self.discountRewards.append(self._discountRewards(episodeRewards))
                    self.batchCounter += 1

                    self.episodeRewards.append(sum(episodeRewards)) 
                    self._saveEpisodeGradients()
                    episodeRewards = []
                    self.currentEpisode += 1
                    self.episodeTimeList.append(self.episodeTimeStep)
                    self.episodeTimeStep = 0

                    if self.batchCounter == self.batchSize:
                        self._performBatchStep()
                        self.currentMiniBatch += 1
                        self.batchCounter = 0

        self.end = time.time()
        self.close(self.numEpisodes)

    def close(self, episode):
        '''Function for saving the NN parameters at defined interval *episodeSaveInterval* '''
        
        directory=(f'{self.directory}'+f'trainedNumEpisodes_{episode}')
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        torch.save(self.actor.state_dict(), directory+'/trained_model.pt')


