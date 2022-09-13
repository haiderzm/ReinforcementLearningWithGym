## Q-Learning

##### Q-learning is a model-free reinforcement learning algorithm to learn the value of an action in a particular state. It does not require a model of the environment (hence "model-free"), and it can handle problems with stochastic transitions and rewards without requiring adaptations.

<br>

#### In Q learning
<br>

##### We replace iterative format of bellman equation
![alt text](/home/haider/Desktop/bellman_iterative.png)
<br>

##### And use it in form of expectation
![alt text](/home/haider/Desktop/bellman_expect.png)

##### Equation for training of the RL agent
![alt text](/home/haider/Desktop/rl_train.png)

##### Q-Learning using Q Table
![alt text](/home/haider/Desktop/q_learning.png)

<br>
<br>

## Deep Q-Learning
##### In Deep Q-learning, we use a neural network to approximate the Q-value function. The state is given as the input and the Q-value of all possible actions is generated as the output. 

<br>

##### Deep Q Algorithm
![alt text](/home/haider/Desktop/deep_q_algo.png)
<br>

##### Bellman Equation

##### In deep Q-learning, we use a neural network to approximate the Q-value function. The state is given as the input and the Q-value of all possible actions is generated as the output. 
<br>

##### Deep Q-Learning using Neural Networks
![alt text](/home/haider/Desktop/deep_q_learning.png)

## Some Results
<br>
<br>

##### Result of Q-learning on MountainCar-v0 Gym Environment

##### Result of Q-learning on CartPole Gym Environment

##### Result of Q-learning on Breakout Atari Gym Environment
