# Navigation With Deep Reinforcement Learning
 First Project of Udacity's deep reinforcement learning course !
By Romain Fournier

## 1. Project Details
 Here is my implementation of a deep reinforcement learning algorithm to solve an environment similar to the Unity's Banana Collector one. An agent moves in an environment containing blue and yellow bananas. The goal is to collect the yellow ones while avoiding the blue ones. This environment provides states to the agent in the form of a 37-dimensional continuous space. It contains the agent's velocity, along with a ray-based perception of objects around the agent's forward direction. In response to this state, the agent can take one of these four actions: move forward, move backward, turn left or turn right. A reward of +1 is obtained for each yellow bananas encountered and -1 for the blue ones. The environment is considered solved when the average score over 100 episodes overreaches 13.
 
 Here is an example of a trained agent.
 
 ![Alt Text](https://github.com/rmnfournier/navigation-with-deep-reinforcement-learning/blob/master/p1_navigation.gif?raw=true)

## 2. Getting started 
 1. You need to have python 3 installed on your machine, as well as [pytorch](https://pytorch.org/). 
 2. You can download the environment provided by udacity for [linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip), [macosx](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip), [windows 32-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip) or [windows 64-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip).
 You will have to extract the archive file and notice its emplacement.
 3.open the navigation.ipynb and in the second cell, put the emplacement of the environment after filename="

 4. You are ready to use the environment :D
 
## 3. Instructions
 You can play with the parameters in different ways. 
 1. Model.py : you can change the architecture of the network. 
 2. Agent.py : the usual parameters of reinforcement learning are present at the top of the file
 3. navigation.ipynb : when you load the agent, you can enable Double DQN and/or Dueling DQN
 