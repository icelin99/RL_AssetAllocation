# Asset Allocation Assignment for Reinforcement Learning
HKUST MSBD6000M course project for the Asset Allocation in Reinforcement Learning


## Introduction
This project implements a discrete-time asset allocation strategy using Temporal Difference (TD) learning methods, specifically TD SARSA and Q-learning algorithms. The problem involves optimizing investment decisions between risky and risk-free assets over a fixed time horizon of T=10 periods. The **objective** of the project is to find the optimal policy, which describe how much money(action) you 
should invest in the risky asset under different time & total wealth(states).


In our model, the risky asset follows a simple binary return process where:
- With probability p, the return is a (positive return)
- With probability 1-p, the return is a (negative return)


The environment is structured as a Markov Decision Process (MDP) where:
- States represent the current wealth level
- Actions determine the proportion of wealth allocated to the risky asset
- Rewards are calculated using a CARA (Constant Absolute Risk Aversion) utility function
- The risk-free asset provides a fixed return

We implemented two TD learning approaches:

1. **TD SARSA (On-Policy Learning)**: Updates Q-values using the action selected by the current policy, making it more conservative but potentially more stable.

2. **TD Q-Learning (Off-Policy Learning)**: Updates Q-values using the maximum Q-value of the next state, potentially leading to more optimal solutions but with higher variance.

The key advantage of using TD methods for this problem is their ability to learn from each interaction without requiring complete episodes, making them more efficient than traditional Monte Carlo methods. Additionally, these methods can handle the continuous state space (wealth levels) through discretization while maintaining reasonable computational efficiency.

****

## **Mathematical Derivation**

While the textbook (Rao and Jelvis, section 8.4) assumes normally distributed returns for the risky asset, our implementation considers a discrete binary distribution. This necessitates a modification of the Q-function derivation. Below is our adapted mathematical reasoning:



****


## **Project Layout**


### **Model Assumption & Env Setting**
Considering the fact that, in asset allocation, our investment decision is based on the current amount 
of money and the time due to maturity, I set the time and amount of wealth as the state in the Env 
system. Besides, due to the constraints of algorithm running speed, I simply set two action to do 
in each state, which means the agent can either invest 0.2 or 0.8 of his total wealth into risky asset. 
Based on the above assumption, I simply view it as a finite discrete problem in reinforcement learning.
  
The Env setting is as below:
- **State**: (t, wealth)
- **Action**: (0.2, 0.8)
- **Reward**: CARA utility function of wealth  - exp(- a * w) / a
- Risky returns ~ Bernoulli {0.4: go up 0.5, 0.6: go down -0.3}
- Riskless return: 0.05

Since the prob and act is known and state is finite, I first generate all the possible wealth states at different time 
using For loop, then use Dict to restore the transition probabilistic matrix and 
the states space(see the AssetAllocation class). I also implement the step and reset function in the class. After finish 
the env setting, we can generate agent algorithm.

***

### **Algorithm Code**



#### **TD SARSA**
Unlike the MC method have to compute the mean of q value, which means we have to update policy after 
we get all data, the TD SARSA method allow us to compute q value and update policy once we have 
only one data.

Due to the problem request and env setting, the First-Visit and Every-Visit method can not be deployed in this project, 
hope to compete them in the future.

_The Pseudocode can be checked in the above Book link._

***
#### **TD Q-Learning**
Unlike the TD SARSA have to get the experience of (s, a, r, s, a), the TD Q-learning method only needs
(s, a, r, s), which decrease the variance and become more effective.

I write the **On-Policy** and **Off-Policy** iteration method of the Q-Learning algorithm. The only difference 
between these two method is whether the update policy is as same as the sampling policy.

_The Pseudocode can be checked in the above Book link._

****
### **Result Analysis**
The training results can be visualized in the following figures:

![Q-Learning Training Results](./training_results_QL.png)
*Figure 1: Q-Learning algorithm's training results showing the wealth accumulation over episodes*

![SARSA Training Results](./training_results_SARSA.png)
*Figure 2: SARSA algorithm's training results showing the wealth accumulation over episodes*



