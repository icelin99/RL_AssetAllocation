import math
import numpy as np
import matplotlib.pyplot as plt

# Environment parameters
REWARD_TYPE = 'CARA'   # basic denotes difference; CARA denotes CARA utility function
INITIAL_WEALTH = 100    # initial wealth
MAX_STEPS = 10          # max invest setp
RISK_RETURN_HIGH = 0.2  # high risky return 
RISK_RETURN_LOW = -0.1  # low risky return
RISK_HIGH_PROB = 0.6    # probability of high risky return
SAFE_RETURN = 0.03      # riskless return
RHO = 2.0               # risk aversion coefficient
ACTIONS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # action space

# Q-learning parameters
ALPHA = 0.1   # learning rate
GAMMA = 0.99   # discount factor
EPSILON = 0.2  # explorating rate
NUM_EPISODES = 30000  # epoches


def power_utility_function(wealth, rho):
    """
    :param wealth: wealth
    :param rho: Risk aversion coefficient
    :return: utility
    """
    if wealth <= 0:
        return -float('inf')
    if rho == 1:
        return math.log(wealth)  # when rho=1, the function is log-style
    return (wealth ** (1 - rho)) / (1 - rho)

def get_reward(current_wealth, initial_wealth, rho):
    """
    reward function based on CARA
    """
    if current_wealth < 0:
        return -100  # penality for negative wealth
    else:
        current_utility = power_utility_function(current_wealth, rho)
        initial_utility = power_utility_function(initial_wealth, rho)
        return current_utility - initial_utility


# Intialize Q table (discrete)
Q = {}
wealth_bins = np.arange(-100, 1000, 10)
for wealth in wealth_bins:
    for step in range(MAX_STEPS + 1):
        Q[(wealth, step)] = {action: 0 for action in ACTIONS}

# Invest and conduct action
def invest(wealth, action):
    risk_investment = wealth * action
    safe_investment = wealth * (1 - action)
    
    # riskless 
    safe_return = safe_investment * SAFE_RETURN
    
    # risky, repeat the investment in case of random influence
    risk_return = 0
    for x in range(1000):
        if np.random.rand() < RISK_HIGH_PROB:
            risk_return += risk_investment * RISK_RETURN_HIGH
        else:
            risk_return += risk_investment * RISK_RETURN_LOW
    risk_return /= 1000
    
    # risk_return = risk_investment*0.08
    
    new_wealth = safe_investment + risk_investment + safe_return + risk_return
    return new_wealth

# Train
train_final_wealth = []  # final wealth
best_policy_results = []
best_wealth = -1000000

for episode in range(NUM_EPISODES):
    # ----------- training phrase -----------
    wealth = INITIAL_WEALTH
    step = 0
    
    policy_record = []
    while True:
        # 离散化当前状态
        discretized_wealth = (wealth // 10) * 10
        state = (discretized_wealth, step)
        
        # 选择动作 (epsilon-greedy)
        if np.random.rand() < EPSILON:
            action = np.random.choice(ACTIONS)
        else:
            action = max(Q[state], key=Q[state].get)
        
        policy_record.append([state, action])
        
        # action
        new_wealth = invest(wealth, action)
        step += 1
        
        # discretize new state
        new_discretized_wealth = (new_wealth // 10) * 10
        new_state = (new_discretized_wealth, step)
        
        # compute reward
        if REWARD_TYPE == 'basic':
            reward = new_wealth - INITIAL_WEALTH
        else:
            reward = get_reward(new_wealth, INITIAL_WEALTH, RHO)
        
        # terminal condition
        done = (new_wealth < 0) or (step >= MAX_STEPS)
        
        # update Q value
        if done:
            target = reward
        else:
            max_future_q = max(Q[new_state].values())
            target = reward + GAMMA * max_future_q
            
        Q[state][action] += ALPHA * (target - Q[state][action])
        
        # update state
        wealth = new_wealth
        if done:
            break
         
    if wealth > best_wealth:
        best_wealth = wealth
        best_policy_results = policy_record
        
    train_final_wealth.append(wealth)
    
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode+1}/{NUM_EPISODES}")
        print(f"Train Final Wealth: {wealth:.2f}")


plt.figure(figsize=(10, 6))
plt.plot(train_final_wealth, alpha=0.5, label='Final Wealth')
plt.xlabel('Episode')
plt.ylabel('Final Wealth')
plt.title('QL Training Performance')
plt.axhline(INITIAL_WEALTH, color='red', linestyle='--')
plt.legend()

plt.savefig(f"training_results_QL_{REWARD_TYPE}.png")


# save the best policy
with open(f"best_policy_QL_{REWARD_TYPE}.txt",'w') as f:
    for x in best_policy_results:
        f.write('State: '+str(round(x[0][0],2))+'\tAction:'+str(x[1])+'\n')