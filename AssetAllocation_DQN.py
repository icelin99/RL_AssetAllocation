import numpy as np
import random
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt

# Environment parameters
REWARD_TYPE = 'CARA'   # basic denotes difference; CARA denotes CARA utility function
INITIAL_WEALTH = 100    # initial wealth
MAX_STEPS = 10          # max invest setp
RISK_RETURN_HIGH = 0.2  # high risky return 
RISK_RETURN_LOW = -0.1  # low risky return
RISKLESS_RETURN = 0.03  # riskless return
RISK_HIGH_PROB = 0.6    # probability of high risky return
RHO = 2.0               # risk aversion coefficient
ACTIONS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # action space
NUM_ACTIONS = len(ACTIONS)  # action num

# Hyper-parameters
BATCH_SIZE = 64          # batch size of experience replay
GAMMA = 0.99             # discount factor
EPSILON_START = 1.0      # initial explorating rate
EPSILON_END = 0.01       # min explorating rate
EPSILON_DECAY = 0.995    # explorating decay rate
LEARNING_RATE = 0.001    # learning rate
TARGET_UPDATE_FREQ = 10  # target network update frequency
MEMORY_SIZE = 10000      # buffer size
NUM_EPISODES = 500       # epoches


# Network model
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_shape=(2,)),  # the input is (wealth, step)
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(NUM_ACTIONS, activation="linear")  # output is Q value of each action
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="mse")
    return model

# Experience replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

# Invest funciton with random
def invest(wealth, action):
    risk_investment = wealth * action
    safe_investment = wealth * (1 - action)

    risk_return = 0
    for x in range(1000):
        if np.random.rand() < RISK_HIGH_PROB:
            risk_return += risk_investment * RISK_RETURN_HIGH
        else:
            risk_return += risk_investment * RISK_RETURN_LOW
    risk_return /= 1000
    # risk_return = risk_investment*0.08

    new_wealth = safe_investment*(1+RISKLESS_RETURN) + risk_investment + risk_return
    return new_wealth

# Baisc reward funciton
def get_reward_basic(current_wealth, initial_wealth, step, is_terminal):
    if current_wealth < 0:
        return -100  # penality for negative wealth
    elif is_terminal:
        return current_wealth - initial_wealth
    else:
        return (current_wealth - initial_wealth) / step if step != 0 else 0

# Utility function
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

# CARA reward funciton
def get_reward_CARA(current_wealth, initial_wealth, step, is_terminal):
    if current_wealth < 0:
        return -100  # penality for negative wealth
    else:
        current_utility = power_utility_function(current_wealth, rho)
        initial_utility = power_utility_function(initial_wealth, rho)
        return current_utility - initial_utility

# Initialize model and buffer
model = build_model()
target_model = build_model()
target_model.set_weights(model.get_weights())
replay_buffer = ReplayBuffer(MEMORY_SIZE)

# Train
epsilon = EPSILON_START
final_wealth_history = []
best_policy_results = []
best_wealth = -1000000

for episode in range(NUM_EPISODES):
    wealth = INITIAL_WEALTH
    step = 0
    total_reward = 0

    policy_record = []
    while step < MAX_STEPS and wealth >= 0:
        state = np.array([wealth, step])

        # choose action（epsilon-greedy）
        if np.random.rand() < epsilon:
            action_idx = np.random.randint(NUM_ACTIONS)  # random choose
        else:
            q_values = model.predict(state[np.newaxis], verbose=0)
            action_idx = np.argmax(q_values[0])

        action = ACTIONS[action_idx]

        # update wealth
        policy_record.append([state, action])
        new_wealth = invest(wealth, action)
        step += 1

        # judge if terminated
        is_terminal = (new_wealth < 0 or step >= MAX_STEPS)

        # compute rewards
        if REWARD_TYPE == 'basic':
            reward = get_reward_basic(new_wealth, INITIAL_WEALTH, step, is_terminal)
        else:
            reward = get_reward_CARA(new_wealth, INITIAL_WEALTH, RHO)
            
        total_reward += reward

        # store experience
        next_state = np.array([new_wealth, step])
        replay_buffer.add(state, action_idx, reward, next_state, is_terminal)

        
        # update state
        wealth = new_wealth

        # experinece replay training
        if replay_buffer.size() >= BATCH_SIZE:
            batch = replay_buffer.sample(BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = np.array(states)
            next_states = np.array(next_states)

            # compute Q value
            target_q_values = target_model.predict(next_states, verbose=0)
            targets = rewards + GAMMA * np.max(target_q_values, axis=1) * (1 - np.array(dones))

            # update model
            q_values = model.predict(states, verbose=0)
            q_values[np.arange(BATCH_SIZE), actions] = targets
            model.train_on_batch(states, q_values)

    # update target model 
    if episode % TARGET_UPDATE_FREQ == 0:
        target_model.set_weights(model.get_weights())

    # explore decay
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    
    if wealth > best_wealth:
        best_wealth = wealth
        best_policy_results = policy_record
        
    # record final wealth
    final_wealth_history.append(wealth)

    if (episode + 1) % 10 == 0:
        print(f"Episode: {episode + 1}, Final Wealth: {wealth}, Epsilon: {epsilon:.2f}")


plt.figure(figsize=(10, 6))
plt.plot(final_wealth_history, alpha=0.5, label='Final Wealth')
plt.xlabel("Epoch")
plt.ylabel("Final Wealth")
plt.title("DQN Training Performance")
plt.axhline(y=INITIAL_WEALTH, color="r", linestyle="--", label="Initial Wealth")
plt.legend()
plt.savefig(f"training_results_DQN_{REWARD_TYPE}.png")



# save the best policy
with open(f"best_policy_DQN_{REWARD_TYPE}.txt",'w') as f:
    for x in best_policy_results:
        f.write('State: '+str(round(x[0][0],2))+'\tAction:'+str(x[1])+'\n')