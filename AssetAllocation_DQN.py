import numpy as np
import random
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt

# 环境参数
INITIAL_WEALTH = 100  # 初始财富
MAX_STEPS = 10        # 每个 epoch 的最大投资步数
RISK_RETURN_HIGH = 0.2  # 高风险投资的高回报率
RISK_RETURN_LOW = -0.1  # 高风险投资的低回报率
RISKLESS_RETURN = 0.03
RISK_HIGH_PROB = 0.6    # 高风险投资获得高回报的概率
ACTIONS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # 动作空间
NUM_ACTIONS = len(ACTIONS)  # 动作数量

# DQN 超参数
BATCH_SIZE = 64          # 经验回放的批量大小
GAMMA = 0.99             # 折扣因子
EPSILON_START = 1.0      # 初始探索率
EPSILON_END = 0.01       # 最小探索率
EPSILON_DECAY = 0.995    # 探索率衰减率
LEARNING_RATE = 0.001    # 学习率
TARGET_UPDATE_FREQ = 10  # 目标网络更新频率
MEMORY_SIZE = 10000      # 经验回放缓冲区大小
NUM_EPISODES = 500       # 训练的总 epoch 数


# 定义神经网络模型
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_shape=(2,)),  # 输入是 (wealth, step)
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(NUM_ACTIONS, activation="linear")  # 输出是每个动作的 Q 值
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="mse")
    return model

# 定义经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

# 定义环境动态
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

# 定义奖励函数
def get_reward(current_wealth, initial_wealth, step, is_terminal):
    if current_wealth < 0:
        return -100  # 对负财富的惩罚
    elif is_terminal:
        return current_wealth - initial_wealth  # 最终财富的差值
    else:
        return (current_wealth - initial_wealth) / step if step != 0 else 0

# 初始化模型和目标网络
model = build_model()
target_model = build_model()
target_model.set_weights(model.get_weights())

# 初始化经验回放缓冲区
replay_buffer = ReplayBuffer(MEMORY_SIZE)

# 训练过程
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

        # 选择动作（epsilon-greedy）
        if np.random.rand() < epsilon:
            action_idx = np.random.randint(NUM_ACTIONS)  # 随机选择动作
        else:
            q_values = model.predict(state[np.newaxis], verbose=0)
            action_idx = np.argmax(q_values[0])

        action = ACTIONS[action_idx]

        # 执行动作，更新财富
        policy_record.append([state, action])
        new_wealth = invest(wealth, action)
        step += 1

        # 判断是否为终止状态
        is_terminal = (new_wealth < 0 or step >= MAX_STEPS)

        # 计算奖励
        reward = get_reward(new_wealth, INITIAL_WEALTH, step, is_terminal)
        total_reward += reward

        # 存储经验
        next_state = np.array([new_wealth, step])
        replay_buffer.add(state, action_idx, reward, next_state, is_terminal)

        
        # 更新状态
        wealth = new_wealth

        # 经验回放训练
        if replay_buffer.size() >= BATCH_SIZE:
            batch = replay_buffer.sample(BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = np.array(states)
            next_states = np.array(next_states)

            # 计算目标 Q 值
            target_q_values = target_model.predict(next_states, verbose=0)
            targets = rewards + GAMMA * np.max(target_q_values, axis=1) * (1 - np.array(dones))

            # 更新模型
            q_values = model.predict(states, verbose=0)
            q_values[np.arange(BATCH_SIZE), actions] = targets
            model.train_on_batch(states, q_values)

    # 更新目标网络
    if episode % TARGET_UPDATE_FREQ == 0:
        target_model.set_weights(model.get_weights())

    # 衰减探索率
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    
    if wealth > best_wealth:
        best_wealth = wealth
        best_policy_results = policy_record
    # 记录最终财富
    final_wealth_history.append(wealth)

    # 打印训练进度
    if (episode + 1) % 10 == 0:
        print(f"Episode: {episode + 1}, Final Wealth: {wealth}, Epsilon: {epsilon:.2f}")

# 绘制最终财富随 epoch 变化的曲线
plt.figure(figsize=(10, 6))
plt.plot(final_wealth_history, alpha=0.5, label='Final Wealth')
plt.xlabel("Epoch")
plt.ylabel("Final Wealth")
plt.title("DQN Training Performance")
plt.axhline(y=INITIAL_WEALTH, color="r", linestyle="--", label="Initial Wealth")
plt.legend()
plt.savefig('training_results_DQN.png')



# save the best policy
with open('best_policy_DQN.txt','w') as f:
    for x in best_policy_results:
        f.write('State: '+str(round(x[0][0],2))+'\tAction:'+str(x[1])+'\n')