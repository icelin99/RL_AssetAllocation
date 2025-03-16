import numpy as np
import matplotlib.pyplot as plt

# 环境参数
INITIAL_WEALTH = 100  # 初始财富
MAX_STEPS = 10        # 每个 epoch 的最大投资步数
RISK_RETURN_HIGH = 0.2  # 高风险投资的高回报率
RISK_RETURN_LOW = -0.1  # 高风险投资的低回报率
RISK_HIGH_PROB = 0.6    # 高风险投资获得高回报的概率
SAFE_RETURN = 0.03      # 无风险投资的回报率
ACTIONS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # 动作空间

# Q-learning 参数
ALPHA = 0.1   # 学习率
GAMMA = 0.99   # 折扣因子
EPSILON = 0.2  # 探索率
NUM_EPISODES = 30000  # 训练的总 epoch 数
EVAL_REPEATS = 20    # 每个 epoch 评估时重复次数

# 初始化 Q 表（离散化状态）
# 财富离散化为 10 的倍数，步数为 0~10
Q = {}
wealth_bins = np.arange(-100, 1000, 10)  # 离散化范围
for wealth in wealth_bins:
    for step in range(MAX_STEPS + 1):
        Q[(wealth, step)] = {action: 0 for action in ACTIONS}

# 环境动态
def invest(wealth, action):
    risk_investment = wealth * action
    safe_investment = wealth * (1 - action)
    
    # 无风险收益
    safe_return = safe_investment * SAFE_RETURN
    
    # 高风险收益
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

# 训练过程
train_final_wealth = []  # 记录训练时的最终财富
eval_final_wealth = []   # 记录评估时的平均最终财富
best_policy_results = []
best_wealth = -1000000

for episode in range(NUM_EPISODES):
    # ----------- 训练阶段 -----------
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
        
        # 执行动作
        new_wealth = invest(wealth, action)
        step += 1
        
        # 离散化新状态
        new_discretized_wealth = (new_wealth // 10) * 10
        new_state = (new_discretized_wealth, step)
        
        # 计算奖励
        reward = new_wealth - INITIAL_WEALTH  # 简单差值奖励
        
        # 终止条件
        done = (new_wealth < 0) or (step >= MAX_STEPS)
        
        # 更新 Q 值
        if done:
            target = reward
        else:
            max_future_q = max(Q[new_state].values())
            target = reward + GAMMA * max_future_q
            
        Q[state][action] += ALPHA * (target - Q[state][action])
        
        # 更新状态
        wealth = new_wealth
        if done:
            break
         
    if wealth > best_wealth:
        best_wealth = wealth
        best_policy_results = policy_record
        
    train_final_wealth.append(wealth)
    
    
    # 打印进度
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode+1}/{NUM_EPISODES}")
        print(f"Train Final Wealth: {wealth:.2f}")


# 绘制结果
plt.figure(figsize=(10, 6))
plt.plot(train_final_wealth, alpha=0.5, label='Final Wealth')
# plt.plot(np.convolve(train_final_wealth, np.ones(50)/50, mode='valid'), 
#          color='red', label='50-episode Moving Avg')
plt.xlabel('Episode')
plt.ylabel('Final Wealth')
plt.title('QL Training Performance')
plt.axhline(INITIAL_WEALTH, color='red', linestyle='--')
plt.legend()

plt.savefig('training_results_QL.png')


# save the best policy
with open('best_policy_QL.txt','w') as f:
    for x in best_policy_results:
        f.write('State: '+str(round(x[0][0],2))+'\tAction:'+str(x[1])+'\n')