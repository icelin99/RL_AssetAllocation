import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.99
EXPLORATION_RATE = 0.1
PRECISION = 3


class DecisionLogger:
    def __init__(self):
        self.decisions = []

    def log_decision(self, state, action, reward, next_state):
        self.decisions.append(
            {
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
            }
        )

    def print_summary(self):
        print("\n--- 决策过程详细日志 ---")
        for i, decision in enumerate(self.decisions):
            print(
                f"步骤 {i+1}: \t 状态: {decision['state']} \t 动作: {decision['action']} \t 奖励: {decision['reward']} \t 下一状态: {decision['next_state']}"
            )

        print("\n总决策步数:", len(self.decisions))

    def print_q_values(self, q_values):
        print("\n--- Q值表 ---")
        sorted_q_values = sorted(q_values.items(), key=lambda x: (x[0][0], x[0][1]))
        current_state = None
        for (state, action), value in sorted_q_values:
            if state != current_state:
                print(f"状态 财富: {state}: ")
                current_state = state
            print(f"动作(风险资产配置比例): {action:.1f} \t Q值: {value:.4f}")


class PortfolioInvestmentEnv(gym.Env):
    """
    投资组合强化学习环境

    特点：
    - 部分资金投资风险资产
    - 风险资产有两种可能回报
    - 剩余资金投资无风险资产
    """

    def __init__(
        self,
        initial_wealth=10000,  # 初始财富
        max_steps=10,  # 最大投资步数
        riskless_return=0.03,  # 无风险资产回报率
        risk_high_return=0.2,  # 高回报率
        risk_low_return=-0.1,  # 低回报率
        risk_high_prob=0.6,  # 高回报概率
        risk_aversion=1.0,  # 风险厌恶系数
    ):
        super().__init__()

        # 环境参数
        self.initial_wealth = initial_wealth
        self.max_steps = max_steps
        self.riskless_return = riskless_return
        self.risk_high_return = risk_high_return
        self.risk_low_return = risk_low_return
        self.risk_high_prob = risk_high_prob
        self.risk_aversion = risk_aversion

        # 状态和动作空间
        # 动作空间：[0, 1]，表示投资风险资产的比例
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # 观测空间：当前财富
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(1,), dtype=np.float32
        )

        # 内部状态变量
        self.current_wealth = None
        self.current_step = None

    def reset(self, seed=None):
        """重置环境"""
        super().reset(seed=seed)
        self.current_wealth = self.initial_wealth
        self.current_step = 0
        return np.array([self.current_wealth], dtype=np.float32), {}

    def step(self, action):
        """
        执行投资动作

        Args:
            action (np.ndarray): 风险资产投资比例 [0, 1]

        Returns:
            observation, reward, terminated, truncated, info
        """
        # 确保动作在合法范围
        risk_allocation = np.clip(action, 0, 1)[0]

        # 风险资产回报
        if np.random.random() < self.risk_high_prob:
            risk_return = self.risk_high_return
        else:
            risk_return = self.risk_low_return

        new_wealth = self.current_wealth * risk_allocation * (
            1 + risk_return
        ) + self.current_wealth * (1 - risk_allocation) * (1 + self.riskless_return)

        self.current_wealth = round(new_wealth, PRECISION)
        self.current_step += 1

        # 计算奖励（使用CARA效用函数）
        reward = self._cara_utility(self.current_wealth)

        # 终止条件
        terminated = self.current_step >= self.max_steps or self.current_wealth <= 0

        return (
            np.array([self.current_wealth], dtype=np.float32),
            reward,
            terminated,
            False,
            {},
        )

    def _cara_utility(self, wealth):
        """
        恒定绝对风险规避(CARA)效用函数

        Args:
            wealth (float): 当前财富

        Returns:
            float: 效用值
        """
        return -np.exp(-self.risk_aversion * wealth)


class QLearningAgent:
    def __init__(
        self,
        action_space,
        observation_space,
        learning_rate=LEARNING_RATE,
        discount_factor=DISCOUNT_FACTOR,
        exploration_rate=EXPLORATION_RATE,
    ):
        self.action_space = action_space
        self.observation_space = observation_space

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

        self.q_values = {}


    def choose_action(self, state):
        """
        ε-贪婪策略选择动作

        Args:
            state (np.ndarray): 当前状态

        Returns:
            float: 选择的动作
        """
        state_key = self._get_state_key(state)

        if np.random.random() < self.exploration_rate:
            return round(float(self.action_space.sample()[0]), PRECISION)
        else:
            # 如果状态不存在，返回随机动作
            state_exists = any((key[0] == state_key for key in self.q_values.keys()))
            if not state_exists:
                return round(float(self.action_space.sample()[0]), PRECISION)

            # 找到当前状态下最佳动作
            return self._get_best_action(state_key)

    def _get_state_key(self, state):
        """
        将状态转换为可哈希的键

        Args:
            state (np.ndarray): 状态

        Returns:
            float: 状态值（四舍五入到3位小数）
        """
        return round(float(state[0]), PRECISION)

    def _get_best_action(self, state_key):
        """
        获取给定状态下最佳动作

        Args:
            state_key (float): 状态键

        Returns:
            float: 最佳动作
        """
        state_actions = {
            action: self.q_values.get((state_key, action), 0.0)
            for action in np.round(np.linspace(0, 1, 11), PRECISION)
        }
        return max(state_actions, key=state_actions.get)

    def learn(self, state, action, reward, next_state, done):
        """
        Q-Learning更新

        Args:
            state (np.ndarray): 当前状态
            action (float): 执行的动作
            reward (float): 获得的奖励
            next_state (np.ndarray): 下一个状态
            done (bool): 是否结束
        """
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)

        # 精确匹配动作
        action = round(float(action), PRECISION)

        # 获取当前Q值
        current_q = self.q_values.get((state_key, action), 0.0)

        # 计算下一状态最大Q值
        if done:
            max_next_q = 0.0
        else:
            next_state_actions = [
              self.q_values[key] for key in self.q_values.keys()
              if key[0] == next_state_key 
            ]
            max_next_q = max(next_state_actions) if next_state_actions else 0.0
            # max_next_q = max(next_state_actions)

        # Q值更新
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )

        # 存储更新后的Q值
        self.q_values[(state_key, action)] = new_q


class SARSAAgent:
    def __init__(
        self,
        action_space,
        observation_space,
        learning_rate=LEARNING_RATE,
        discount_factor=DISCOUNT_FACTOR,
        exploration_rate=EXPLORATION_RATE,
    ):
        self.action_space = action_space
        self.observation_space = observation_space

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

        self.q_values = {}

    def choose_action(self, state):
        """
        ε-贪婪策略选择动作

        Args:
            state (np.ndarray): 当前状态

        Returns:
            float: 选择的动作
        """
        state_key = self._get_state_key(state)

        if np.random.random() < self.exploration_rate:
            return round(float(self.action_space.sample()[0]), PRECISION)
        else:
            # 如果状态不存在，返回随机动作
            state_exists = any((key[0] == state_key for key in self.q_values.keys()))
            if not state_exists:
                return round(float(self.action_space.sample()[0]), PRECISION)

            # 找到当前状态下最佳动作
            return self._get_best_action(state_key)

    def _get_state_key(self, state):
        """
        将状态转换为可哈希的键

        Args:
            state (np.ndarray): 状态

        Returns:
            float: 状态值（四舍五入到3位小数）
        """
        return round(float(state[0]), PRECISION)

    def _get_best_action(self, state_key):
        """
        获取给定状态下最佳动作

        Args:
            state_key (float): 状态键

        Returns:
            float: 最佳动作
        """
        state_actions = {
            action: self.q_values.get((state_key, action), 0.0)
            for action in np.round(np.linspace(0, 1, 11), PRECISION)
        }
        return max(state_actions, key=state_actions.get)

    def learn(self, state, action, reward, next_state, next_action, done):
        """
        SARSA更新

        Args:
            state (np.ndarray): 当前状态
            action (float): 执行的动作
            reward (float): 获得的奖励
            next_state (np.ndarray): 下一个状态
            next_action (float): 下一个动作
            done (bool): 是否结束
        """
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)

        # 精确匹配动作
        action = round(float(action), PRECISION)
        next_action = round(float(next_action), PRECISION)

        # 获取当前Q值
        current_q = self.q_values.get((state_key, action), 0.0)

        # 计算下一状态-动作对的Q值
        if done:
            next_q = 0.0
        else:
            next_q = self.q_values.get((next_state_key, next_action), 0.0)

        # Q值更新 (SARSA关键区别在于使用next_action而非max(next_actions))
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_q - current_q
        )

        # 存储更新后的Q值
        self.q_values[(state_key, action)] = new_q


def train_sarsa_agent(env, agent, episodes=500):
    """
    使用SARSA训练智能体

    Args:
        env (PortfolioInvestmentEnv): 投资环境
        agent (SARSAAgent): 学习智能体
        episodes (int): 训练轮数

    Returns:
        list: 每轮训练的最终财富
    """
    final_wealths = []
    decision_logger = DecisionLogger()

    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0

        # 选择初始动作
        action = agent.choose_action(state)

        while not done:
            next_state, reward, done, _, _ = env.step(np.array([action]))

            # 选择下一个动作
            next_action = agent.choose_action(next_state)

            agent.learn(state, action, reward, next_state, next_action, done)
            decision_logger.log_decision(
                state=state, action=action, reward=reward, next_state=next_state
            )

            state = next_state
            action = next_action
            total_reward += reward

        final_wealths.append(state[0])

        if episode % 50 == 0:
            print(f"Episode {episode}: Final Wealth = {float(state[0]):.2f}")

    print("final_wealths:", final_wealths)
    decision_logger.print_summary()
    # decision_logger.print_q_values(agent.q_values)
    return final_wealths


def train_QL_agent(env, agent, episodes=500):
    """
    训练智能体

    Args:
        env (PortfolioInvestmentEnv): 投资环境
        agent (QLearningAgent): 学习智能体
        episodes (int): 训练轮数

    Returns:
        list: 每轮训练的最终财富
    """
    final_wealths = []
    decision_logger = DecisionLogger()

    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(np.array([action]))

            agent.learn(state, action, reward, next_state, done)
            decision_logger.log_decision(
                state=state, action=action, reward=reward, next_state=next_state
            )

            state = next_state
            total_reward += reward

        final_wealths.append(state[0])

        if episode % 50 == 0:
            print(f"Episode {episode}: Final Wealth = {float(state[0]):.2f}")
    print("final_wealths:", final_wealths)
    decision_logger.print_summary()
    decision_logger.print_q_values(agent.q_values)
    return final_wealths


def plot_training_results(final_wealths, method):
    """
    绘制训练结果

    Args:
        final_wealths (list): 每轮训练的最终财富
    """
    plt.figure(figsize=(12, 6))
    plt.plot(
        np.cumsum(final_wealths) / np.arange(1, len(final_wealths) + 1),
        label="acumulated average wealth with " + method,
    )
    plt.title("acumulated wealth during training")
    plt.xlabel("episodes")
    plt.ylabel("average wealth")
    plt.legend()
    filename = "training_results_" + method + ".png"
    plt.savefig(filename)
    plt.close()


def main():
    # 设置随机种子
    np.random.seed(42)

    # 创建投资环境
    env = PortfolioInvestmentEnv(
        initial_wealth=100,
        max_steps=10,
        riskless_return=0.03,
        risk_high_return=0.2,
        risk_low_return=-0.1,
        risk_high_prob=0.6,
        risk_aversion=1.0,
    )

    # 创建Q-Learning智能体
    agent1 = QLearningAgent(env.action_space, env.observation_space)
    agent2 = SARSAAgent(env.action_space, env.observation_space)

    # 训练智能体
    final_QL_wealths = train_QL_agent(env, agent1, episodes=500)

    # 绘制训练结果
    plot_training_results(final_QL_wealths, "QL")

    final_SARSA_wealths = train_sarsa_agent(env, agent2, episodes=500)
    plot_training_results(final_SARSA_wealths, 'SARSA')
    print("Training complete.")


if __name__ == "__main__":
    main()
