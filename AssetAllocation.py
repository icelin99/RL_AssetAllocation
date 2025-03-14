import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

LEARNING_RATE = 0.005
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
        print("\n--- decision process ---")
        for i, decision in enumerate(self.decisions):
            print(
                f"step {i+1}: \t state: {decision['state']} \t action: {decision['action']} \t reward: {decision['reward']} \t next state: {decision['next_state']}"
            )

        print("\nTotal decision steps:", len(self.decisions))

    def print_q_values(self, q_values):
        print("\n--- Q values table ---")
        sorted_q_values = sorted(q_values.items(), key=lambda x: (x[0][0], x[0][1]))
        current_state = None
        for (state, action), value in sorted_q_values:
            if state != current_state:
                print(f"state: {state}: ")
                current_state = state
            print(f"action(risk asset allocation ratio): {action:.1f} \t Q value: {value:.4f}")


class PortfolioInvestmentEnv(gym.Env):
    """
    Environment for portfolio investment

    Features:
    - Part of the funds are invested in risk assets
    - Risk assets have two possible returns
    - The remaining funds are invested in risk-free assets
    """

    def __init__(
        self,
        initial_wealth=10000,  # initial wealth
        max_steps=10,  # maximum investment steps
        riskless_return=0.03,  # risk-free asset return
        risk_high_return=0.2,  # high return
        risk_low_return=-0.1,  # low return
        risk_high_prob=0.6,  # high return probability
        risk_aversion=1.0,  # risk aversion coefficient
    ):
        super().__init__()

        # environment parameters
        self.initial_wealth = initial_wealth
        self.max_steps = max_steps
        self.riskless_return = riskless_return
        self.risk_high_return = risk_high_return
        self.risk_low_return = risk_low_return
        self.risk_high_prob = risk_high_prob
        self.risk_aversion = risk_aversion

        # state and action space
        # action space: [0, 1], represents the proportion of investment in risk assets
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # observation space: current wealth
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(1,), dtype=np.float32
        )

        # internal state variables
        self.current_wealth = None
        self.current_step = None

    def reset(self, seed=None):
        """reset environment"""
        super().reset(seed=seed)
        self.current_wealth = self.initial_wealth
        self.current_step = 0
        return np.array([self.current_wealth], dtype=np.float32), {}

    def step(self, action):
        """
        execute investment action

        Args:
            action (np.ndarray): risk asset investment ratio [0, 1]

        Returns:
            observation, reward, terminated, truncated, info
        """
        # ensure action is within legal range
        risk_allocation = np.clip(action, 0, 1)[0]

        # risk asset return
        if np.random.random() < self.risk_high_prob:
            risk_return = self.risk_high_return
        else:
            risk_return = self.risk_low_return

        new_wealth = self.current_wealth * risk_allocation * (
            1 + risk_return
        ) + self.current_wealth * (1 - risk_allocation) * (1 + self.riskless_return)

        self.current_wealth = round(new_wealth, PRECISION)
        self.current_step += 1

        # calculate reward (using CARA utility function)
        reward = self._cara_utility(self.current_wealth)

        # termination condition
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
        Constant Absolute Risk Aversion (CARA) utility function

        Args:
            wealth (float): current wealth

        Returns:
            float: utility value
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
        ε-greedy strategy to choose action

        Args:
            state (np.ndarray): current state

        Returns:
            float: chosen action
        """
        state_key = self._get_state_key(state)

        if np.random.random() < self.exploration_rate:
            return round(float(self.action_space.sample()[0]), PRECISION)
        else:
            # if state does not exist, return random action
            state_exists = any((key[0] == state_key for key in self.q_values.keys()))
            if not state_exists:
                return round(float(self.action_space.sample()[0]), PRECISION)

            # find the best action in the current state
            return self._get_best_action(state_key)

    def _get_state_key(self, state):
        """
        convert state to a hashable key

        Args:
            state (np.ndarray): state

        Returns:
            float: state value (rounded to 3 decimal places)
        """
        return round(float(state[0]), PRECISION)

    def _get_best_action(self, state_key):
        """
        get the best action in the given state

        Args:
            state_key (float): state key

        Returns:
            float: best action
        """
        state_actions = {
            action: self.q_values.get((state_key, action), 0.0)
            for action in np.round(np.linspace(0, 1, 11), PRECISION)
        }
        return max(state_actions, key=state_actions.get)

    def learn(self, state, action, reward, next_state, done):
        """
        Q-Learning update

        Args:
            state (np.ndarray): current state
            action (float): executed action
            reward (float): received reward
            next_state (np.ndarray): next state
            done (bool): whether to terminate
        """
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)

        # exact match action
        action = round(float(action), PRECISION)

        # get current Q value
        current_q = self.q_values.get((state_key, action), 0.0)

        # calculate max Q value of next state
        if done:
            max_next_q = 0.0
        else:
            next_state_actions = [
              self.q_values[key] for key in self.q_values.keys()
              if key[0] == next_state_key 
            ]
            max_next_q = max(next_state_actions) if next_state_actions else 0.0
            # max_next_q = max(next_state_actions)

        # Q value update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )

        # store updated Q value
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
        ε-greedy strategy to choose action

        Args:
            state (np.ndarray): current state

        Returns:
            float: chosen action
        """
        state_key = self._get_state_key(state)

        if np.random.random() < self.exploration_rate:
            return round(float(self.action_space.sample()[0]), PRECISION)
        else:
            # if state does not exist, return random action
            state_exists = any((key[0] == state_key for key in self.q_values.keys()))
            if not state_exists:
                return round(float(self.action_space.sample()[0]), PRECISION)

            # find the best action in the current state
            return self._get_best_action(state_key)

    def _get_state_key(self, state):
        """
        convert state to a hashable key

        Args:
            state (np.ndarray): state

        Returns:
            float: state value (rounded to 3 decimal places)
        """
        return round(float(state[0]), PRECISION)

    def _get_best_action(self, state_key):
        """
        get the best action in the given state

        Args:
            state_key (float): state key

        Returns:
            float: best action
        """
        state_actions = {
            action: self.q_values.get((state_key, action), 0.0)
            for action in np.round(np.linspace(0, 1, 11), PRECISION)
        }
        return max(state_actions, key=state_actions.get)

    def learn(self, state, action, reward, next_state, next_action, done):
        """
        SARSA update

        Args:
            state (np.ndarray): current state
            action (float): executed action
            reward (float): received reward
            next_state (np.ndarray): next state
            next_action (float): next action
            done (bool): whether to terminate
        """
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)

        # exact match action
        action = round(float(action), PRECISION)
        next_action = round(float(next_action), PRECISION)

        # get current Q value
        current_q = self.q_values.get((state_key, action), 0.0)

        # calculate Q value of next state-action pair
        if done:
            next_q = 0.0
        else:
            next_q = self.q_values.get((next_state_key, next_action), 0.0)

        # Q value update (SARSA key difference is using next_action instead of max(next_actions))
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_q - current_q
        )

        # store updated Q value
        self.q_values[(state_key, action)] = new_q


def train_sarsa_agent(env, agent, episodes=500):
    """
    train agent using SARSA

    Args:
        env (PortfolioInvestmentEnv): investment environment
        agent (SARSAAgent): learning agent
        episodes (int): number of training episodes

    Returns:
        list: final wealths of each training episode
    """
    final_wealths = []
    decision_logger = DecisionLogger()

    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0

        # choose initial action
        action = agent.choose_action(state)

        while not done:
            next_state, reward, done, _, _ = env.step(np.array([action]))

            # choose next action
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

    print("final_wealths:", [float(w) for w in final_wealths], len(final_wealths),np.mean(final_wealths))
    decision_logger.print_summary()
    # decision_logger.print_q_values(agent.q_values)
    return final_wealths


def train_QL_agent(env, agent, episodes=500):
    """
    train agent using Q-Learning

    Args:
        env (PortfolioInvestmentEnv): investment environment
        agent (QLearningAgent): learning agent
        episodes (int): number of training episodes

    Returns:
        list: final wealths of each training episode
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
    print("final_wealths:", [float(w) for w in final_wealths], len(final_wealths),np.mean(final_wealths))
    decision_logger.print_summary()
    # decision_logger.print_q_values(agent.q_values)
    return final_wealths


def plot_training_results(final_wealths, method):
    """
    plot training results

    Args:
        final_wealths (list): final wealths of each training episode
    """
    plt.figure(figsize=(12, 6))
    plt.plot(final_wealths,
        label="average wealth with " + method,
    )
    plt.title("wealth during training")
    plt.xlabel("episodes")
    plt.ylabel("average wealth")
    plt.legend()
    filename = "training_results_" + method + ".png"
    plt.savefig(filename)
    plt.close()


def main():
    # set random seed
    np.random.seed(42)

    # create investment environment
    env = PortfolioInvestmentEnv(
        initial_wealth=100,
        max_steps=10,
        riskless_return=0.03,
        risk_high_return=0.2,
        risk_low_return=-0.1,
        risk_high_prob=0.6,
        risk_aversion=1.0,
    )

    # create Q-Learning agent
    agent1 = QLearningAgent(env.action_space, env.observation_space)
    agent2 = SARSAAgent(env.action_space, env.observation_space)

    # train agent
    # final_QL_wealths = train_QL_agent(env, agent1, episodes=100)

    final_SARSA_wealths = train_sarsa_agent(env, agent2, episodes=100)

    print("Training complete.")


if __name__ == "__main__":
    main()
