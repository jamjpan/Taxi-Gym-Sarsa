import gym
import random
import os
import time

# How many episodes you want?
episodes = 5000

# Set the epsilon for e-greedy action selection.
epsilon = 0.1

# Set the step size for updating action-value function.
alpha = 0.1

# What's the maximum episode length you want?
T = 2000

################################################################################
def select_action(s):
    if s in Q_s_a:
        # Greedily select next action with probability 1-epsilon
        return (max(Q_s_a[s], key=Q_s_a[s].get)
            if random.random() > epsilon else env.action_space.sample())
    else:
        return env.action_space.sample()


def train():
    for k in range(episodes):

        # Initialize the first state and get action according to policy
        S0 = env.reset()
        A0 = select_action(S0)
        reward = 0

        for t in range(T):
            # env.render()

            # Perform action and collect reward
            S1, reward, done, info = env.step(A0)

            # Select the next action on S1
            A1 = select_action(S1)

            # Update Q_s_a
            if S0 not in Q_s_a:
                Q_s_a[S0] = {}
            q0 = Q_s_a[S0][A0] if S0 in Q_s_a and A0 in Q_s_a[S0] else 0
            q1 = Q_s_a[S1][A1] if S1 in Q_s_a and A1 in Q_s_a[S1] else 0
            qn = q0 + alpha*(reward + q1 - q0)
            Q_s_a[S0][A0] = qn

            # Prepare for next state step
            S0 = S1
            A0 = A1;

            if done:
                print("ep={:04d} finished after {:03d} timesteps, reward={}"
                    .format(k, t+1, reward))
                break

    env.close()


def test():
    state = env.reset()
    done = False

    while not done:
        os.system("clear")
        env.render()
        time.sleep(0.5)
        action = max(Q_s_a[state], key=Q_s_a[state].get)
        state, reward, done, info = env.step(action)

    print("Test finished with reward={}".format(reward))

################################################################################
env = gym.make('Taxi-v3')

Q_s_a = {}  # state -> { action -> value }

print("Start")
train()
test()
print("Finish")

