class REINFORCEBaselineAgent:
def 	init	(self, num_actions, num_states, gamma=0.99, learning_rate=0.01):
self.num_actions = num_actions self.num_states = num_states self.gamma = gamma self.learning_rate = learning_rate
self.theta = np.zeros((num_states, num_actions)) self.theta_baseline = np.zeros(num_states)
def get_action(self, state):
action_probs = self._softmax(np.dot(self.theta[state], self.theta_baseline[state]))
action = np.random.choice(self.num_actions, p=action_probs) return action
def train(self, episode):
states, actions, rewards = zip(*episode) returns = self._calculate_returns(rewards)
for t, (state, action) in enumerate(zip(states, actions)): delta = returns[t] - self.theta_baseline[state] self.theta_baseline[state] += self.learning_rate * delta self.theta[state, action] += self.learning_rate * delta
def _calculate_returns(self, rewards): returns = []
G = 0
 
 	 

for r in reversed(rewards):
G = r + self.gamma * G returns.insert(0, G)
return returns
def _softmax(self, x):
exp_values = np.exp(x - np.max(x)) return exp_values / np.sum(exp_values)
class ActorCriticBaselineAgent(REINFORCEBaselineAgent): def train(self, episode):
states, actions, rewards = zip(*episode) returns = self._calculate_returns(rewards)
for t, (state, action) in enumerate(zip(states, actions)): delta = returns[t] - self.theta_baseline[state] self.theta_baseline[state] += self.learning_rate * delta
action_probs = self._softmax(np.dot(self.theta[state], self.theta_baseline[state]))
G = returns[t]
for a in range(self.num_actions):
self.theta[state, a] += self.learning_rate * (G - self.theta[state, a]) * (int(a == action) - action_probs[a])
# Simple environment class SimpleEnvironment:
def 	init	(self, num_states, num_actions): self.num_states = num_states self.num_actions = num_actions
def reset(self):
 
 	 

return 0
def step(self, state, action): if action == 0: # Left
new_state = max(0, state - 1) else: # Right
new_state = min(self.num_states - 1, state + 1) reward = 1 if new_state == self.num_states - 1 else 0 return new_state, reward
# Training loop num_states = 5
num_actions = 2
num_episodes = 1000
env = SimpleEnvironment(num_states, num_actions) # REINFORCE with baseline
reinforce_agent = REINFORCEBaselineAgent(num_actions, num_states) for _ in range(num_episodes):
state = env.reset() episode = []
done = False while not done:
action = reinforce_agent.get_action(state) next_state, reward = env.step(state, action) episode.append((state, action, reward)) state = next_state
done = next_state == num_states - 1
 
 	 

reinforce_agent.train(episode) # Actor-Critic with baseline
actor_critic_agent = ActorCriticBaselineAgent(num_actions, num_states) for _ in range(num_episodes):
state = env.reset() episode = []
done = False while not done:
action = actor_critic_agent.get_action(state) next_state, reward = env.step(state, action) episode.append((state, action, reward)) state = next_state
done = next_state == num_states - 1 actor_critic_agent.train(episode)
# Training loop for REINFORCE with baseline
reinforce_agent = REINFORCEBaselineAgent(num_actions, num_states) reinforce_rewards = []
for episode in range(num_episodes): state = env.reset()
episode_states, episode_actions, episode_rewards = [], [], 0 done = False
while not done:
action = reinforce_agent.get_action(state) next_state, reward = env.step(state, action) episode_rewards += reward
 
 	 

episode_states.append(state) episode_actions.append(action) state = next_state
done = next_state == num_states - 1
reinforce_agent.train(zip(episode_states, episode_actions, [episode_rewards]
* len(episode_states))) reinforce_rewards.append(episode_rewards)
print("REINFORCE Episode {}: Total Reward = {}".format(episode + 1, episode_rewards))
# Training loop for Actor-Critic with baseline
actor_critic_agent = ActorCriticBaselineAgent(num_actions, num_states) actor_critic_rewards = []
for episode in range(num_episodes): state = env.reset()
episode_states, episode_actions, episode_rewards = [], [], 0 done = False
while not done:
action = actor_critic_agent.get_action(state) next_state, reward = env.step(state, action) episode_rewards += reward episode_states.append(state) episode_actions.append(action)
state = next_state
done = next_state == num_states - 1 actor_critic_agent.train(zip(episode_states, episode_actions,
[episode_rewards] * len(episode_states)))
 
 	 
actor_critic_rewards.append(episode_rewards)
print("Actor-Critic Episode {}: Total Reward = {}".format(episode + 1, episode_rewards))
 