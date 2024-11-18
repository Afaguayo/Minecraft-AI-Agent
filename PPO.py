from torch.distributions import MultivariateNormal   # Way to "Explore" actions
from torch.optim import Adam                         # Optimizer 
from PolicyNetwork import PolicyNetwork              # Simple Feed Forward Network
import torch
import numpy as np

def process_observation(obs):
    inventory = obs['inventory']
    inventory_vector = np.array([inventory[item] for item in inventory.keys()], dtype=np.float32)
    normalizedInvenotry =inventory_vector / 2304.0  # Normalize by max count
    inventory_tensor = torch.tensor(normalizedInvenotry, dtype=torch.float32)

    pov = obs['pov']
    # Normalize to [0, 1] and convert to tensor
    pov_tensor = torch.tensor(np.array(pov), dtype=torch.float32) / 255.0  # Normalize pixel values
    # Change to (Channel, Height, Width) for PyTorch
    pov_tensor = pov_tensor.permute(2, 0, 1)  # Permute dimensions

    # Need to combine POV INVENTORY becasue networks need a single unified input
    # Inventory provides scalar information
    # POV provides spatial data

    # Flatten the POV tensor
    # Ensure the tensor is contiguous before flattening
    pov_flattened = pov_tensor.contiguous().view(-1)  # From (3, 360, 640) to (3 * 360 * 640,)
    # Concatenate inventory and POV
    combined_features = torch.cat((inventory_tensor, pov_flattened), dim=0)
    return combined_features

class PPO():
    def __init__(self, env, observation_dimension, actor_dimension):
        self.env = env

        self.observation_dimension = observation_dimension   # In minecraft it is 692176, combined feature size from POV and inventory
        self.actor_dimension = actor_dimension               # 23 descrete + 2 continuous (25 in total)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameter for PPO
        self.hyperparameter()

        # Initialize critic and actor networks
        # Critic outputs a single scalar value (state value)
        self.critic = PolicyNetwork(self.observation_dimension,1,0).to(self.device)
        # Actor outputs discrete logits (23) and continuous means (2)
        self.actor = PolicyNetwork(self.observation_dimension,23,2).to(self.device)

        # Optimizer for actor and critic networks
        self.criticOptimizer = Adam(self.critic.parameters(), lr=self.lr)
        self.actorOptimizer = Adam(self.actor.parameters(), lr= self.lr)

        # Covariance matrix for continuous actions
        self.covariancematrix = torch.diag(torch.full(size=(2,), fill_value=0.5))

    def hyperparameter(self):
        # Total timesteps per batch of training
        self.timestep_per_batch = 1200
        # Maximum timesteps per episode
        self.max_steps = 1600
        # Discounted facter for future rewards
        self.gamma = 0.95
        # Clipping parameter for PPO updates
        self.clip = 0.2
        # Learning rate for Adam optimizer
        self.lr = 0.005
        # Number of update steps for actor and critic networks per batch
        self.updates_for_Actor_critic = 5

    def get_action(self, observation):
        # Convert observations to tensor and add batch dimension
        observation_tensor = observation.clone().detach().unsqueeze(0).float().to(self.device)

        # observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        # Ensure covariance matrix is on the correct device
        self.covariancematrix = self.covariancematrix.to(self.device)

        # Forward pass thrugh the actor network
        discrete_logits, continuous_mean = self.actor(observation_tensor)

        # Sample descrete action
        discrete_probs = torch.softmax(discrete_logits, dim=1)
        discrete_action = torch.multinomial(discrete_probs, num_samples=1).item()

        # Sample continuus action
        continuous_distribution = MultivariateNormal(continuous_mean, self.covariancematrix)
        continuous_action = continuous_distribution.sample().detach().cpu().numpy()

        # Compute log probabilities for the sampled actions
        discrete_log_prob = torch.log(discrete_probs[0, discrete_action])
        continuous_log_prob = continuous_distribution.log_prob(torch.tensor(continuous_action).to(self.device))

        # Return the sampled action and log probabilities
        action = {
            "discrete": discrete_action,
            "continuous": continuous_action
        }

        log_prob = discrete_log_prob + continuous_log_prob
        return action, log_prob
    

    def rollout(self):
        # Initialize storage for the abtch data
        observationList, actionList, log_probsList, rewardsList = [],[],[],[]

        # Count the tiemsteps in the current batch
        timestep = 0

        while timestep < self.timestep_per_batch:
            # Reset the environment and episode storage
            obs = self.env.reset()

            episode_rewards = []
            done = False

            while not done and timestep < self.timestep_per_batch:
                # Combined inventory and POV to create a single feature vector
                combined_features = process_observation(obs).to(self.device)
                
                # Get action and log probability
                action, log_prob = self.get_action(combined_features)

                obs, reward, done, _ = self.env.step(action)

                # Store data
                observationList.append(combined_features)
                actionList.append(action)
                log_probsList.append(log_prob)
                episode_rewards.append(reward)

                # Increment timestep count
                timestep +=1
            # Store rewards for this episode
            rewardsList.append(episode_rewards)

        # Convert to tensor
        observationList = torch.tensor(observationList, dtype=torch.float32).to(self.device)
        log_probsList = torch.Tensor(log_probsList, dtype= torch.float32).to(self.device)

        # Compute rewarsd-to-go for each episode
        rewards_to_goList = self.rewards_to_go_computation(rewardsList).to(self.device)

        return observationList, actionList, log_probsList, rewards_to_goList

    def rewards_to_go_computation(self, rewardsList):
        # Store the reards to go for all episodes
        rewards_to_goList = []

        for episode_rewards in rewardsList:
            discounted_reward = 0

            epsiode_rewards_to_go = []
            # Compute rewards to go in reverse
            for reward in reversed(episode_rewards):
                discounted_reward = reward + self.gamma * discounted_reward

                epsiode_rewards_to_go.insert(0, discounted_reward)
            rewards_to_goList.extend(epsiode_rewards_to_go)

        # Convert to tensor
        return torch.tensor(rewards_to_goList, dtype=torch.float32).to(self.device)
    
    def evaluate(self, observationList, actionList):

        observationList = observationList.to(self.device)
        # Forwards pass through the actor network
        discrete_logits, continuous_mean = self.actor(observationList)

        # Process discrete action
        discrete_actions = torch.tensor([a["discrete"] for a in actionList], dtype=torch.int64).to(self.device)
        discrete_probs = torch.softmax(discrete_logits, dim=-1)
        discrete_log_probs = torch.log(discrete_probs.gather(1, discrete_actions.unsqueeze(-1)).squeeze())

        # Process continuous action
        continuous_actions = torch.tensor([a["continuous"] for a in actionList], dtype=torch.float32).to(self.device)
        continuous_distribution = MultivariateNormal(continuous_mean, self.covarianceMatrix)
        continuous_log_probs = continuous_distribution.log_prob(continuous_actions)

        # Combine log probabilities
        log_probs = discrete_log_probs + continuous_log_probs

        # Query critic network for value estimates
        state_values = self.critic(observationList).squeeze()

        return state_values, log_probs
    
    def learn(self, timestep):
        # Track the total number of timesteps processed
        currentTimestep = 0

        while currentTimestep < timestep:
            # Collect data using rollout
            observationList, actionList, log_probsList, reward_to_goList = self.rollout()

            # Move reward_to_goList and log_probsList to device
            reward_to_goList = reward_to_goList.to(self.device)
            log_probsList = log_probsList.to(self.device)


            # Update timestep count
            currentTimestep += len(observationList)

            # Evaluate the observations and actions
            state_values, current_log_probs = self.evaluate(observationList, actionList)

            # Compute advantages
            advantages = reward_to_goList - state_values.detach().to(self.device)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)  # Normalize advantages

            # Perform multiple update steps for actor and critic
            for _ in range(self.updates_for_actor_critic):
                # Re-evaluate state values and log probabilities
                state_values, current_log_probs = self.evaluate(observationList, actionList)

                # Compute the ratio for PPO
                ratios = torch.exp(current_log_probs - log_probsList)

                # Compute surrogate loss terms
                surrogate1 = ratios * advantages
                surrogate2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages

                # Policy loss (actor)
                actor_loss = -torch.min(surrogate1, surrogate2).mean()
                self.actorOptimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actorOptimizer.step()

                # Value loss (critic)
                critic_loss = torch.nn.MSELoss()(state_values, reward_to_goList)
                self.criticOptimizer.zero_grad()
                critic_loss.backward()
                self.criticOptimizer.step()

            print(f"Completed update for timestep: {currentTimestep}/{timestep}")

    def train(self, total_timesteps):
        timesteps_completed = 0
        rewards_per_episode = []  # To track rewards for logging

        while timesteps_completed < total_timesteps:
            # Perform rollout
            observationList, actionList, log_probsList, reward_to_goList = self.rollout()

            # Calculate total rewards for the last rollout
            total_reward = sum(reward_to_goList.tolist())
            rewards_per_episode.append(total_reward)

            # Perform learning
            self.learn(len(observationList))

            # Update the number of completed timesteps
            timesteps_completed += len(observationList)

            # Logging
            print(f"Timesteps Completed: {timesteps_completed}/{total_timesteps}")
            print(f"Episode Reward: {total_reward}")

        # Return the reward history for analysis
        return rewards_per_episode
