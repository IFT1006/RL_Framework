import pandas as pd
import numpy as np
from tqdm import tqdm

from agentSpace import AgentSpace
from learningAlgo import LearningAlgo
from agent import Agent
from envPD import EnvPD
from envBandit import EnvBandit
from utils import normalizeMatrix

class Execute:
    def __init__(self, n_instance, T, n_agents, const, title):
        self.n_instance = n_instance
        self.T = T
        self.n_agents = n_agents
        self.const = const # Julien constante n'est pas utilisé de la même façon pour ts et ucb
        # C'est de ma faute et ce n'est pas idéal. 
        # Lorsqu'on séparera les algos, ça va aider
        self.title = title # Julien pt changer ça

    def runOnePDExperiment(self, matrices, algo, noise_dist='uniform', noise_params=(0.0, 0.05)):

        # Initialisation de l'environnement
        env = EnvPD(matrices, self.n_agents, noise_dist, noise_params)
        for agent in range(0, self.n_agents):
            a_space = AgentSpace(len(matrices[0][0]), self.n_agents, 'PD', agent+1)
            learning_algo = LearningAlgo(self.const[agent], algo[agent], a_space)
            env.ajouter_agents(Agent(a_space, learning_algo))
        
        plays = []
        for time_step in range(0, self.T):
            actions = env.step()
            plays.append(actions)

        # Il faudrait enlever cumul_reward et le calucler ailleurs
        actions_played = [[i[0] for i in plays],[i[1] for i in plays]]
        cumul_rewards = [env.agents[k].cumul_reward for k in range(self.n_agents)]
        rewards = [env.agents[k].reward for k in range(self.n_agents)]

        return actions_played, cumul_rewards, rewards
    

    def getPDResult(self, matrices, algo, noise_dist='uniform', noise_params=(0.0, 0.05)):
        # Normalisation de matrices
        matrices_norm = [normalizeMatrix(mat,0) for mat in matrices]

        # Boucle sur les itérations
        df = pd.DataFrame()
        all_rewards = []
        all_cum_rewards = []
        all_plays = []

        for realisation in tqdm(range(0, self.n_instance)):
            plays, cumul_rewards, rewards = self.runOnePDExperiment(matrices_norm, algo, noise_dist, noise_params)
            all_plays.append(np.array(plays).T)
            all_rewards.append(np.array(rewards).T)
            all_cum_rewards.append (np.array(cumul_rewards).T)

        # 2) Empilement en tableaux 3D : (timesteps, n_agents, n_instance)
        plays_arr = np.stack(all_plays, axis=2)
        rewards_arr  = np.stack(all_rewards, axis=2)
        cum_rewards_arr = np.stack(all_cum_rewards, axis=2)

        # 3) Calcul de la moyenne et de l’écart-type le long de l’axe “réalisations”
        mean_r     = rewards_arr.mean(axis=2)
        std_r      = rewards_arr.std (axis=2)
        mean_cr    = cum_rewards_arr.mean(axis=2)
        std_cr     = cum_rewards_arr.std (axis=2)

        # 4) Construction du DataFrame résumé
        n_steps = mean_r.shape[0]
        df = pd.DataFrame({'step': np.arange(n_steps)})
        for i in range(self.n_agents):
            df[f'mean_reward_agent_{i}']     = mean_r[:, i]
            df[f'std_reward_agent_{i}']      = std_r[:, i]
            df[f'mean_cum_reward_agent_{i}'] = mean_cr[:, i]
            df[f'std_cum_reward_agent_{i}']  = std_cr[:, i]

        # 5) Calcul des proportions d'actions
        for a in range(len(matrices[0][0])):
            prop = np.mean(plays_arr == a, axis=2)  # shape (n_steps, n_agents)
            for i in range(self.n_agents):
                df[f'prop_action_{a}_agent_{i}'] = prop[:, i]

        df.to_csv(f'Workshop/Data/{self.title}.csv', index=False)
        return df

    def getBanditResult(self, win_rate, use_rand_win, algo):

        # Julien: probalement à revoir, j'ai fait beaucoup de changements.

        experiments = []
        for e in tqdm(range(0, self.n_instance)):
            # define the random win rate for the current instance if asked - index 0 should have the biggest value
            if use_rand_win is True:
                rand_win_rate = np.random.rand(2)
                while rand_win_rate[0] <= rand_win_rate[1]:
                    rand_win_rate = np.random.rand(2)

            env = EnvBandit(self.n_agents, win_rate if use_rand_win is False else rand_win_rate)

            for i in range(0, self.n_agents):
                a_space = AgentSpace(len(win_rate), self.n_agents, 'Bandit', i+1)
                learning_algo = LearningAlgo(self.const, algo, a_space)
                env.ajouter_agents(Agent(a_space, learning_algo))

            for t in range(0, self.T):
                env.step()

            experiments.append(env.agents[0].cumul_regret)

        # if more than 1 instance the key "agents" returns the cumul_regret of agents in the last instance
        return { 'avg': pd.DataFrame(experiments).mean(), 'std': pd.DataFrame(experiments).std(), 'agents': env.agents }
