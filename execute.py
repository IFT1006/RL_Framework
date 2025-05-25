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
        self.const = const
        self.title = title # Julien pt changer ça

    def runOnePDExperiment(self, matrices, algo, noise_dist='uniform', noise_params=(0.0, 0.05)):

        # Initialisation de l'environnement
        env = EnvPD(matrices, self.n_agents, noise_dist, noise_params)
        for agent in range(0, self.n_agents):
            a_space = AgentSpace(len(matrices[0]), self.n_agents, 'PD', agent+1)
            learning_algo = LearningAlgo(self.const, algo[agent], a_space)
            env.ajouter_agents(Agent(a_space, learning_algo))
        
        plays = []
        for time_step in range(0, self.T):
            actions = env.step()
            print(actions)
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
        for realisation in tqdm(range(0, self.n_instance)):
            plays, cumul_rewards, rewards = self.runOnePDExperiment(matrices_norm, algo, noise_dist, noise_params)

            # Enregistrement des résultats dans un Dataframe à faire
            for i in range(self.n_agents):
                df[f'action_agent_{i}_{algo[i]}_{realisation}'] = np.array(plays).T[:, i]
                df[f'reward_agent_{i}_{algo[i]}_{realisation}'] = np.array(cumul_rewards).T[:, i]
                df[f'cum_reward_agent_{i}_{algo[i]}_{realisation}'] = np.array(rewards).T[:, i]
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
