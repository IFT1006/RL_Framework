import pandas as pd
import numpy as np

from agentSpace import AgentSpace
from learningAlgo import LearningAlgo
from agent import Agent
from environment import Environnement
from utils import normalizeMatrix

class Execute:
    def __init__(self, n_instance, T, n_agents, const, title):
        self.n_instance = n_instance
        self.T = T
        self.n_agents = n_agents
        self.const = const
        self.title = title

    def runOnePDExperiment(self, matrices, algo, noise_dist, noise_params):

        # Initialisation de l'environnement
        env = Environnement(matrices, noise_dist, noise_params)
        for agent in range(0, self.n_agents):
            a_space = AgentSpace(len(matrices[0][0]))
            learning_algo = LearningAlgo(self.const[agent], algo[agent], a_space, noise_params[1])
            env.ajouter_agents(Agent(a_space, learning_algo))
        
        plays = []
        for time_step in range(0, self.T):
            actions = env.step()
            plays.append(actions)

        # Il faudrait enlever cumul_reward et le calucler ailleurs
        actions_played = [[i[0] for i in plays],[i[1] for i in plays]]
        cumul_rewards = [env.agents[k].cumul_reward for k in range(self.n_agents)]
        rewards = [env.agents[k].reward for k in range(self.n_agents)]

        # TODO - Audrey veut qu'on ait regrets plutôt comme c'est plus pertinente comme la mesure
        return actions_played, cumul_rewards, rewards
    

    def getPDResult(self, matrices, algo, noise_dist='uniform', noise_params=(0.0, 0.05)):
        # Normalisation de matrices
        matrices_norm = [normalizeMatrix(mat,0) for mat in matrices]

        # Boucle sur les itérations
        all_rewards = []
        all_cum_rewards = []
        all_plays = []

        for realisation in range(0, self.n_instance): #for realisation in tqdm(range(0, self.n_instance)):
            plays, cumul_rewards, rewards = self.runOnePDExperiment(matrices_norm, algo, noise_dist, noise_params)
            all_plays.append(np.array(plays).T)
            all_rewards.append(np.array(rewards).T)
            all_cum_rewards.append (np.array(cumul_rewards).T)

        # 2) Empilement en tableaux 3D : (timesteps, n_agents, n_instance)
        plays_arr = np.stack(all_plays, axis=2)
        rewards_arr  = np.stack(all_rewards, axis=2)
        cum_rewards_arr = np.stack(all_cum_rewards, axis=2) # À supprimer

        # Julien : Attention au 1
        inst_regrets_arr = 1.0 - rewards_arr 
        cum_regrets_arr = inst_regrets_arr.cumsum(axis=0) 
        cum_rewards_arr = rewards_arr.cumsum(axis=0)

        # 3) Calcul de la moyenne et de l’écart-type le long de l’axe réalisations
        mean_r     = rewards_arr.mean(axis=2)
        std_r      = rewards_arr.std (axis=2)
        mean_cr    = cum_rewards_arr.mean(axis=2)
        std_cr     = cum_rewards_arr.std (axis=2)
        mean_inst_regrets = inst_regrets_arr.mean(axis=2)
        std_inst_regrets = inst_regrets_arr.std (axis=2)
        mean_cum_regrets = cum_regrets_arr.mean(axis=2)
        std_cum_regrets = cum_regrets_arr.std (axis=2)

        # 4) Construction du DataFrame résumé
        n_steps = mean_r.shape[0]
        df = pd.DataFrame({'step': np.arange(n_steps)})
        for i in range(self.n_agents):
            df[f'mean_reward_agent_{i}']     = mean_r[:, i]
            df[f'std_reward_agent_{i}']      = std_r[:, i]
            df[f'mean_cum_reward_agent_{i}'] = mean_cr[:, i]
            df[f'std_cum_reward_agent_{i}']  = std_cr[:, i]
            df[f'mean_inst_regret_agent_{i}']  = mean_inst_regrets[:, i]
            df[f'std_inst_regret_agent_{i}']  = std_inst_regrets[:, i]
            df[f'mean_cum_regret_agent_{i}']  = mean_cum_regrets[:, i]
            df[f'std_cum_regret_agent_{i}']  = std_cum_regrets[:, i]

        # 5) Calcul des proportions d'actions
        for a in range(len(matrices[0][0])):
            prop = np.mean(plays_arr == a, axis=2)  # shape (n_steps, n_agents)
            for i in range(self.n_agents):
                df[f'prop_action_{a}_agent_{i}'] = prop[:, i]

        df.to_csv(f'Workshop/Data/{self.title}.csv', index=False)
        return df
