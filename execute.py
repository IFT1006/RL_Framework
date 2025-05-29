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
        regrets = [env.agents[k].regret for k in range(self.n_agents)]
        rewards = [env.agents[k].reward for k in range(self.n_agents)]

        return actions_played, regrets, rewards
    

    def getPDResult(self, matrices, algo, noise_dist='normal', noise_params=(0, 0.05)):
        # Normalisation de matrices
        matrices_norm = [normalizeMatrix(mat,0) for mat in matrices]
        # Boucle sur les itérations
        all_rewards = []
        all_regrets = []
        all_plays = []

        for realisation in range(0, self.n_instance): #for realisation in tqdm(range(0, self.n_instance)):
            plays, regrets, rewards = self.runOnePDExperiment(matrices_norm, algo, noise_dist, noise_params)
            all_plays.append(np.array(plays).T)
            all_rewards.append(np.array(rewards).T)
            all_regrets.append (np.array(regrets).T)

        # 2) Empilement en tableaux 3D : (timesteps, n_agents, n_instance)
        plays_arr = np.stack(all_plays, axis=2)
        rewards_arr  = np.stack(all_rewards, axis=2)
        regrets_arr = np.stack(all_regrets, axis=2)
        cum_regrets_arr = regrets_arr.cumsum(axis=0)

        # 3) Calcul de la moyenne et de l’écart-type le long de l’axe réalisations
        mean_r     = rewards_arr.mean(axis=2)
        std_r      = rewards_arr.std (axis=2)
        mean_reg    = cum_regrets_arr.mean(axis=2)
        std_reg     = cum_regrets_arr.std (axis=2)


        results = {
        'experiment':   self.title,
        'algo':         algo,
        'noise_dist':   noise_dist,
        'noise_params': noise_params,
        'metrics': {}
    }

        for name, arr in [
            ('mean_reward',     mean_r),
            ('std_reward',      std_r),
            ('mean_cum_regret', mean_reg),
            ('std_cum_regret',  std_reg)
        ]:
            results['metrics'][name] = {
                f'agent_{i}': arr[:, i]
                for i in range(self.n_agents)
            }

        # 5) Optionnel : proportions d'actions
        props = {}
        for a in range(len(matrices[0][0])):
            prop = np.mean(plays_arr == a, axis=2)  # shape (n_steps, n_agents)
            props[f'action_{a}'] = {
                f'agent_{i}': prop[:, i]
                for i in range(self.n_agents)
            }
        results['metrics']['prop_action'] = props

        return results
