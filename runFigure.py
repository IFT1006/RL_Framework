from run import *

with open('data.json', 'r') as json_file:
    loaded_results = json.load(json_file)

for game in tqdm(games.keys()):
    plot_results(game, loaded_results, noise_levels, algo_pairs, save_folder="Workshop/Test")
    plot_results_action(game, results, noise_levels, algo_pairs, 500, save_folder="Workshop/Figure25")
    plot_results_exploration(game, results, noise_levels, algo_pairs, save_folder="Workshop/Figure25")
