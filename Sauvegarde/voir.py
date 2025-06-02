import numpy
from matplotlib import pyplot
numpy.set_printoptions(precision=3, suppress=True)


class SingleMatrixGame:

    def __init__(self, matrix, noise_var, seed=None):
        self.matrix = matrix
        self.noise_std = numpy.sqrt(noise_var)
        self.random = numpy.random.RandomState(seed)

        self.a_star = numpy.unravel_index(numpy.argmax(matrix), matrix.shape)
        self.gaps = matrix[self.a_star] - matrix

        self.history = {"actions": [], "gaps": []}

    def play(self, action1, action2):
        self.history["actions"].append((action1, action2))
        self.history["gaps"].append(self.gaps[action1, action2])

        noises = self.random.normal(0, self.noise_std, 2)
        rewards = noises + self.matrix[action1, action2]
        return rewards[0], rewards[1]


class Player:
    def __init__(self, nb_actions, seed):
        self.nb_actions = nb_actions
        self.random = numpy.random.RandomState(seed)
        self.sums = numpy.zeros(nb_actions)
        self.counts = numpy.zeros(nb_actions, dtype=int)
        self.counts_other = numpy.zeros(nb_actions, dtype=int)

    def select(self):
        pass

    def update(self, action, action_other, reward):
        self.sums[action] += reward
        self.counts[action] += 1
        self.counts_other[action_other] += 1


class UCB1(Player):
    def __init__(self, nb_actions, seed=None):
        super().__init__(nb_actions, seed)
        self.init_sequence = self.random.permutation(nb_actions)

    def select(self, timestep, verbose=False):
        index_t = timestep - 1
        if index_t < self.nb_actions:
            action = self.init_sequence[index_t]
        else:
            means = self.sums / self.counts
            ucbs = means + numpy.sqrt(2 * numpy.log(timestep) / self.counts)

            best = numpy.flatnonzero(ucbs == ucbs.max())
            if best.size == 1:
                action = int(best[0])
            else:
                action = int(numpy.random.choice(best))
        return action


class KLUCB(Player):
    def __init__(self, nb_actions, var, c=3, seed=None):
        super().__init__(nb_actions, seed)
        self.var = var
        self.c = c
        self.init_sequence = self.random.permutation(nb_actions)

    def select(self, timestep, verbose=False):
        index_t = timestep - 1
        if index_t < self.nb_actions:
            action = self.init_sequence[index_t]
        else:
            means = self.sums / self.counts
            f_t = 2 * self.var * (numpy.log(timestep) + self.c * numpy.log(numpy.log(timestep)))
            ucbs = means + numpy.sqrt(f_t / self.counts)
            best = numpy.flatnonzero(ucbs == ucbs.max())
            if best.size == 1:
                action = int(best[0])
            else:
                action = int(numpy.random.choice(best))
        return action


class TargetUCB(UCB1):
    def select(self, timestep, verbose=False):
        index_t = timestep - 1
        if index_t < self.nb_actions:
            action = self.init_sequence[index_t]
        else:
            # print("selecting action")
            # print("sums", self.sums, "counts", self.counts)
            means = self.sums / self.counts
            est_opt = numpy.sqrt(3/2 * numpy.log(timestep) / self.counts)
            # est_opt = numpy.sqrt(5/2 * numpy.log(timestep) / self.counts) # to account for 1/2 noise + [0, 1] bounded reward
            ratio_counts = (self.counts_other - self.counts) / self.counts_other
            target_opt = numpy.sqrt(ratio_counts.clip(min=0))
            # print("target_opt", target_opt)
            ucbs = means + est_opt * target_opt
            if verbose:
                print("counts", self.counts, "counts_other", self.counts_other)
                print("means", means, "ucbs", ucbs)
            action = numpy.argmax(ucbs)
        return action


class TS(Player):
    def __init__(self, nb_actions, mu_0, var_0, var, seed):
        super().__init__(nb_actions, seed)
        self.mu_0 = mu_0
        self.var_0 = var_0
        self.var = var
        self.init_sequence = self.random.permutation(nb_actions)

    def select(self, timestep, verbose=False):
        index_t = timestep - 1
        if index_t < self.nb_actions:
            action = self.init_sequence[index_t]
        else:
            mu_post = (self.mu_0 / self.var_0 + self.sums / self.var) / (1 / self.var_0 + self.counts / self.var)
            var_post = 1 / (1 / self.var_0 + self.counts / self.var)
            samples = self.random.normal(mu_post, numpy.sqrt(var_post))
            action = numpy.argmax(samples)
        return action


class EpsilonGreedy(Player):
    def __init__(self, nb_actions, epsilon=None, seed=None):
        super().__init__(nb_actions, seed)
        self.epsilon = epsilon
        self.init_sequence = self.random.permutation(nb_actions)

    def select(self, timestep, verbose=False):
        index_t = timestep - 1
        if index_t < self.nb_actions:
            action = self.init_sequence[index_t]
        else:
            if self.epsilon is None:
                epsilon = 1 / numpy.sqrt(timestep)
            else:
                epsilon = self.epsilon
            if self.random.rand() < epsilon:
                action = self.random.choice(self.nb_actions)
            else:
                means = self.sums / self.counts
                action = numpy.argmax(means)
        return action


def run(game, player1, player2, T):
    for t in range(1, T):
        # verbose = (t==T-1)
        i_t = player1.select(t, verbose=False)
        j_t = player2.select(t, verbose=False)
        r1_t, r2_t = game.play(i_t, j_t)
        # print("t", t, "i_t", i_t, "j_t", j_t, "r1_t", r1_t, "r2_t", r2_t)
        player1.update(i_t, j_t, r1_t)
        player2.update(j_t, i_t, r2_t)

def run_ts_ts(matrix, noise_var, T, N):
    # TS vs TS
    nb_actions1, nb_actions2 = matrix.shape
    cumul_regrets = []
    for i in range(N):
        game = SingleMatrixGame(matrix, noise_var, i)
        player1, player2 = TS(nb_actions1, 1, 1, max(noise_var, 1e-2) , seed=i+1), TS(nb_actions2, 1, 1, max(noise_var, 1e-2), seed=i+2)
        run(game, player1, player2, T)
        cumul_regrets.append(numpy.cumsum(game.history["gaps"]))
    return cumul_regrets


def run_ucb_ucb(matrix, noise_var, T, N):
    #UCB vs UCB
    nb_actions1, nb_actions2 = matrix.shape
    cumul_regrets = []
    for i in range(N):
        game = SingleMatrixGame(matrix, noise_var, i)
        player1, player2 = UCB1(nb_actions1, seed=i+1), UCB1(nb_actions2, seed=i+2)
        run(game, player1, player2, T)
        cumul_regrets.append(numpy.cumsum(game.history["gaps"]))
    return cumul_regrets


def run_klucb_klucb(matrix, noise_var, T, N):
    #KL-UCB vs KL-UCB
    nb_actions1, nb_actions2 = matrix.shape
    cumul_regrets = []
    for i in range(N):
        game = SingleMatrixGame(matrix, noise_var, i)
        player1, player2 = KLUCB(nb_actions1, max(noise_var, 1e-2), seed=i+1), KLUCB(nb_actions2, max(noise_var, 1e-2), seed=i+2)
        run(game, player1, player2, T)
        cumul_regrets.append(numpy.cumsum(game.history["gaps"]))
    return cumul_regrets


def run_ucb_ts(matrix, noise_var, T, N):
    # UCB vs TS
    nb_actions1, nb_actions2 = matrix.shape
    cumul_regrets = []
    for i in range(N):
        game = SingleMatrixGame(matrix, noise_var, i)
        player1, player2 = UCB1(nb_actions1, seed=i+1), TS(nb_actions2, 1, 1, max(noise_var, 1e-2), seed=i+2)
        run(game, player1, player2, T)
        cumul_regrets.append(numpy.cumsum(game.history["gaps"]))
    return cumul_regrets

def run_targetucb_ts(matrix, noise_var, T, N):
    # Target-UCB vs TS
    nb_actions1, nb_actions2 = matrix.shape
    cumul_regrets = []
    for i in range(N):
        game = SingleMatrixGame(matrix, noise_var, i)
        player1, player2 = TargetUCB(nb_actions1, seed=i+1), TS(nb_actions2, 1, 1, max(noise_var, 1e-2), seed=i+2)
        run(game, player1, player2, T)
        cumul_regrets.append(numpy.cumsum(game.history["gaps"]))
    return cumul_regrets

def run_targetucb_ucb(matrix, noise_var, T, N):
    # Target-UCB vs UCB
    nb_actions1, nb_actions2 = matrix.shape
    cumul_regrets = []
    for i in range(N):
        game = SingleMatrixGame(matrix, noise_var, i)
        player1, player2 = TargetUCB(nb_actions1, seed=i+1), UCB1(nb_actions2, seed=i+2)
        run(game, player1, player2, T)
        cumul_regrets.append(numpy.cumsum(game.history["gaps"]))
    return cumul_regrets


def run_targetucb_targetucb(matrix, noise_var, T, N):
    # Target-UCB vs TargetUCB
    nb_actions1, nb_actions2 = matrix.shape
    cumul_regrets = []
    for i in range(N):
        game = SingleMatrixGame(matrix, noise_var, i)
        player1, player2 = TargetUCB(nb_actions1, seed=i+1), TargetUCB(nb_actions2, seed=i+2)
        run(game, player1, player2, T)
        cumul_regrets.append(numpy.cumsum(game.history["gaps"]))
    return cumul_regrets

matrix = numpy.array([[1, 0], [0, 0.5]])

T = 101
N = 1000
results_simple = {}



# noise_var = 0
# results_simple["no_noise"] = {}
# results_simple["no_noise"]["UCB_UCB"] = run_ucb_ucb(matrix, noise_var, T, N)
# results_simple["no_noise"]["KLUCB_KLUCB"] = run_klucb_klucb(matrix, noise_var, T, N)
# results_simple["no_noise"]["TS_TS"] = run_ts_ts(matrix, noise_var, T, N)
# results_simple["no_noise"]["UCB_TS"] = run_ucb_ts(matrix, noise_var, T, N)

noise_var = 0.1
results_simple["low_noise"] = {}
results_simple["low_noise"]["UCB_UCB"] = run_ucb_ucb(matrix, noise_var, T, N)
results_simple["low_noise"]["KLUCB_KLUCB"] = run_klucb_klucb(matrix, noise_var, T, N)
results_simple["low_noise"]["TS_TS"] = run_ts_ts(matrix, noise_var, T, N)
results_simple["low_noise"]["UCB_TS"] = run_ucb_ts(matrix, noise_var, T, N)

# noise_var = 1
# results_simple["high_noise"] = {}
# results_simple["high_noise"]["UCB_UCB"] = run_ucb_ucb(matrix, noise_var, T, N)
# results_simple["high_noise"]["KLUCB_KLUCB"] = run_klucb_klucb(matrix, noise_var, T, N)
# results_simple["high_noise"]["TS_TS"] = run_ts_ts(matrix, noise_var, T, N)
# results_simple["high_noise"]["UCB_TS"] = run_ucb_ts(matrix, noise_var, T, N)

timehorizon = numpy.arange(1, T)
for config in ["UCB_UCB", "KLUCB_KLUCB", "TS_TS", "UCB_TS"]:
    cumul_regrets = results_simple["low_noise"][config]
    avg, std = numpy.average(cumul_regrets, axis=0), numpy.std(cumul_regrets, axis=0)
    pyplot.plot(timehorizon, avg, label=config)
    pyplot.fill_between(timehorizon, avg, avg+std, alpha=0.4)
pyplot.legend()
pyplot.show()
