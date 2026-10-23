from pymoo.algorithms.base.genetic import Algorithm
from pymoo.core.callback import Callback


class HistorySaver:
    def __init__(self):
        self.history = []

    def __call__(self, algorithm: Algorithm):
        # Only save lightweight data
        snapshot = {
            'n_gen': algorithm.n_gen,
            'n_eval': algorithm.evaluator.n_eval,
            'X': algorithm.pop.get("X").copy(),
            'F': algorithm.pop.get("F").copy()
        }
        self.history.append(snapshot)


class GenUpdater(Callback):
    def __init__(self, problem, fallback_max_gen=250):
        super().__init__()
        self.problem = problem
        self.fallback_max_gen = fallback_max_gen

    def notify(self, algorithm):
        current_gen = algorithm.n_gen

        n_max = getattr(algorithm.termination, 'n_max_gen', self.fallback_max_gen)

        progress = current_gen / max(n_max, 1)
        self.problem.update_progress(min(progress, 1.0))

class CallbackCombiner(Callback):
    def __init__(self, problem, *callbacks):
        super().__init__()
        self.problem = problem
        self.callbacks = callbacks

    def notify(self, algorithm):
        self.problem.algorithm = algorithm
        for cb in self.callbacks:
            if hasattr(cb, 'notify'):
                cb.notify(algorithm)
            elif callable(cb):
                cb(algorithm)