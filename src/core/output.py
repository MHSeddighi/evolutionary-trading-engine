"""Optimiser console output: objective columns only (fold scores stay inside
the ROB objective — no per-generation fold visibility)."""
import time
import numpy as np
from pymoo.util.display.column import Column
from pymoo.util.display.output import Output


class DynamicOutput(Output):

    def __init__(self):
        super().__init__()
        self.n_gen = Column("gen", width=5)
        self.n_eval = Column("evals", width=7)
        self.nds = Column("nds", width=5)
        self.time_col = Column("time", width=7)
        self._obj_cols = []
        self._last_time = None

    def initialize(self, algorithm):
        problem = algorithm.problem
        objectives = problem.OBJECTIVES

        for obj in objectives:
            self._obj_cols.append(Column(f"{obj.display}", width=18))

        self.columns = [self.n_gen, self.n_eval, self.nds]
        for col in self._obj_cols:
            self.columns.append(col)
        self.columns.append(self.time_col)

        self._last_time = time.perf_counter()

    def update(self, algorithm):
        super().update(algorithm)

        self.n_gen.set(algorithm.n_gen)
        self.n_eval.set(algorithm.evaluator.n_eval)

        pop = algorithm.pop
        F = pop.get("F")

        if F is not None and len(F) > 0:
            for i in range(algorithm.problem.n_obj):
                f_vals = F[:, i]
                best = np.min(f_vals)
                avg = np.mean(f_vals)

                self._obj_cols[i].set(f"{best:.4f} | {avg:.4f}")
        else:
            for col in self._obj_cols:
                col.set("-")

        self.nds.set(len(algorithm.opt))

        now = time.perf_counter()
        elapsed = now - self._last_time if self._last_time else 0.0
        self.time_col.set(f"{elapsed:.1f}s")
        self._last_time = now
