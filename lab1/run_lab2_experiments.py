"""
run_lab2_experiments.py — Главный скрипт исследования Lab 2.

Запуск из директории lab1/:
    /usr/local/bin/python3.12 run_lab2_experiments.py

Архитектурное решение по производительности:
    - Градиентные методы (GD, Momentum, Adam) используют дерево ConstructiveNumber
      для точного символьного дифференцирования.
    - Kiefer-Wolfowitz и Nelder-Mead работают только с вычислением значений функции.
      Для них используется быстрый NumPy-враппер, что ускоряет работу в 100-1000 раз.
    - PRECISION намеренно снижен до 15 цифр для разумного времени выполнения.
"""

import decimal
import math
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from Core.constructive_numbers import CNVariable
from Core.functions import (
    get_rastrigin_nd,
    get_rosenbrock,
)
from Optimization.optimizers import (
    AdamOptimizer,
    GradientDescent,
    MomentumGradientDescent,
)
from Visuals.visualizer import OptimizationVisualizer

# ─── Константы ───────────────────────────────────────────────────────────────
PRECISION = 15
RESULTS_DIR = "Results_Lab2"
SEPARATOR = "=" * 80

decimal.getcontext().prec = PRECISION


# ─── NumPy-версии функций (для визуализации) ─────────────────────────────────


def rosenbrock_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (1.0 - x) ** 2 + 100.0 * (y - x**2) ** 2


def rastrigin_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (
        20.0 + x**2 - 10.0 * np.cos(2 * np.pi * x) + y**2 - 10.0 * np.cos(2 * np.pi * y)
    )


def desmos_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    rsy = np.round(np.sin(10.0 * y)) + 2.0
    rsx = np.round(np.sin(7.0 * x)) + 2.0
    return ((x * rsy) ** 2 + y - 10.0) ** 2 + (x + (y * rsx) ** 2 - 7.0) ** 2


# ─── Быстрые NumPy-обёртки для N-мерных функций ──────────────────────────────


def rosenbrock_nd_np(point: List[float]) -> float:
    x = np.array(point)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def rastrigin_nd_np(point: List[float]) -> float:
    x = np.array(point)
    n = len(x)
    return float(10.0 * n + np.sum(x**2 - 10.0 * np.cos(2 * np.pi * x)))


def desmos_nd_np(point: List[float]) -> float:
    x = np.array(point)
    n = len(x)
    total = 0.0
    for i in range(n):
        xi = float(x[i])
        xn = float(x[(i + 1) % n])
        A = round(math.sin(10.0 * xn)) + 2.0
        B = round(math.sin(7.0 * xi)) + 2.0
        total += (xi * A + xn - 10.0) ** 2 + (xi + xn * B - 7.0) ** 2
    return total


# ─── Быстрый оптимизатор Нелдера-Мида на NumPy ───────────────────────────────


class FastNelderMead:
    """
    Нелдер-Мид на чистом Python/NumPy — не использует ConstructiveNumber.
    Возвращает словарь того же формата, что BaseOptimizer.optimize().
    """

    def __init__(
        self,
        func: Callable[[List[float]], float],
        max_iter: int = 2000,
        tolerance: float = 1e-8,
        alpha: float = 1.0,
        gamma: float = 2.0,
        rho: float = 0.5,
        sigma: float = 0.5,
        initial_step: float = 0.5,
    ) -> None:
        self.func = func
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.sigma = sigma
        self.initial_step = initial_step

    def optimize(self, start: List[float], verbose: bool = False) -> Dict:
        n = len(start)
        simplex = [list(start)]
        for i in range(n):
            v = list(start)
            v[i] += self.initial_step
            simplex.append(v)

        f_vals = [self.func(v) for v in simplex]
        n_evals = len(simplex)
        point_history = [list(start)]
        loss_history = [f_vals[0]]

        for iteration in range(1, self.max_iter + 1):
            order = sorted(range(len(simplex)), key=lambda i: f_vals[i])
            simplex = [simplex[i] for i in order]
            f_vals = [f_vals[i] for i in order]

            mean_f = sum(f_vals) / len(f_vals)
            std_f = (sum((fv - mean_f) ** 2 for fv in f_vals) / len(f_vals)) ** 0.5
            if std_f < self.tolerance:
                if verbose:
                    print(f"  [NM] Сходится на итерации {iteration}")
                break

            # Центроид (без наихудшей вершины)
            c = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]

            # Отражение
            x_r = [c[j] + self.alpha * (c[j] - simplex[-1][j]) for j in range(n)]
            f_r = self.func(x_r)
            n_evals += 1

            if f_vals[0] <= f_r < f_vals[-2]:
                simplex[-1] = x_r
                f_vals[-1] = f_r
            elif f_r < f_vals[0]:
                # Расширение
                x_e = [c[j] + self.gamma * (x_r[j] - c[j]) for j in range(n)]
                f_e = self.func(x_e)
                n_evals += 1
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_vals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
            else:
                # Сжатие
                x_ct = [c[j] + self.rho * (simplex[-1][j] - c[j]) for j in range(n)]
                f_ct = self.func(x_ct)
                n_evals += 1
                if f_ct < f_vals[-1]:
                    simplex[-1] = x_ct
                    f_vals[-1] = f_ct
                else:
                    # Редукция
                    best = simplex[0]
                    simplex = [best] + [
                        [
                            best[j] + self.sigma * (simplex[i][j] - best[j])
                            for j in range(n)
                        ]
                        for i in range(1, len(simplex))
                    ]
                    f_vals = [self.func(v) for v in simplex]
                    n_evals += len(simplex)

            point_history.append(list(simplex[0]))
            loss_history.append(f_vals[0])

            if verbose and iteration % 200 == 0:
                print(f"  [NM] iter={iteration} f={f_vals[0]:.4e} std={std_f:.2e}")

        best = simplex[0]
        return {
            "final_point": [decimal.Decimal(str(v)) for v in best],
            "point_history": point_history,
            "loss_history": loss_history,
            "n_func_evals": n_evals,
        }


# ─── Быстрый KW на NumPy ─────────────────────────────────────────────────────


class FastKieferWolfowitz:
    """
    Kiefer-Wolfowitz на NumPy-функции — не использует ConstructiveNumber.
    """

    def __init__(
        self,
        func: Callable[[List[float]], float],
        learning_rate: float = 0.05,
        c: float = 0.01,
        max_iter: int = 2000,
        tolerance: float = 1e-6,
        clip_grad: float = 20.0,
    ) -> None:
        self.func = func
        self.lr = learning_rate
        self.c = c
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.clip_grad = clip_grad

    def optimize(self, start: List[float], verbose: bool = False) -> Dict:
        x = list(start)
        n = len(x)
        f0 = self.func(x)
        point_history = [list(x)]
        loss_history = [f0]
        n_evals = 1

        for t in range(1, self.max_iter + 1):
            c_t = self.c / (t**0.25)
            a_t = self.lr / t

            grads = []
            for i in range(n):
                xp = list(x)
                xp[i] += c_t
                xm = list(x)
                xm[i] -= c_t
                g = (self.func(xp) - self.func(xm)) / (2.0 * c_t)
                # Clipping для стабильности
                g = max(-self.clip_grad, min(self.clip_grad, g))
                grads.append(g)
                n_evals += 2

            max_g = max(abs(g) for g in grads)
            if max_g < self.tolerance:
                if verbose:
                    print(f"  [KW] Сходится на итерации {t}")
                break

            x_new = [x[i] - a_t * grads[i] for i in range(n)]
            fval = self.func(x_new)
            n_evals += 1

            # Защита от взрыва: откат если значение выросло катастрофически
            if not math.isfinite(fval) or fval > loss_history[-1] * 1e8 + 1e10:
                if verbose:
                    print(f"  [KW] iter={t}: откат (f={fval:.2e})")
                break

            x = x_new
            point_history.append(list(x))
            loss_history.append(fval)

            if verbose and t % 200 == 0:
                print(f"  [KW] iter={t} f={fval:.4e} |g|={max_g:.2e}")

        return {
            "final_point": [decimal.Decimal(str(v)) for v in x],
            "point_history": point_history,
            "loss_history": loss_history,
            "n_func_evals": n_evals,
        }


# ─── Вывод таблицы ───────────────────────────────────────────────────────────


def print_comparison_table(
    experiment_name: str,
    results: Dict[str, Dict],
    true_optimum: Optional[List[float]] = None,
) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  РЕЗУЛЬТАТЫ: {experiment_name}")
    print(SEPARATOR)

    col_w = [22, 10, 12, 16, 16, 12]
    headers = [
        "Алгоритм",
        "Итерации",
        "Вызовов f",
        "f(x*)",
        "||x*-x_opt||",
        "Время (с)",
    ]
    divider = "+".join("-" * w for w in col_w)

    print(divider)
    print("".join(f"{h:^{w}}" for h, w in zip(headers, col_w)))
    print(divider)

    for name, data in results.items():
        iters = len(data["loss_history"]) - 1
        n_evals = data.get("n_func_evals", iters)
        final_f = float(data["loss_history"][-1])
        elapsed = data.get("elapsed", float("nan"))

        if true_optimum is not None:
            fp = [float(v) for v in data["final_point"]]
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(fp, true_optimum)))
            dist_str = f"{dist:.4e}"
        else:
            dist_str = "N/A"

        row = [
            name[:20],
            str(iters),
            str(n_evals),
            f"{final_f:.4e}",
            dist_str,
            f"{elapsed:.2f}",
        ]
        print("".join(f"{v:^{w}}" for v, w in zip(row, col_w)))

    print(divider)


# ─── Ядро эксперимента ────────────────────────────────────────────────────────


def run_gradient_optimizers(
    func_builder: Callable,
    optimizers_cfg: Dict,
    n_dims: int,
    start_point: List[float],
) -> Dict[str, Dict]:
    """Запускает градиентные оптимизаторы через ConstructiveNumber."""
    results = {}
    for opt_name, optimizer in optimizers_cfg.items():
        variables = [
            CNVariable(name=f"x{i}", initial_val=str(start_point[i]))
            for i in range(n_dims)
        ]
        target_tree = func_builder(variables)
        t0 = time.perf_counter()
        result = optimizer.optimize(target_tree, variables, verbose=False)
        elapsed = time.perf_counter() - t0
        result["elapsed"] = elapsed
        if "n_func_evals" not in result:
            result["n_func_evals"] = len(result["loss_history"]) - 1
        results[opt_name] = result
        print(
            f"  → {opt_name}: f={float(result['loss_history'][-1]):.4e}  t={elapsed:.2f}s"
        )
    return results


def run_fast_optimizers(
    fast_opts: Dict,
    start_point: List[float],
) -> Dict[str, Dict]:
    """Запускает быстрые оптимизаторы (FastNelderMead, FastKieferWolfowitz)."""
    results = {}
    for opt_name, optimizer in fast_opts.items():
        t0 = time.perf_counter()
        result = optimizer.optimize(start_point, verbose=False)
        elapsed = time.perf_counter() - t0
        result["elapsed"] = elapsed
        results[opt_name] = result
        print(
            f"  → {opt_name}: f={float(result['loss_history'][-1]):.4e}  t={elapsed:.2f}s"
        )
    return results


def run_experiment(
    experiment_name: str,
    func_builder: Optional[Callable],
    func_np: Callable,
    gradient_opts: Dict,
    fast_opts: Dict,
    n_dims: int,
    start_point: List[float],
    true_optimum: Optional[List[float]],
    viz: OptimizationVisualizer,
    x_range: Tuple[float, float] = (-5.0, 5.0),
    y_range: Tuple[float, float] = (-5.0, 5.0),
    plot_3d: bool = True,
) -> Dict[str, Dict]:
    print(f"\n{'─' * 60}")
    print(f"  Эксперимент: {experiment_name}  ({n_dims}D)")
    print(f"{'─' * 60}")

    if plot_3d and n_dims == 2:
        viz.plot_3d_surface(
            func_np=func_np,
            x_range=x_range,
            y_range=y_range,
            filename=f"{experiment_name}_surface_3d.png",
        )

    all_results: Dict[str, Dict] = {}

    if gradient_opts and func_builder is not None:
        all_results.update(
            run_gradient_optimizers(func_builder, gradient_opts, n_dims, start_point)
        )

    if fast_opts:
        all_results.update(run_fast_optimizers(fast_opts, start_point))

    print_comparison_table(experiment_name, all_results, true_optimum)

    prefix = experiment_name.lower().replace(" ", "_")

    viz.plot_convergence_comparison(
        results=all_results,
        filename=f"{prefix}_convergence.png",
        title=f"Сходимость — {experiment_name}",
    )

    if n_dims == 2:
        viz.plot_paths_comparison(
            results=all_results,
            func_np=func_np,
            filename=f"{prefix}_paths.png",
            title=f"Траектории — {experiment_name}",
        )

    return all_results


# ─── ГЛАВНАЯ ФУНКЦИЯ ──────────────────────────────────────────────────────────


def main() -> None:
    print(SEPARATOR)
    print("  LAB 2 — Исследование алгоритмов оптимизации")
    print("  Функции: Розенброк, Растригин, Desmos")
    print("  Алгоритмы: Vanilla GD, Momentum, Adam, Kiefer-Wolfowitz, Nelder-Mead")
    print(SEPARATOR)

    viz = OptimizationVisualizer(save_dir=RESULTS_DIR)

    # ══════════════════════════════════════════════════════════════════════════
    # БЛОК 1: ФУНКЦИЯ РОЗЕНБРОКА
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{SEPARATOR}")
    print("  БЛОК 1: ФУНКЦИЯ РОЗЕНБРОКА")
    print(SEPARATOR)

    run_experiment(
        experiment_name="Rosenbrock_2D",
        func_builder=get_rosenbrock,
        func_np=rosenbrock_np,
        gradient_opts={
            "Vanilla GD": GradientDescent(
                learning_rate=0.002,
                max_iter=1500,
                tolerance=1e-6,
                precision_digits=PRECISION,
                clip_value=100.0,
            ),
            "Momentum GD": MomentumGradientDescent(
                learning_rate=0.01,
                momentum=0.9,
                max_iter=1500,
                tolerance=1e-6,
                precision_digits=PRECISION,
            ),
            "Adam": AdamOptimizer(
                learning_rate=0.005,
                max_iter=1500,
                tolerance=1e-6,
                precision_digits=PRECISION,
            ),
        },
        fast_opts={
            "Kiefer-Wolfowitz": FastKieferWolfowitz(
                func=rosenbrock_nd_np,
                learning_rate=0.1,
                c=0.05,
                max_iter=5000,
                tolerance=1e-7,
                clip_grad=50.0,
            ),
            "Nelder-Mead": FastNelderMead(
                func=rosenbrock_nd_np,
                max_iter=2000,
                tolerance=1e-8,
                initial_step=0.5,
            ),
        },
        n_dims=2,
        start_point=[-1.5, 1.5],
        true_optimum=[1.0, 1.0],
        viz=viz,
        x_range=(-2.5, 2.5),
        y_range=(-1.0, 3.5),
    )

    run_experiment(
        experiment_name="Rosenbrock_5D",
        func_builder=get_rosenbrock,
        func_np=rosenbrock_np,
        gradient_opts={
            "Adam": AdamOptimizer(
                learning_rate=0.005,
                max_iter=1500,
                tolerance=1e-6,
                precision_digits=PRECISION,
            ),
            "Momentum GD": MomentumGradientDescent(
                learning_rate=0.005,
                momentum=0.9,
                max_iter=1500,
                tolerance=1e-6,
                precision_digits=PRECISION,
            ),
        },
        fast_opts={
            "Nelder-Mead": FastNelderMead(
                func=rosenbrock_nd_np,
                max_iter=5000,
                tolerance=1e-8,
                initial_step=0.5,
            ),
        },
        n_dims=5,
        start_point=[-1.0] * 5,
        true_optimum=[1.0] * 5,
        viz=viz,
        plot_3d=False,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # БЛОК 2: ФУНКЦИЯ РАСТРИГИНА
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{SEPARATOR}")
    print("  БЛОК 2: ФУНКЦИЯ РАСТРИГИНА")
    print(SEPARATOR)

    run_experiment(
        experiment_name="Rastrigin_2D",
        func_builder=get_rastrigin_nd,
        func_np=rastrigin_np,
        gradient_opts={
            "Vanilla GD": GradientDescent(
                learning_rate=0.01,
                max_iter=1500,
                tolerance=1e-6,
                precision_digits=PRECISION,
                clip_value=50.0,
            ),
            "Momentum GD": MomentumGradientDescent(
                learning_rate=0.01,
                momentum=0.9,
                max_iter=1500,
                tolerance=1e-6,
                precision_digits=PRECISION,
            ),
            "Adam": AdamOptimizer(
                learning_rate=0.01,
                max_iter=1500,
                tolerance=1e-6,
                precision_digits=PRECISION,
            ),
        },
        fast_opts={
            "Kiefer-Wolfowitz": FastKieferWolfowitz(
                func=rastrigin_nd_np,
                learning_rate=0.1,
                c=0.05,
                max_iter=2000,
                tolerance=1e-6,
            ),
            "Nelder-Mead": FastNelderMead(
                func=rastrigin_nd_np,
                max_iter=2000,
                tolerance=1e-8,
                initial_step=1.0,
            ),
        },
        n_dims=2,
        start_point=[2.5, 2.5],
        true_optimum=[0.0, 0.0],
        viz=viz,
        x_range=(-5.5, 5.5),
        y_range=(-5.5, 5.5),
    )

    run_experiment(
        experiment_name="Rastrigin_5D",
        func_builder=get_rastrigin_nd,
        func_np=rastrigin_np,
        gradient_opts={
            "Adam": AdamOptimizer(
                learning_rate=0.01,
                max_iter=1500,
                tolerance=1e-6,
                precision_digits=PRECISION,
            ),
            "Momentum GD": MomentumGradientDescent(
                learning_rate=0.005,
                momentum=0.9,
                max_iter=1500,
                tolerance=1e-6,
                precision_digits=PRECISION,
            ),
        },
        fast_opts={
            "Nelder-Mead": FastNelderMead(
                func=rastrigin_nd_np,
                max_iter=5000,
                tolerance=1e-8,
                initial_step=1.0,
            ),
        },
        n_dims=5,
        start_point=[2.0] * 5,
        true_optimum=[0.0] * 5,
        viz=viz,
        plot_3d=False,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # БЛОК 3: ФУНКЦИЯ DESMOS (негладкая)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{SEPARATOR}")
    print("  БЛОК 3: ФУНКЦИЯ DESMOS (негладкая, round(sin(...)))")
    print(SEPARATOR)

    print("  [Info] Grid search для приближённого минимума Desmos 2D...")
    xs = np.linspace(-4.0, 4.0, 400)
    ys = np.linspace(-4.0, 4.0, 400)
    XG, YG = np.meshgrid(xs, ys)
    ZG = desmos_np(XG, YG)
    # Ищем глобальный минимум по всей сетке
    flat_sorted = np.argsort(ZG.ravel())
    best_val = float("inf")
    desmos_approx_opt = [0.0, 0.0]
    for idx in flat_sorted[:20]:
        r, c = np.unravel_index(idx, ZG.shape)
        if float(ZG[r, c]) < best_val:
            best_val = float(ZG[r, c])
            desmos_approx_opt = [float(XG[r, c]), float(YG[r, c])]
    print(f"  [Info] Приближённый минимум: {desmos_approx_opt}, f={best_val:.4e}")

    # Уточняем минимум Нелдером-Мидом из найденной точки
    _nm_refine = FastNelderMead(
        func=desmos_nd_np, max_iter=5000, tolerance=1e-12, initial_step=0.3
    )
    _refined = _nm_refine.optimize(desmos_approx_opt)
    desmos_approx_opt = [float(v) for v in _refined["final_point"]]
    print(
        f"  [Info] Уточнённый минимум:   {desmos_approx_opt}, f={_refined['loss_history'][-1]:.4e}"
    )

    # 2D Desmos — только безградиентные методы
    run_experiment(
        experiment_name="Desmos_2D",
        func_builder=None,
        func_np=desmos_np,
        gradient_opts={},
        fast_opts={
            "Kiefer-Wolfowitz": FastKieferWolfowitz(
                func=desmos_nd_np,
                learning_rate=0.05,
                c=0.02,
                max_iter=5000,
                tolerance=1e-8,
                clip_grad=5.0,
            ),
            "Nelder-Mead": FastNelderMead(
                func=desmos_nd_np,
                max_iter=5000,
                tolerance=1e-10,
                initial_step=0.8,
            ),
        },
        n_dims=2,
        start_point=desmos_approx_opt,
        true_optimum=desmos_approx_opt,
        viz=viz,
        x_range=(-4.0, 4.0),
        y_range=(-4.0, 4.0),
    )

    # 4D Desmos
    run_experiment(
        experiment_name="Desmos_4D",
        func_builder=None,
        func_np=desmos_np,
        gradient_opts={},
        fast_opts={
            "Kiefer-Wolfowitz": FastKieferWolfowitz(
                func=desmos_nd_np,
                learning_rate=0.05,
                c=0.02,
                max_iter=5000,
                tolerance=1e-8,
                clip_grad=5.0,
            ),
            "Nelder-Mead": FastNelderMead(
                func=desmos_nd_np,
                max_iter=5000,
                tolerance=1e-10,
                initial_step=0.8,
            ),
        },
        n_dims=4,
        start_point=[1.0, 0.8, 1.0, 0.8],
        true_optimum=None,
        viz=viz,
        plot_3d=False,
    )

    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{SEPARATOR}")
    print("  ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ")
    print(f"  Графики сохранены в: {os.path.abspath(RESULTS_DIR)}")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
