"""
Алгоритм Нелдера-Мида (метод деформируемого многогранника).

Метод не требует вычисления градиента и работает только с вычислением
значений функции, что делает его пригодным для негладких функций
(например, функции Desmos с операцией round).

Алгоритм оперирует симплексом из (n+1) вершин в n-мерном пространстве
и итеративно деформирует его, приближаясь к минимуму.

Операции симплекса:
    - Отражение (reflection):   x_r = x_c + α*(x_c - x_w)
    - Расширение (expansion):   x_e = x_c + γ*(x_r - x_c)
    - Сжатие (contraction):     x_ct = x_c + ρ*(x_w - x_c)
    - Редукция (shrink):        x_i = x_b + σ*(x_i - x_b)

Стандартные параметры (Nelder & Mead, 1965):
    α=1, γ=2, ρ=0.5, σ=0.5
"""

import decimal
from typing import Dict, List

from Core.constructive_numbers import CNVariable, ConstructiveNumber


class NelderMeadOptimizer:
    """
    Оптимизатор методом Нелдера-Мида.

    Не наследует BaseOptimizer, так как принципиально отличается
    по интерфейсу: не использует градиенты и работает с симплексом.
    Возвращает словарь того же формата, что и BaseOptimizer.optimize(),
    для совместимости с визуализатором.
    """

    def __init__(
        self,
        max_iter: int = 2000,
        tolerance: float = 1e-6,
        precision_digits: int = 25,
        alpha: float = 1.0,
        gamma: float = 2.0,
        rho: float = 0.5,
        sigma: float = 0.5,
        initial_step: float = 0.5,
    ) -> None:
        """
        Args:
            max_iter (int): Максимальное число итераций.
            tolerance (float): Критерий остановки по разбросу значений симплекса.
            precision_digits (int): Точность вычислений ConstructiveNumber.
            alpha (float): Коэффициент отражения.
            gamma (float): Коэффициент расширения.
            rho (float): Коэффициент сжатия.
            sigma (float): Коэффициент редукции.
            initial_step (float): Начальный размер симплекса.
        """
        self.max_iter = max_iter
        self.tolerance = decimal.Decimal(str(tolerance))
        self.precision_digits = precision_digits
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.sigma = sigma
        self.initial_step = initial_step

    def _eval(
        self,
        target_func: ConstructiveNumber,
        variables: List[CNVariable],
        point: List[float],
    ) -> float:
        """Вычисляет значение функции в заданной точке."""
        for var, val in zip(variables, point):
            var.set_val(str(val))
        return float(target_func.evaluate(self.precision_digits).middle)

    def _build_initial_simplex(
        self,
        start: List[float],
    ) -> List[List[float]]:
        """
        Строит начальный симплекс из (n+1) вершин.
        Первая вершина — стартовая точка, остальные — сдвинуты на initial_step
        по каждой оси.
        """
        n = len(start)
        simplex = [list(start)]
        for i in range(n):
            vertex = list(start)
            vertex[i] += self.initial_step
            simplex.append(vertex)
        return simplex

    def _centroid(self, simplex: List[List[float]], worst_idx: int) -> List[float]:
        """Вычисляет центроид всех вершин, кроме наихудшей."""
        n = len(simplex[0])
        c = [0.0] * n
        count = 0
        for i, v in enumerate(simplex):
            if i != worst_idx:
                for j in range(n):
                    c[j] += v[j]
                count += 1
        return [c[j] / count for j in range(n)]

    def _reflect(self, centroid: List[float], worst: List[float]) -> List[float]:
        n = len(centroid)
        return [centroid[j] + self.alpha * (centroid[j] - worst[j]) for j in range(n)]

    def _expand(self, centroid: List[float], reflected: List[float]) -> List[float]:
        n = len(centroid)
        return [
            centroid[j] + self.gamma * (reflected[j] - centroid[j]) for j in range(n)
        ]

    def _contract(self, centroid: List[float], worst: List[float]) -> List[float]:
        n = len(centroid)
        return [centroid[j] + self.rho * (worst[j] - centroid[j]) for j in range(n)]

    def _shrink(self, simplex: List[List[float]], best_idx: int) -> List[List[float]]:
        best = simplex[best_idx]
        n = len(best)
        new_simplex = [list(best)]
        for i, v in enumerate(simplex):
            if i != best_idx:
                new_v = [best[j] + self.sigma * (v[j] - best[j]) for j in range(n)]
                new_simplex.append(new_v)
        return new_simplex

    def optimize(
        self,
        target_func: ConstructiveNumber,
        variables: List[CNVariable],
        verbose: bool = False,
    ) -> Dict:
        """
        Запускает оптимизацию методом Нелдера-Мида.

        Args:
            target_func (ConstructiveNumber): Целевая функция (дерево вычислений).
            variables (List[CNVariable]): Список переменных функции.
            verbose (bool): Выводить ли прогресс в консоль.

        Returns:
            Dict: Словарь с ключами 'final_point', 'point_history', 'loss_history',
                  'n_func_evals' — совместим с форматом BaseOptimizer.
        """
        decimal.getcontext().prec = self.precision_digits

        start = [float(var.val) for var in variables]
        n = len(start)

        simplex = self._build_initial_simplex(start)
        # Вычисляем значения функции во всех вершинах начального симплекса
        f_vals = [self._eval(target_func, variables, v) for v in simplex]
        n_func_evals = len(simplex)

        point_history = [list(start)]
        loss_history = [f_vals[0]]

        for iteration in range(1, self.max_iter + 1):
            # Сортируем вершины: лучшая (min) → наихудшая (max)
            order = sorted(range(len(simplex)), key=lambda i: f_vals[i])
            simplex = [simplex[i] for i in order]
            f_vals = [f_vals[i] for i in order]

            best_val = f_vals[0]
            worst_val = f_vals[-1]
            second_worst_val = f_vals[-2]

            # Критерий остановки: стандартное отклонение значений функции
            mean_f = sum(f_vals) / len(f_vals)
            std_f = (sum((fv - mean_f) ** 2 for fv in f_vals) / len(f_vals)) ** 0.5
            if std_f < float(self.tolerance):
                if verbose:
                    print(
                        f"[Nelder-Mead] Сходится на итерации {iteration} (std={std_f:.2e})"
                    )
                break

            centroid = self._centroid(simplex, worst_idx=len(simplex) - 1)

            # --- Отражение ---
            x_r = self._reflect(centroid, simplex[-1])
            f_r = self._eval(target_func, variables, x_r)
            n_func_evals += 1

            if f_vals[0] <= f_r < second_worst_val:
                # Отражение принято
                simplex[-1] = x_r
                f_vals[-1] = f_r
            elif f_r < f_vals[0]:
                # --- Расширение ---
                x_e = self._expand(centroid, x_r)
                f_e = self._eval(target_func, variables, x_e)
                n_func_evals += 1
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_vals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
            else:
                # --- Сжатие ---
                x_ct = self._contract(centroid, simplex[-1])
                f_ct = self._eval(target_func, variables, x_ct)
                n_func_evals += 1
                if f_ct < worst_val:
                    simplex[-1] = x_ct
                    f_vals[-1] = f_ct
                else:
                    # --- Редукция ---
                    simplex = self._shrink(simplex, best_idx=0)
                    f_vals = [self._eval(target_func, variables, v) for v in simplex]
                    n_func_evals += len(simplex)

            # Записываем лучшую точку итерации
            best_point = simplex[0]
            point_history.append(list(best_point))
            loss_history.append(f_vals[0])

            if verbose and iteration % 100 == 0:
                print(
                    f"[Nelder-Mead] Итерация {iteration}/{self.max_iter} | "
                    f"Loss: {f_vals[0]:.8f} | std: {std_f:.2e}"
                )

        # Устанавливаем переменные в найденный минимум
        best_point = simplex[0]
        for var, val in zip(variables, best_point):
            var.set_val(str(val))

        return {
            "final_point": [decimal.Decimal(str(v)) for v in best_point],
            "point_history": point_history,
            "loss_history": loss_history,
            "n_func_evals": n_func_evals,
        }
