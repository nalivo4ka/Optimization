import math
from typing import List

from .constructive_numbers import (
    CNConstant,
    CNRound,
    CNSin,
    CNVariable,
    ConstructiveNumber,
)


def get_rosenbrock(variables: List[CNVariable]) -> ConstructiveNumber:
    """
    Генерирует Дерево для N-мерной функции Розенброка
    f(x) = sum(100 * (x_{i+1} - x_i^2)^2 + (1 - x_i)^2) для i от 0 до N-2
    """

    n = len(variables)
    if n < 2:
        raise ValueError("Для функции Розенброка требуется как минимум 2 переменные")

    c_100 = CNConstant(100)
    c_1 = CNConstant(1)
    c_2 = CNConstant(2)

    result_tree: ConstructiveNumber = (
        c_100 * ((variables[1] - (variables[0] ** c_2)) ** c_2)
        + (c_1 - variables[0]) ** c_2
    )

    for i in range(1, n - 1):
        x_i = variables[i]
        x_next = variables[i + 1]

        left_term = c_100 * ((x_next - (x_i**c_2)) ** c_2)
        right_term = (c_1 - x_i) ** c_2
        result_tree = result_tree + left_term + right_term

    return result_tree


def get_dixon_price_nd(variables: List[CNVariable]) -> ConstructiveNumber:
    """
    Дерево для N-мерной функции Диксона-Прайса
    f(x) = (x_0 - 1)^2 + sum( (i+1) * (2 * x_i^2 - x_{i-1})^2 ) для i от 1 до N-1
    """
    n = len(variables)
    if n < 2:
        raise ValueError(
            "Для функции Диксона-Прайса требуется как минимум 2 переменные"
        )

    c_1 = CNConstant(1)
    c_2 = CNConstant(2)

    result_tree: ConstructiveNumber = (variables[0] - c_1) ** c_2

    for i in range(1, n):
        x_i = variables[i]
        x_prev = variables[i - 1]

        i_coeff = CNConstant(i + 1)

        inner_term = (c_2 * (x_i**c_2)) - x_prev
        current_sum = i_coeff * (inner_term**c_2)
        result_tree = result_tree + current_sum

    return result_tree


def create_variables(n: int, initial_val: float = 0.0) -> List[CNVariable]:
    """
    Вспомогательная функция для быстрой генерации списка переменных
    """
    if n < 1:
        raise ValueError("Размерность пространства должна быть >= 1")

    return [CNVariable(name=f"x{i}", initial_val=initial_val) for i in range(n)]


def get_rastrigin_nd(variables: List[CNVariable]) -> ConstructiveNumber:
    """
    N-мерная функция Растригина:
        f(x) = A*n + sum_{i=1}^{n} [ x_i^2 - A * cos(2*pi*x_i) ]
    где A = 10.

    Глобальный минимум: f(0, 0, ..., 0) = 0.

    Обобщение на N измерений тривиально — функция является суммой
    одномерных слагаемых, каждое из которых зависит только от x_i.
    """
    n = len(variables)
    if n < 1:
        raise ValueError("Для функции Растригина требуется хотя бы 1 переменная")

    A = CNConstant(10)
    c_2 = CNConstant(2)
    two_pi = CNConstant(str(2 * math.pi))

    result_tree: ConstructiveNumber = A * CNConstant(n)

    for x_i in variables:
        cos_term = CNSin(
            two_pi * x_i + CNConstant(str(math.pi / 2))
        )  # cos(u) = sin(u + π/2)
        term = (x_i**c_2) - A * cos_term
        result_tree = result_tree + term

    return result_tree


def get_desmos_2d(variables: List[CNVariable]) -> ConstructiveNumber:
    """
    Специальная 2D-функция из Desmos:
        f(x, y) = ((x * (round(sin(10*y)) + 2))^2 + y - 10)^2
                + (x + (y * (round(sin(7*x)) + 2))^2 - 7)^2

    Функция негладкая из-за операции round(sin(...)).
    Глобальный минимум ищется численно (приблизительно около (2, 3)).
    """
    if len(variables) != 2:
        raise ValueError("get_desmos_2d требует ровно 2 переменные")

    x, y = variables[0], variables[1]
    c_2 = CNConstant(2)
    c_7 = CNConstant(7)
    c_10 = CNConstant(10)

    # round(sin(10*y)) + 2
    rsy = CNRound(CNSin(c_10 * y)) + c_2
    # round(sin(7*x)) + 2
    rsx = CNRound(CNSin(c_7 * x)) + c_2

    # ((x * rsy)^2 + y - 10)^2
    term1 = ((x * rsy) ** c_2 + y - c_10) ** c_2
    # (x + (y * rsx)^2 - 7)^2
    term2 = (x + (y * rsx) ** c_2 - c_7) ** c_2

    return term1 + term2


def get_desmos_nd(variables: List[CNVariable]) -> ConstructiveNumber:
    """
    N-мерное обобщение функции Desmos.

    Математическое обоснование обобщения:
    Исходная 2D-функция имеет структуру суммы двух «перекрёстных» слагаемых:
        T1 = (x_0 * A(x_1) + x_1 - 10)^2
        T2 = (x_0 + x_1 * B(x_0) - 7)^2
    где A(u) = round(sin(10*u)) + 2, B(u) = round(sin(7*u)) + 2.

    Для N измерений обобщаем как сумму N слагаемых по кольцевой схеме:
        f(x) = sum_{i=0}^{N-1} (x_i * A(x_{i+1 mod N}) + x_{(i+1) mod N} - 10)^2
             + sum_{i=0}^{N-1} (x_i + x_{(i+1) mod N} * B(x_i) - 7)^2

    При N=2 это точно совпадает с исходной формулой.
    """
    n = len(variables)
    if n < 2:
        raise ValueError("get_desmos_nd требует хотя бы 2 переменные")

    c_2 = CNConstant(2)
    c_7 = CNConstant(7)
    c_10 = CNConstant(10)

    x0, x1 = variables[0], variables[1]
    A0 = CNRound(CNSin(c_10 * x1)) + c_2
    B0 = CNRound(CNSin(c_7 * x0)) + c_2
    result_tree: ConstructiveNumber = (x0 * A0 + x1 - c_10) ** c_2 + (
        x0 + x1 * B0 - c_7
    ) ** c_2

    for i in range(1, n):
        x_i = variables[i]
        x_next = variables[(i + 1) % n]

        A = CNRound(CNSin(c_10 * x_next)) + c_2
        B = CNRound(CNSin(c_7 * x_i)) + c_2

        t1 = (x_i * A + x_next - c_10) ** c_2
        t2 = (x_i + x_next * B - c_7) ** c_2

        result_tree = result_tree + t1 + t2

    return result_tree
