import sys
import time
from Core.constructive_numbers import CNConstant, CNVariable

if __name__ == "__main__":
    x = CNVariable('x')
    f = (x ** 2 + 5 * x) * 3
    print('Исходная функция f(x):')
    print(f)
    print('-' * 40)

    df_raw = f.derivative(x)
    print('Сырая производная (до упрощения):')
    print(df_raw)
    print('-' * 40)

    df_simple = df_raw.simplify()
    print('Упрощенная производная:')
    print(df_simple)
