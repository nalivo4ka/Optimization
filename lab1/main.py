import sys
import time
from Core.constructive_numbers import CNConstant

def run_perf_test(depth=1000, precision=20):
    print(f"Глубина {depth}")
    print(f"точность {precision}")

    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, depth + 200000))

    start_build = time.perf_counter()
        
    tree = CNConstant(1)
    for i in range(2, depth + 1):
        tree += CNConstant(1) / CNConstant(i)
            
    build_time = time.perf_counter() - start_build
    print(f"Построено за {build_time:.5f} сек")

    start_eval = time.perf_counter()
    result_interval = tree.evaluate(precision)
    print(result_interval.middle)
        
    eval_time = time.perf_counter() - start_eval
    print(f"Вычислено за {eval_time:.5f} сек")

    start_eval = time.perf_counter()
    result_interval = tree.evaluate(precision)
    eval_time = time.perf_counter() - start_eval
    print(f"Второе вычисление выполнено за {eval_time:.5f} сек")

if __name__ == "__main__":
    run_perf_test(depth=10000, precision=100)
