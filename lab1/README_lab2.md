# Lab 2 — Исследование алгоритмов оптимизации

## Структура проекта

```
lab1/
├── Core/
│   ├── constructive_numbers.py   # ConstructiveNumber + CNSin, CNRound (новые)
│   ├── functions.py              # Розенброк, Диксон-Прайс, Растригин (новые), Desmos (новые)
│   ├── interval.py               # Интервальная арифметика
│   └── math_tools.py             # Дифференцирование + упрощение (+ правила для CNSin, CNRound)
├── Optimization/
│   ├── optimizers.py             # Vanilla GD, Momentum, Adam, Kiefer-Wolfowitz (Lab 1)
│   └── nelder_mead.py            # Нелдер-Мид (новый, для негладких функций)
├── Visuals/
│   └── visualizer.py             # Визуализатор (+ plot_convergence_comparison, plot_paths_comparison)
├── run_lab2_experiments.py       # ← ГЛАВНЫЙ СКРИПТ ИССЛЕДОВАНИЯ
└── README_lab2.md                # Этот файл
```

## Запуск

```bash
cd lab1
/usr/local/bin/python3.12 run_lab2_experiments.py
```

> ⚠️ Используйте `/usr/local/bin/python3.12` — он содержит numpy и matplotlib.
> Системный `python3` (Homebrew 3.14) не имеет этих пакетов.

## Что исследуется

### Функции
| Функция | 2D | N-D | Тип |
|---|---|---|---|
| Розенброк | ✓ | ✓ (5D) | Гладкая, овражная |
| Растригин | ✓ | ✓ (5D) | Гладкая, многоэкстремальная |
| Desmos | ✓ | ✓ (4D) | **Негладкая** (round(sin(...))) |

### Алгоритмы
| Алгоритм | Тип | Порядок | Файл |
|---|---|---|---|
| Vanilla GD | Градиентный | 1-й | `optimizers.py` |
| Momentum GD | Градиентный | 1-й | `optimizers.py` |
| Adam | Адаптивный | 1-й | `optimizers.py` |
| Kiefer-Wolfowitz | Стохастический | 0-й (конечные разности) | `optimizers.py` |
| **Nelder-Mead** | Прямой поиск | **0-й (без градиента)** | `nelder_mead.py` |

### Метрики сравнения
- Время выполнения (с)
- Число итераций
- Число вызовов функции
- Финальное значение f(x*)
- Расстояние до истинного оптимума ||x* - x_opt||

## Выходные данные

Скрипт создаёт папку `Results_Lab2/` со следующими файлами:

```
Results_Lab2/
├── rosenbrock_2d_surface_3d.png       # 3D-поверхность
├── rosenbrock_2d_convergence.png      # Сравнение сходимости всех алгоритмов
├── rosenbrock_2d_paths.png            # Траектории спуска на контурном графике
├── rosenbrock_5d_convergence.png
├── rastrigin_2d_surface_3d.png
├── rastrigin_2d_convergence.png
├── rastrigin_2d_paths.png
├── rastrigin_5d_convergence.png
├── desmos_2d_surface_3d.png
├── desmos_2d_convergence.png
├── desmos_2d_paths.png
└── desmos_4d_convergence.png
```

## Математическое обоснование N-мерных обобщений

### Функция Растригина (N-D)
$$f(\mathbf{x}) = An + \sum_{i=1}^{n} \left[ x_i^2 - A\cos(2\pi x_i) \right], \quad A = 10$$

Обобщение **тривиально** — функция сепарабельна: каждое слагаемое зависит только от $x_i$.
Глобальный минимум: $f(\mathbf{0}) = 0$.

### Функция Desmos (N-D)
Исходная 2D-формула:
$$f(x, y) = \bigl((x \cdot A(y))^2 + y - 10\bigr)^2 + \bigl(x + (y \cdot B(x))^2 - 7\bigr)^2$$
где $A(u) = \text{round}(\sin(10u)) + 2$, $B(u) = \text{round}(\sin(7u)) + 2$.

**Кольцевое обобщение** на $N$ измерений:
$$f(\mathbf{x}) = \sum_{i=0}^{N-1} \left[ \bigl(x_i \cdot A(x_{i+1 \bmod N}) + x_{i+1 \bmod N} - 10\bigr)^2 + \bigl(x_i + x_{i+1 \bmod N} \cdot B(x_i) - 7\bigr)^2 \right]$$

При $N=2$ формула точно совпадает с исходной.

## Алгоритм Нелдера-Мида

Метод деформируемого симплекса — не требует градиента, работает с любыми функциями.

**Операции симплекса** (стандартные параметры Nelder & Mead, 1965):
- Отражение: $\mathbf{x}_r = \mathbf{x}_c + \alpha(\mathbf{x}_c - \mathbf{x}_w)$, $\alpha=1$
- Расширение: $\mathbf{x}_e = \mathbf{x}_c + \gamma(\mathbf{x}_r - \mathbf{x}_c)$, $\gamma=2$
- Сжатие: $\mathbf{x}_{ct} = \mathbf{x}_c + \rho(\mathbf{x}_w - \mathbf{x}_c)$, $\rho=0.5$
- Редукция: $\mathbf{x}_i = \mathbf{x}_b + \sigma(\mathbf{x}_i - \mathbf{x}_b)$, $\sigma=0.5$

**Критерий остановки**: стандартное отклонение значений функции в вершинах симплекса $< \varepsilon$.
