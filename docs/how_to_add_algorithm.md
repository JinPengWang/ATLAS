# 如何在 ATLAS 中添加全新的优化算法

ATLAS (v0.2+) 提供了学术研究级别的极简单类算法扩展接口。开发者**只需编写一个 Python 类**，便可无缝获得：
- ✅ **代数（Iteration-based）与评价次数（NFE-based）双停止准则** 自动兼容
- ✅ **自动化 NFE 计数、边界越界防护、早停机制（Early Stopping）**
- ✅ **自动全局最优解（$g^*, x^*$）与收敛曲线追踪**
- ✅ **多进程并行调度（`n_jobs > 1`）与多维度笛卡尔积测试**
- ✅ **自动学术级非参数检验（Wilcoxon Signed-Rank / Friedman Test）与 LaTeX/Markdown 报表导出**

---

## 快速上手：3 步添加自定义算法

### 第 1 步：新建算法文件
在 `atlas/algorithms/iteration/` 目录下（或你自己的项目任意目录中）创建算法文件，例如 `gwo.py`（灰狼优化算法）。

### 第 2 步：继承 `BaseAlgorithm` 并实现接口
你只需要实现两个抽象方法：
1. `initialize()`: 初始化种群和个体适应度。
2. `iterate(iter_idx: int)`: 执行单代搜索更新，返回当前代找到的全局最优适应度。

### 第 3 步：使用 `@register_algorithm` 注册

---

## 完整代码模板（以 Grey Wolf Optimizer 为例）

```python
"""Grey Wolf Optimizer (GWO) implementation in ATLAS."""

from __future__ import annotations
from typing import Any
import numpy as np

from atlas.core.base_algorithm import BaseAlgorithm
from atlas.core.base_problem import BaseProblem
from atlas.utils.registry import register_algorithm


@register_algorithm("gwo", aliases=["gwo_nfe"])
class GWO(BaseAlgorithm):
    """Grey Wolf Optimizer.

    Reference:
        Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014).
        Grey Wolf Optimizer. Advances in Engineering Software, 69, 46-61.
    """

    def __init__(
        self,
        problem: BaseProblem,
        max_iter: int = 500,
        pop_size: int = 30,
        seed: int = 42,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            problem,
            max_iter=max_iter,
            pop_size=pop_size,
            seed=seed,
            verbose=verbose,
            **kwargs,
        )

    def initialize(self) -> None:
        # 1. 自动生成在问题边界 [lb, ub] 内的均匀初始种群
        self.population = self.init_population()
        # 2. 自动批量评估（自动累计 NFE 并更新全局最优）
        self.fitness = self.evaluate_population(self.population)
        
        # 3. 初始化三头领狼 (Alpha, Beta, Delta)
        order = np.argsort(self.fitness)
        self.alpha_x = self.population[order[0]].copy()
        self.alpha_f = self.fitness[order[0]]
        self.beta_x = self.population[order[1]].copy()
        self.beta_f = self.fitness[order[1]]
        self.delta_x = self.population[order[2]].copy()
        self.delta_f = self.fitness[order[2]]

    def iterate(self, iter_idx: int) -> float:
        # 线性衰减因子 a: 2 -> 0
        total_it = max(self.max_iter, 1) if self.max_iter > 0 else 500
        a = 2.0 - 2.0 * min((iter_idx + 1) / total_it, 1.0)

        # 遍历狼群更新位置
        for i in range(self.pop_size):
            # Alpha 狼引导
            r1, r2 = self.rng.random(self.dim), self.rng.random(self.dim)
            A1 = 2.0 * a * r1 - a
            C1 = 2.0 * r2
            D_alpha = np.abs(C1 * self.alpha_x - self.population[i])
            X1 = self.alpha_x - A1 * D_alpha

            # Beta 狼引导
            r1, r2 = self.rng.random(self.dim), self.rng.random(self.dim)
            A2 = 2.0 * a * r1 - a
            C2 = 2.0 * r2
            D_beta = np.abs(C2 * self.beta_x - self.population[i])
            X2 = self.beta_x - A2 * D_beta

            # Delta 狼引导
            r1, r2 = self.rng.random(self.dim), self.rng.random(self.dim)
            A3 = 2.0 * a * r1 - a
            C3 = 2.0 * r2
            D_delta = np.abs(C3 * self.delta_x - self.population[i])
            X3 = self.delta_x - A3 * D_delta

            # 综合位置并限制在边界内
            new_pos = self._clip((X1 + X2 + X3) / 3.0)

            # 单点评估（自动递增 NFE、比对 global best）
            f = self.evaluate(new_pos)

            self.population[i] = new_pos
            self.fitness[i] = f

            # 更新三头领狼
            if f < self.alpha_f:
                self.delta_x, self.delta_f = self.beta_x.copy(), self.beta_f
                self.beta_x, self.beta_f = self.alpha_x.copy(), self.alpha_f
                self.alpha_x, self.alpha_f = new_pos.copy(), f
            elif f < self.beta_f:
                self.delta_x, self.delta_f = self.beta_x.copy(), self.beta_f
                self.beta_x, self.beta_f = new_pos.copy(), f
            elif f < self.delta_f:
                self.delta_x, self.delta_f = new_pos.copy(), f

        return self.g_best_f
```

---

## 核心 API 与辅助方法说明

基类 `BaseAlgorithm` 已经内建了算法开发者最常用的一系列工具函数：

| 方法 / 属性 | 功能说明 |
| :--- | :--- |
| `self.evaluate(x)` | **核心评估函数**。评估候选解 $x$，自动增加 NFE 计数器，自动判断并更新 `self.g_best_f` 和 `self.g_best_x`，支持边界惩罚和早停。 |
| `self.evaluate_population(pop)` | **批量评估函数**。对整个人群矩阵 `(pop_size, dim)` 进行评估并返回 `(pop_size,)` 适应度数组。 |
| `self.init_population(pop_size=None)` | 在上下界 $[lb, ub]$ 内均匀随机采样生成初始解。 |
| `self._clip(x)` | 将解向量截断限制在问题边界 $[lb, ub]$ 内。 |
| `self.rng` | 内建的 `numpy.random.Generator`，保障多进程/多线程下独立可复现的随机数生成。 |
| `self.dim`, `self.lb`, `self.ub` | 当前测试问题的维度、下界、上界。 |
| `self.g_best_x`, `self.g_best_f` | 当前算法在历史上搜索到的全局最优解与最优适应度。 |

---

## 如何使用新算法

### 1. 在 Python 脚本中直接调用
```python
from atlas.problems.benchmark.unimodal import Sphere
from atlas.utils.registry import get_algorithm

# 获取注册的算法类
algo_cls = get_algorithm("gwo")

# 实例化并运行（代数模式）
prob = Sphere(dim=30)
algo = algo_cls(problem=prob, max_iter=500, pop_size=30, seed=42)
result = algo.run()

print(f"Best fitness: {result.best_fitness:.6e}")
print(f"Total NFE: {result.nfe}")
```

### 2. 在 NFE 模式下运行
不需要修改任何算法代码，只需指定 `max_nfe` 或别名 `gwo_nfe`：
```python
# NFE 预算模式
algo = algo_cls(problem=prob, max_nfe=10000, pop_size=30)
result = algo.run()
```

### 3. 使用命令行基准测试器
```bash
# 在 Sphere 和 Rastrigin 测试 GWO，开启 4 进程并行
python experiments/run_benchmark.py --algorithms gwo pso de --problems sphere rastrigin --dim 30 --runs 30 --n_jobs 4
```
运行完成后，ATLAS 会在 `results/statistical_reports/` 下自动生成：
- `benchmark_table.tex`（可直接插入 LaTeX 论文的带 Mean $\pm$ Std 与加粗最优的三线表）
- `summary_statistics.md`
- `summary_statistics.csv`
- `friedman_test.txt`（Friedman 秩次统计检验结果）
