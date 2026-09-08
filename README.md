# sksfolio

`sksfolio` 求解带基数约束的稀疏 Markowitz 投资组合问题。项目当前只保留两种连续松弛算法：

- `corrected_fista`：外层 corrected FISTA，约束近端子问题使用 corrected dual-FISTA；
- `corrected_lbfgs`：相同外层算法，近端子问题优先使用 L-BFGS-B，并用独立残差检查；检查失败时安全回退到 corrected dual-FISTA。

两种算法求解同一个 long-only perspective relaxation：

```text
minimize    0.5 ||B' x||² + omega g_k(x) - rho mu' x
subject to  lower <= C x <= upper
            x in dom(g_k) = {x : 0 <= x <= 1, 1' x <= k}
```

这里的 `lower <= C x <= upper` 是**向量形式**，不是只有一条标量约束；`C` 的每一行都代表一条线性约束，并有各自的下界和上界。默认的 `hybrid` 问题生成器（`k >= 5`）共有 **15 行线性约束**：

| 约束类型 | `C` 的行数 | 展开后的标量形式 |
| --- | ---: | ---: |
| 预算等式 | 1 | 1 条等式 |
| 最低收益 | 1 | 1 条单边不等式 |
| 行业敞口 | 5 | 10 条不等式 |
| 风格因子敞口 | 3 | 6 条不等式 |
| 压力损失 | 5 | 5 条单边不等式 |
| 合计 | **15 行** | **1 条等式 + 22 条不等式** |

因此，代码按矩阵行计是 15 条；若把等式也写成两个不等式，则展开后共 24 条不等式。一般情况下，实际行数由输入的 `C.shape[0]` 决定。

其中 `g_k` 是基数约束的闭 perspective 凸包，并以扩展值形式包含 `0 <= x <= 1` 和 `1' x <= k` 的 long-only perspective 定义域；这些定义域约束不存入 `C`，也不计入上面的 15 行。默认全额投资模型中，在 `x >= 0`、`1' x = 1`、`k >= 1` 下，`x <= 1` 和 `1' x <= k` 是冗余的。PAVA、Fenchel 安全对偶证书和问题模型由两个算法共享。

## 后续流程

连续松弛之后的流程继续保留：

```text
corrected relaxation
        ↓
OSQP 固定支撑 QP → 可行稀疏 incumbent（上界）
        ↓
Fenchel safe screening
        ↓
certificate-driven branch-and-bound
```

因此，算法清理不会移除 incumbent、screening 或自定义 BnB。

## 安装

```bash
python -m pip install .
```

核心依赖是 NumPy、SciPy、OSQP 和 threadpoolctl。可选的 C 扩展只加速 partial-sort PAVA；没有编译扩展时会自动使用 NumPy 实现。

## 连续松弛 API

```python
from sksfolio import solve_relaxation

fista_result = solve_relaxation(
    problem,
    backend="corrected_fista",
    options={"tolerance": 1e-7, "max_iterations": 5_000},
)

lbfgs_result = solve_relaxation(
    problem,
    backend="corrected_lbfgs",
    options={"tolerance": 1e-7, "max_iterations": 5_000},
)

print(lbfgs_result.objective)
print(lbfgs_result.safe_dual_bound)
print(lbfgs_result.dual_certificate.verify(problem))
```

默认算法是 `corrected_lbfgs`：

```python
result = solve_relaxation(problem)
```

公共算法注册表严格为：

```python
from sksfolio import CORRECTED_ALGORITHMS

assert CORRECTED_ALGORITHMS == (
    "corrected_fista",
    "corrected_lbfgs",
)
```

## 完整求解流程

```python
from sksfolio import solve_bnb, solve_incumbent, solve_relaxation

relaxation = solve_relaxation(problem, "corrected_lbfgs")
incumbent = solve_incumbent(
    problem,
    relaxation,
    restricted_solver="osqp",
)
result = solve_bnb(
    problem,
    relaxation=relaxation,
    incumbent=incumbent if incumbent.feasible else None,
    restricted_solver="osqp",
)
```

`safe_dual_bound` 是连续松弛下界；incumbent 的 `upper_bound` 来自满足原始基数与线性约束的稀疏组合。BnB 使用这两个界、安全筛选和节点对偶更新推进搜索。

## 命令行示例

运行两个 corrected 算法：

```bash
python -m sksfolio.benchmarks.run_corrected_algorithms \
  --dimension 100 --rank 10 --k 10
```

运行完整后续流程：

```bash
python -m sksfolio.benchmarks.run_bnb \
  --relaxation-backend corrected_lbfgs \
  --dimension 100 --rank 10 --k 10
```

也可以运行教学示例：

```bash
python examples/student_quickstart.py --dimension 80 --rank 10 --k 8
```

## 项目结构

```text
sksfolio/
  relaxation/
    api.py              # 仅注册两个 corrected 算法
    fista/               # 外层 FISTA 与两种约束近端 oracle
    pava/                # 共享 perspective prox
    safe_dual.py         # Fenchel 安全下界
    problem.py
    state.py
  incumbent/             # OSQP/SciPy 固定支撑 QP 与启发式
  screening/             # 安全筛选
  bnb/                   # 自定义 branch-and-bound
  benchmarks/
examples/
tests/
```

## 验证

```bash
python -m pytest -q
```

浮点结果附带可重新计算的 Fenchel 弱对偶证书。其数学安全性是在精确算术意义下陈述的；浮点实现会额外记录可行性残差和界一致性诊断。
