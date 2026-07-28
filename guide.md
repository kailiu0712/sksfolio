Below is the clean mental model for the three files.

* `Sparse_Portfolio_main(1).pdf` gives the mathematical problem, the dual reduction, Algorithm 1 restarted dual FISTA, and Algorithm 2 globalized semismooth Newton. 
* `prox_compare(1).py` implements those two algorithms, the PAVA proximal oracle, a Gurobi reference solve, stopping certificates, tests, and benchmarking. 
* `markowitz_linesearch_pdhg_benchmark(1).py` implements a different outer algorithm: adaptive restarted PDHG with a Malitsky–Pock line search. It still uses PAVA for the primal proximal step, but it is not Algorithm 1 or Algorithm 2 from the PDF. 

The easiest way to understand everything is:

[
\boxed{
\text{hard constrained prox}
;\longrightarrow;
\text{smooth low-dimensional dual problem}
;\longrightarrow;
\text{each dual evaluation calls PAVA}
}
]

The outer algorithm can then be FISTA, semismooth Newton, or PDHG.

---

# 1. What optimization problem are the files solving?

The basic constrained proximal problem is

[
x^\star
=======

\arg\min_{Ax\le b}
\left{
G(x)+\frac{1}{2\gamma}|x-v|_2^2
\right}.
\tag{P}
]

Here:

* (x\in\mathbb R^d) is the primal variable;
* (v) is the point being proxed;
* (\gamma>0) is the proximal scale;
* (Ax\le b) are additional linear constraints;
* (G) is the perspective-relaxed sparse regularizer.

In the experiment,

[
G(x)
====

\min_z
\left{
\frac12\sum_{i=1}^d\frac{x_i^2}{z_i}
:
0\le z_i\le1,;
\mathbf1^\top z\le k,;
|x_i|\le Mz_i
\right}.
\tag{1}
]

The original cardinality idea is “use at most (k) coordinates.” The perspective variables (z_i) relax that idea convexly:

* (z_i=0) forces (x_i=0);
* (z_i=1) permits (x_i) up to magnitude (M);
* fractional (z_i) gives a strong convex relaxation;
* (\sum_i z_i\le k) is the sparsity budget.

The PDF assumes (G) is proper, closed, convex, and piecewise linear–quadratic, and that (\operatorname{prox}_{\gamma G}) is inexpensive. In this implementation, PAVA provides that inexpensive base prox. 

---

# 2. Why introduce the dual?

Directly solving (P) combines:

* the structured nonsmooth term (G),
* a quadratic proximal term,
* linear constraints.

The clever step is to dualize only the constraints (Ax\le b).

Introduce multipliers
[
\lambda\ge0.
]

The Lagrangian is

[
L(x,\lambda)
============

G(x)
+
\frac{1}{2\gamma}|x-v|^2
+
\lambda^\top(Ax-b).
]

For fixed (\lambda), complete the square:

[
\frac{1}{2\gamma}|x-v|^2
+
\lambda^\top Ax
===============

\frac{1}{2\gamma}
|x-(v-\gamma A^\top\lambda)|^2
+\text{terms independent of }x.
]

Therefore the minimizing (x) is

[
\boxed{
x(\lambda)
==========

\operatorname{prox}_{\gamma G}
\bigl(v-\gamma A^\top\lambda\bigr).
}
\tag{2}
]

This is the central formula used everywhere.

It means:

> Given a multiplier (\lambda), shift the prox input by (-\gamma A^\top\lambda), then call PAVA once.

The dual function is

[
q_v(\lambda)
============

e_{\gamma G}
\bigl(v-\gamma A^\top\lambda\bigr)
+
\lambda^\top(Av-b)
------------------

\frac{\gamma}{2}|A^\top\lambda|^2,
]

and its gradient is especially simple:

[
\boxed{
\nabla q_v(\lambda)=Ax(\lambda)-b.
}
\tag{3}
]

This gradient is simply the vector of constraint residuals.

* If (A_ix(\lambda)-b_i>0), constraint (i) is violated.
* If it is negative, the constraint has slack.
* If (\lambda_i>0) at the solution, complementary slackness usually makes the corresponding residual zero.

The PDF states that the dual gradient has Lipschitz constant

[
L_d=\gamma|A|_2^2.
]

Both Algorithm 1 and Algorithm 2 maximize (q_v(\lambda)) over (\lambda\ge0). 

---

# 3. The complete computational chain

A single dual evaluation at (\lambda) performs:

[
\lambda
;\xrightarrow{A^\top};
A^\top\lambda
;\xrightarrow{\text{shift}};
u=v-\gamma A^\top\lambda
]

[
u
;\xrightarrow{\text{PAVA prox}};
x(\lambda)=\operatorname{prox}_{\gamma G}(u)
]

[
x(\lambda)
;\xrightarrow{A};
Ax(\lambda)-b
=============

\nabla q_v(\lambda).
]

In code, this is `evaluate_dual`:

```python
at_multiplier = tmatvec(a, multiplier)
shifted = [vi - gamma * ati for vi, ati in zip(v, at_multiplier)]

state = perspective_prox_state_dispatch(
    shifted, gamma, k, m_bound, method=pava_method
)

ax = matvec(a, state.x)
gradient = [axi - bi for axi, bi in zip(ax, b)]
```

The `state` contains:

* `state.x`: the primal point (x(\lambda));
* PAVA pools;
* the Moreau-envelope value needed for (q_v(\lambda));
* a generalized Jacobian-vector product used by Newton. 

---

# 4. What PAVA is doing here

PAVA stands for pool-adjacent-violators algorithm.

The base prox of (G) is not coordinatewise separable because the sparsity budget

[
\sum_i z_i\le k
]

couples coordinates. However, after sorting coordinates by (|u_i|), the prox can be represented by monotone “subtraction” values. If adjacent values violate the required monotone order, they are pooled into one block.

In the code:

```python
permutation = sorted(
    range(len(u)),
    key=lambda i: (-abs(u[i]), i)
)
```

Then singleton pools are created. Each pool stores:

* `start`, `end`: its range in sorted order;
* `size`;
* `selected`: number of indices from the top-(k) side;
* `total`: sum of magnitudes;
* `subtraction`: the shared shrinkage applied to members;
* `kappa`: derivative coefficient;
* `branch`: `"quadratic"` or `"linear"`.

The merge condition is

```python
while len(pools) >= 2 and pools[-2].subtraction < pools[-1].subtraction:
```

If the left subtraction is smaller than the right subtraction, monotonicity is violated. The two pools are merged and assigned a new common subtraction. 

## A simple PAVA example

Suppose sorted magnitudes are

[
|u|=(6,;4,;3.5,;1),
]

with (k=2), (\gamma=1), and very large (M), so all pools remain in the quadratic branch.

For a singleton selected coordinate, the code gives approximately

[
\text{subtraction}
==================

# \frac{\gamma |u_i|}{\gamma+1}

\frac{|u_i|}{2}.
]

For unselected coordinates, `selected=0`, giving

[
\text{subtraction}=|u_i|.
]

Initial singleton subtractions are thus

[
3,\quad 2,\quad 3.5,\quad 1.
]

But monotonicity requires

[
3\ge2\ge3.5\ge1,
]

and (2<3.5) violates the order.

Therefore PAVA merges coordinates 2 and 3.

For the merged pool:

* size (=2);
* selected count (=1);
* total magnitude (=4+3.5=7.5).

The common subtraction becomes

[
\frac{\gamma\cdot7.5}{\gamma\cdot2+1}
=====================================

# \frac{7.5}{3}

2.5.
]

The pooled sequence is

[
3,\quad 2.5,\quad2.5,\quad1,
]

which is monotone.

That pooled structure gives the exact prox within that sorting cell.

---

# 5. Why PAVA also gives a generalized Jacobian

Semismooth Newton needs the derivative of

[
x(\lambda)
==========

\operatorname{prox}_{\gamma G}
(v-\gamma A^\top\lambda).
]

The prox is not globally differentiable because:

* coordinates can change sorting order;
* blocks can merge or split at boundaries;
* the scalar formula changes between quadratic and linear branches.

But within one fixed PAVA cell—same sorting permutation, same blocks, same branches—the map is affine. Therefore it has an ordinary Jacobian there.

At a boundary, it has multiple possible limiting Jacobians. Choosing one of them gives an element of the generalized Jacobian.

The code implements its action on a vector (h) via

```python
state.jvp(h)
```

where JVP means Jacobian-vector product.

Inside each pool, the perturbation coordinates are summed and multiplied by `pool.kappa`:

```python
pool_sum = sum(signed_sorted[pool.start : pool.end])
value = pool.kappa * pool_sum
```

Then this correction is subtracted from (h). This avoids forming a dense (d\times d) Jacobian.

That JVP is one of the most important pieces of the entire Newton method. 

---

# 6. Algorithm 1: restarted dual FISTA

The PDF’s Algorithm 1 solves

[
\max_{\lambda\ge0}q_v(\lambda),
]

or equivalently minimizes

[
\psi_v(\lambda)
===============

-q_v(\lambda)+\delta_{\mathbb R_+^m}(\lambda).
]

The update is accelerated projected gradient ascent.

## 6.1 Basic projected dual-gradient step

At an extrapolated point (y),

[
x_y
===

\operatorname{prox}_{\gamma G}
(v-\gamma A^\top y),
]

[
\nabla q_v(y)=Ax_y-b.
]

Then

[
\lambda^+
=========

\left[
y+\frac{1}{L_d}(Ax_y-b)
\right]_+.
\tag{4}
]

The positive-part projection is coordinatewise:

[
[z]_i^+=\max(z_i,0).
]

Why does this make sense?

If constraint (i) is violated, then

[
(Ax_y-b)_i>0,
]

so (\lambda_i) increases, imposing a stronger penalty on that constraint.

If the constraint has slack, the gradient is negative and (\lambda_i) decreases, but never below zero.

## 6.2 Nesterov acceleration

FISTA adds momentum:

[
t^+
===

\frac{1+\sqrt{1+4t^2}}{2},
]

[
y^+
===

\lambda^+
+
\frac{t-1}{t^+}
(\lambda^+-\lambda).
\tag{5}
]

So the next gradient is evaluated not at (\lambda^+) itself, but at an extrapolated point (y^+).

In code:

```python
multiplier_next = positive_part(
    [
        yi + gradient_i / lipschitz
        for yi, gradient_i in zip(
            extrapolated,
            evaluation_y.gradient
        )
    ]
)

momentum_next = 0.5 * (
    1.0 + math.sqrt(
        1.0 + 4.0 * momentum * momentum
    )
)

extrapolated_next = [
    next_i
    + (momentum - 1.0)
      * (next_i - old_i)
      / momentum_next
    for next_i, old_i
    in zip(multiplier_next, multiplier)
]
```



---

# 7. Why restart FISTA?

Acceleration can overshoot or oscillate, especially for piecewise quadratic functions.

Every (T) iterations, Algorithm 1 resets

[
y=\lambda,\qquad t=1.
]

This removes accumulated momentum.

The PDF explains that because the dual objective is piecewise linear–quadratic, it satisfies a local quadratic-growth property:

[
D(\lambda)
\ge
\frac{\mu_v}{2}
\operatorname{dist}^2(\lambda,\Lambda^\star).
]

Under this property, a sufficiently long FISTA epoch contracts the error by a constant factor. Repeated restarts then give linear convergence at restart points. 

The code uses a tunable `restart_length`, default 25.

At a restart:

```python
extrapolated = multiplier[:]
momentum = 1.0
```

It also computes a rigorous primal–dual certificate.

---

# 8. The feasible certificate

The dual point (\lambda) produces (x(\lambda)), but this point may violate (Ax\le b) before convergence.

The paper assumes a strictly feasible anchor (\bar x) satisfying

[
A\bar x<b.
]

The code uses (\bar x=0), which requires (b>0) componentwise.

Let

[
u=[Ax(\lambda)-b]_+
]

be the positive violation vector, and

[
s=b-A\bar x
]

the strict slack of the anchor.

Define

[
\theta
======

\max_i
\frac{u_i}{u_i+s_i}.
]

Then form

[
x_b
===

(1-\theta)x(\lambda)+\theta\bar x.
\tag{6}
]

Because (\bar x) is strictly feasible, moving sufficiently toward it makes (x_b) feasible.

With (\bar x=0), this simplifies to

[
x_b=(1-\theta)x(\lambda).
]

The primal–dual gap is

[
\Delta(x_b,\lambda)
===================

p_v(x_b)-q_v(\lambda)\ge0.
]

This certifies both:

[
p_v(x_b)-p_v(x^\star)\le\Delta
]

and

[
|x_b-x^\star|
\le
\sqrt{2\gamma\Delta}.
]

So stopping is based on a mathematically meaningful certificate, not merely iteration count. 

---

# 9. Small numerical FISTA example

Take one inequality:

[
x_1+x_2\le0.6.
]

Thus

[
A=\begin{bmatrix}1&1\end{bmatrix},
\qquad
b=0.6,
\qquad
\lambda\ge0.
]

Suppose at (\lambda=0), the base prox produces

[
x(0)=(0.5,0.4).
]

Then the constraint residual is

[
Ax(0)-b
=======

# 0.9-0.6

0.3.
]

With (L_d=1),

[
\lambda^+
=========

# [0+0.3]_+

0.3.
]

The next shifted prox input is

[
v-\gamma A^\top\lambda^+
========================

v-\gamma
\begin{bmatrix}
0.3\0.3
\end{bmatrix}.
]

Both coordinates are pushed downward. Suppose the next prox gives

[
x(0.3)=(0.35,0.28).
]

Then

[
Ax(0.3)-b
=========

# 0.63-0.6

0.03.
]

The multiplier increases slightly again. Eventually the active constraint settles near equality.

That is the whole dual-gradient intuition.

---

# 10. Algorithm 2: globalized semismooth Newton

FISTA uses first-order information only.

Semismooth Newton attempts to solve the dual KKT conditions directly and exploit the fact that the residual is piecewise affine.

## 10.1 Projected KKT residual

The constrained dual optimum satisfies

[
\lambda
=======

\left[
\lambda+c\nabla q_v(\lambda)
\right]_+,
]

for any (c>0).

Since

[
\nabla q_v(\lambda)=Ax(\lambda)-b,
]

define

[
\boxed{
R_v(\lambda)
============

## \lambda

\left[
\lambda+c(Ax(\lambda)-b)
\right]_+.
}
\tag{7}
]

Solving the dual KKT system is equivalent to solving

[
R_v(\lambda)=0.
]

Interpret coordinatewise:

### Case 1: (\lambda_i>0)

At the solution, the projection is locally inactive and

[
R_i
===

\lambda_i-
(\lambda_i+cg_i)
================

-cg_i.
]

Thus (R_i=0) implies

[
g_i=(Ax-b)_i=0.
]

So a positive multiplier corresponds to an active constraint.

### Case 2: (\lambda_i=0)

Then

[
R_i
===

*

[cg_i]_+.
]

For this to be zero, we need

[
g_i\le0.
]

So a zero multiplier corresponds to a satisfied, possibly slack constraint.

This compact residual encodes primal feasibility, dual feasibility, and complementary slackness simultaneously.

---

# 11. Why it is called semismooth

The positive-part map

[
[z]_+=\max(z,0)
]

is not differentiable at (z=0).

The PAVA prox is also not globally differentiable at:

* pool-merger boundaries;
* permutation ties;
* scalar branch boundaries.

However, both maps are piecewise affine or piecewise smooth. Such functions are strongly semismooth under standard conditions.

This permits Newton-like methods using generalized Jacobians.

The PDF gives

[
V
=

I-E+c\gamma E AJA^\top
\in\partial_B R_v(\lambda),
\tag{8}
]

where:

* (J) is a generalized Jacobian of (\operatorname{prox}_{\gamma G});
* (E) is a diagonal generalized derivative of the positive-part map.



---

# 12. Deriving the Newton matrix

Let

[
w(\lambda)
==========

\lambda+c(Ax(\lambda)-b).
]

Then

[
R(\lambda)=\lambda-[w(\lambda)]_+.
]

A generalized derivative of ([w]_+) is a diagonal matrix (E), where approximately

[
E_{ii}
======

\begin{cases}
1,&w_i>0,\
0,&w_i<0.
\end{cases}
]

The code uses

```python
projection_derivative = [
    1.0 if value > tie_tolerance else 0.0
    for value in projected_argument
]
```

Now

[
x(\lambda)
==========

\operatorname{prox}_{\gamma G}
(v-\gamma A^\top\lambda).
]

A perturbation (d\lambda) gives

[
dx
\approx
J(-\gamma A^\top d\lambda).
]

Therefore

[
d(Ax-b)
\approx
-\gamma AJA^\top d\lambda.
]

So

[
dw
\approx
\left(I-c\gamma AJA^\top\right)d\lambda.
]

Hence

[
dR
\approx
d\lambda
--------

E,dw
]

# [

\left(
I-E+c\gamma EAJA^\top
\right)d\lambda.
]

This is precisely the matrix (V).

The Newton equation is

[
(V+\sigma_jI)d=-R(\lambda),
\tag{9}
]

where (\sigma_j\ge0) is a numerical regularization.

---

# 13. How the code constructs (AJA^\top)

Suppose (A) has (m) rows. The desired Newton matrix is (m\times m), which is often small relative to (d).

For column (j), take row (A_j) as a vector in (\mathbb R^d), compute

[
J A_j^\top
]

using the PAVA state’s JVP, then dot against every row (A_i):

[
(AJA^\top)_{ij}
===============

A_i J A_j^\top.
]

The code:

```python
hessian = [[0.0] * len(b) for _ in b]

for column, row in enumerate(a):
    j_row = evaluation.state.jvp(row)

    for i, row_i in enumerate(a):
        hessian[i][column] = (
            gamma * dot(row_i, j_row)
        )
```

The variable is called `hessian`, though technically it is the generalized curvature term

[
\gamma AJA^\top.
]



---

# 14. Why add regularization to the Newton system?

The generalized Newton matrix may be singular or poorly conditioned, especially:

* before the correct active set is identified;
* at cell boundaries;
* when constraints are redundant.

The code attempts

[
(V+\sigma I)d=-R
]

with increasing (\sigma):

```python
regularization = 0.0

for _ in range(8):
    regularized = [row[:] for row in newton_matrix]

    for i in range(len(b)):
        regularized[i][i] += regularization

    try:
        direction = solve_dense(
            regularized,
            [-ri for ri in residual]
        )
        break
    except ArithmeticError:
        regularization = (
            1e-12
            if regularization == 0.0
            else 10.0 * regularization
        )
```

This is analogous to Levenberg–Marquardt stabilization.

Near a regular solution, the matrix should become nonsingular, regularization should vanish, and full Newton steps should be accepted. 

---

# 15. Why project the Newton direction?

The raw Newton direction (d) might produce

[
\lambda+d<0.
]

But dual multipliers must remain nonnegative.

So the code uses

[
p_N=[\lambda+d]_+-\lambda.
\tag{10}
]

This is the feasible projected Newton step.

The fallback projected-gradient step is

[
p_G
===

\left[
\lambda+\frac1{L_d}\nabla q_v(\lambda)
\right]_+
-\lambda.
\tag{11}
]

Both satisfy

[
\lambda+p\ge0.
]

---

# 16. How it decides whether Newton is trustworthy

A Newton direction may solve the linearized KKT system but fail to increase the dual objective.

Because (q_v) is being maximized, an ascent direction should satisfy

[
\nabla q_v(\lambda)^\top p>0.
]

The code compares the Newton slope with a fraction of the projected-gradient reference slope:

```python
newton_slope = dot(
    evaluation.gradient,
    projected_newton_step
)

reference_slope = lipschitz * dot(
    projected_gradient_step,
    projected_gradient_step
)

if newton_slope >= 1e-4 * reference_slope:
    search_direction = projected_newton_step
else:
    search_direction = projected_gradient_step
```

So Newton is used only when it is sufficiently ascent-producing.

Otherwise the algorithm falls back to the globally safer projected-gradient direction.

This is the first globalization safeguard.

---

# 17. Armijo line search in semismooth Newton

Even if (p) is an ascent direction, a full step (\alpha=1) may be too aggressive.

The algorithm tries

[
\lambda_{\text{trial}}
======================

\lambda+\alpha p
]

with

[
\alpha=1,\frac12,\frac14,\ldots
]

until the dual objective satisfies

[
q_v(\lambda+\alpha p)
\ge
q_v(\lambda)
+
c_1\alpha
\nabla q_v(\lambda)^\top p,
\tag{12}
]

where the code uses

[
c_1=10^{-4}.
]

This is the Armijo ascent condition.

In code:

```python
slope = dot(
    evaluation.gradient,
    search_direction
)

step_size = 1.0

for _ in range(40):
    candidate = add_scaled(
        multiplier,
        step_size,
        search_direction
    )

    candidate_evaluation = evaluate_dual(...)

    if candidate_evaluation.value >= (
        evaluation.value
        + 1e-4 * step_size * slope
    ):
        multiplier = candidate
        accepted = True
        break

    step_size *= 0.5
```

Each trial requires another dual evaluation, hence another PAVA call. This is why the benchmark tracks `pava_calls`, not only outer iterations. 

---

# 18. Numerical Armijo example

Suppose at the current multiplier:

[
q(\lambda)=10,
\qquad
\nabla q(\lambda)^\top p=2.
]

With (c_1=10^{-4}), a full step requires

[
q(\lambda+p)
\ge
10+10^{-4}\cdot1\cdot2
======================

10.0002.
]

Suppose the full Newton step gives

[
q(\lambda+p)=9.8.
]

Reject it.

Try (\alpha=0.5). The required increase is

[
10+10^{-4}\cdot0.5\cdot2
========================

10.0001.
]

Suppose

[
q(\lambda+0.5p)=10.03.
]

Accept (\alpha=0.5).

The condition is intentionally mild: it requires only a small fraction of the predicted linear improvement.

---

# 19. Why semismooth Newton can be extremely fast locally

Within a fixed affine cell:

* the PAVA sorting is unchanged;
* pools are unchanged;
* pool branches are unchanged;
* the positive-part active set is unchanged.

Then (R_v(\lambda)) is affine:

[
R_v(\lambda)=V\lambda+r.
]

Newton solves

[
Vd=-R_v(\lambda).
]

Therefore

[
R_v(\lambda+d)=0
]

in one exact step, provided:

* (V) is nonsingular;
* the step stays in the same cell;
* no projection or active-set pattern changes.

This explains the PDF’s statement that after identifying the correct affine cell and active set, one unregularized full Newton step can return the exact proximal solution in exact arithmetic. 

---

# 20. Algorithm 1 versus Algorithm 2

## Restarted FISTA

Strengths:

* simple;
* robust globally;
* only one PAVA call per ordinary iteration;
* does not need a generalized Jacobian;
* cheap when (m) is large.

Weaknesses:

* first-order;
* may need many iterations;
* restart length must be tuned.

## Semismooth Newton

Strengths:

* fast local convergence;
* can solve the correct affine cell essentially exactly;
* attractive when the number of constraints (m) is modest.

Weaknesses:

* must build and solve an (m\times m) Newton system;
* line search may require multiple PAVA evaluations;
* more sensitive to degeneracy and generalized-Jacobian choices.

The benchmark is designed to compare them on:

* wall-clock time;
* number of PAVA calls;
* certificate gap;
* projected KKT residual;
* agreement with Gurobi. 

---

# 21. Approximate computational costs

Let:

* (d) = dimension of (x);
* (m) = number of inequalities;
* (\operatorname{nnz}(A)) = nonzeros in (A).

A full PAVA call costs roughly:

[
O(d\log d)
]

because of sorting, followed by linear pooling.

The top-(k) optimized PAVA attempts to avoid a full sort when (k\ll d).

## One FISTA iteration

Approximately:

[
O(\operatorname{nnz}(A)+\text{PAVA}).
]

## One Newton iteration

In addition to prox evaluation, it computes (AJA^\top). The current pure-Python implementation loops over row pairs, so roughly:

[
O(m^2d)
]

in a dense interpretation, plus solving an (m\times m) system:

[
O(m^3).
]

Thus Newton is most attractive when (m\ll d).

---

# 22. The top-(k) PAVA optimization

The full oracle sorts all (d) magnitudes.

The alternative

```python
perspective_prox_state_topk
```

uses

```python
heapq.nlargest(k, ...)
```

to identify only the largest (k) coordinates, then determines the one possible mixed boundary pool using `_topk_boundary_root`.

The code’s high-level structural claim is:

* top coordinates before the boundary are singleton selected pools;
* one mixed pool may contain some top coordinates and some tail coordinates;
* the remaining tail is inactive.

This can reduce work when

[
k\ll d.
]

The dispatch logic is

```python
if method == "auto":
    method = "topk" if 8 * k < len(u) else "full"
```

The extensive self-check compares:

* prox output;
* JVP;
* Moreau envelope value,

between full and top-(k) implementations. 

---

# 23. What Gurobi is solving

Gurobi solves the same constrained proximal problem directly in variables ((x,z,t)):

[
x_i^2\le t_iz_i,
]

[
|x_i|\le Mz_i,
]

[
0\le z_i\le1,
\qquad
\sum_i z_i\le k,
]

[
Ax\le b.
]

The objective is

[
\frac12\sum_i t_i
+
\frac1{2\gamma}|x-v|^2.
]

This is a convex quadratically constrained program.

It serves as the oracle/reference solution against which FISTA and Newton are checked. 

---

# 24. The third file: line-search PDHG

The PDHG script solves a related full Markowitz model using primal–dual hybrid gradient, not the low-dimensional multiplier dual used in Algorithms 1 and 2.

Its variables include:

* primal portfolio (x);
* factor dual variables;
* constraint dual variables.

The primal step still calls the PAVA prox:

```python
primal_argument = (
    x
    - tau * adjoint_dual
    + tau
      * instance.return_reward
      * instance.expected_returns
)

x_next, main_pava_boundary = _pava_step(
    instance,
    primal_argument,
    tau,
    ...
)
```

So PAVA again handles the perspective regularizer in the primal update. 

---

# 25. Basic PDHG structure

In a generic saddle-point problem

[
\min_x f(x)+g(Kx),
]

PDHG alternates:

[
x^{k+1}
=======

\operatorname{prox}_{\tau f}
(x^k-\tau K^\top y^k),
]

[
\bar x^{k+1}
============

x^{k+1}
+\theta_k(x^{k+1}-x^k),
]

[
y^{k+1}
=======

\operatorname{prox}_{\sigma g^\ast}
(y^k+\sigma K\bar x^{k+1}).
]

In the script:

* the factor block represents quadratic factor risk;
* the constraint block represents interval constraints;
* the primal prox is the perspective/PAVA prox.

---

# 26. Factor dual update

The factor dual update is

```python
factor_trial = (
    factor_dual
    + sigma_factor_trial * extrapolated_factor
) / (1.0 + sigma_factor_trial)
```

This is the prox of a quadratic conjugate. It shrinks the dual update by

[
\frac1{1+\sigma}.
]

---

# 27. Constraint dual update

The constraint block uses

```python
constraint_trial = common.interval_dual_prox(
    constraint_value,
    sigma_constraint_trial,
    scaled_lower,
    scaled_upper,
)
```

This is the dual prox associated with interval constraints

[
l\le Cx\le u.
]

The exact formula is in the imported `markowitz_pdhg_benchmark` module, which was not among the three uploaded files, so the current files do not expose its detailed implementation. The present script clearly treats it as a separate dual proximal block. 

---

# 28. Malitsky–Pock line search

The PDHG script adapts the primal step (\tau).

After computing (x_{k+1}) once, it proposes

[
\tau_{\text{trial}}
===================

\tau_k\sqrt{1+\theta_k}.
]

Then it computes trial dual updates and checks a coupling condition.

Define dual differences

[
\Delta y_f
==========

y_{f,\text{trial}}-y_f,
]

[
\Delta y_c
==========

y_{c,\text{trial}}-y_c.
]

Their adjoint image is

[
K^\top\Delta y
==============

F\Delta y_f+C^\top\Delta y_c.
]

The acceptance test is

[
\tau_{\text{trial}}^2
|K^\top\Delta y|^2
\le
\delta^2
\left(
\frac{|\Delta y_f|^2}{\alpha_f}
+
\frac{|\Delta y_c|^2}{\alpha_c}
\right).
\tag{13}
]

If this fails, shrink

[
\tau_{\text{trial}}
\leftarrow
\beta\tau_{\text{trial}},
\qquad 0<\beta<1.
]

The code defaults to

[
\delta=0.999,\qquad \beta=0.9.
]



---

# 29. Intuition for the PDHG line-search condition

The left-hand side measures how strongly the dual change feeds back into the primal space:

[
|K^\top\Delta y|.
]

The right-hand side measures the size of the dual move in its block-scaled norm.

If the primal step (\tau) is too large, a modest dual change creates excessive primal coupling, and the inequality fails.

The line search shrinks (\tau) until coupling is safely controlled.

This replaces reliance on one globally conservative fixed step based on an exact operator norm.

---

# 30. Numerical PDHG line-search example

Suppose:

[
|\Delta y_f|^2=0.04,
\qquad
|\Delta y_c|^2=0.01,
]

[
\alpha_f=1,
\qquad
\alpha_c=0.25,
]

[
|K^\top\Delta y|^2=0.16,
\qquad
\delta=0.9.
]

The right side is

[
0.9^2
\left(
0.04+\frac{0.01}{0.25}
\right)
=======

# 0.81(0.04+0.04)

0.0648.
]

For (\tau_{\text{trial}}=0.8), the left side is

[
0.8^2\cdot0.16
==============

0.1024.
]

This fails because

[
0.1024>0.0648.
]

If the shrink factor is (0.9), try

[
\tau=0.72.
]

Then

[
0.72^2\cdot0.16
===============

0.082944,
]

still too large.

Try

[
\tau=0.648.
]

Then

[
0.648^2\cdot0.16
\approx0.0672,
]

still slightly high.

Try

[
\tau=0.5832.
]

Then

[
0.5832^2\cdot0.16
\approx0.0544,
]

so the trial is accepted.

---

# 31. Why rejected PDHG trials do not call PAVA again

This is a useful design detail.

The primal step (x_{\text{next}}) is computed before the line-search loop:

```python
x_next, main_pava_boundary = _pava_step(...)
```

Then the line search changes only the trial dual step sizes and dual updates.

Thus rejected trials reuse:

* the same (x_{\text{next}});
* the same factor image;
* the same constraint image.

They do not rerun PAVA.

The script’s docstring emphasizes that each main iteration evaluates the PAVA proximal oracle exactly once, aside from extra diagnostic residual calls. 

---

# 32. Gram-matrix acceleration in the PDHG line search

The line search needs

[
|F\Delta y_f+C^\top\Delta y_c|^2.
]

Computing this directly costs work proportional to the portfolio dimension.

Instead, the script precomputes small Gram matrices:

[
F^\top F,
\qquad
F^\top C^\top,
\qquad
CC^\top.
]

Then

[
|F\Delta y_f+C^\top\Delta y_c|^2
]

can be evaluated entirely in dual dimensions.

That makes rejected line-search trials largely independent of the large portfolio dimension. 

---

# 33. Adaptive restart in PDHG

The PDHG code forms two candidate points:

* the most recent iterate;
* a step-size-weighted average over the current epoch.

It computes residuals for both and picks the better one.

A restart occurs when either:

[
\text{candidate residual}
\le
\text{restart factor}
\times
\text{anchor residual},
]

after the minimum epoch length, or the maximum epoch length is reached.

On restart:

* set the current state to the chosen candidate;
* reset (\theta=1);
* clear weighted averages;
* reset the PAVA boundary warm start;
* start a new epoch.

This is analogous in spirit to FISTA restart: discard stale extrapolation/history after sufficient progress or after an epoch becomes too long. 

---

# 34. Do not confuse the three line-search/restart concepts

There are several mechanisms with similar names.

## FISTA restart

Every fixed number of dual iterations:

[
y\leftarrow\lambda,\qquad t\leftarrow1.
]

Purpose: remove acceleration momentum.

## Newton Armijo line search

Shrinks (\alpha) along a chosen multiplier direction until the dual objective has sufficient ascent.

Purpose: globalize Newton.

## PDHG Malitsky–Pock line search

Shrinks the primal step (\tau) until a primal–dual coupling inequality holds.

Purpose: choose a stable PDHG step adaptively.

## PDHG adaptive restart

Replaces the current state with the best epoch candidate when residual improvement is sufficient or the epoch is too long.

Purpose: improve global convergence behavior.

These are four different devices.

---

# 35. Overall code architecture of `prox_compare(1).py`

A useful map:

## Layer 1: basic linear algebra

* `dot`
* `norm2`
* `matvec`
* `tmatvec`
* `positive_part`

## Layer 2: PAVA structures

* `Pool`
* `PavaState`
* `_make_pool`
* `_assemble_pava_state`

## Layer 3: base proximal oracle

* `perspective_prox_state`
* `perspective_prox_state_topk`
* `perspective_prox_state_dispatch`

Output:

* prox point;
* envelope;
* generalized JVP;
* pool signature.

## Layer 4: primal objective and (G)

* `perspective_value`
* `primal_objective`

## Layer 5: dual evaluation

* `evaluate_dual`

Output:

* multiplier;
* PAVA state;
* primal point;
* dual gradient;
* dual value.

## Layer 6: certificate

* `strict_zero_anchor_certificate`

## Layer 7: outer algorithms

* `restarted_dual_fista`
* `globalized_semismooth_newton`

## Layer 8: reference and benchmarking

* `solve_with_gurobi`
* data generation;
* tests;
* plots;
* CSV summaries.



---

# 36. The single most important mental diagram

For Algorithm 1:

[
\lambda^j
\to
y^j
\to
v-\gamma A^\top y^j
\to
\boxed{\text{PAVA}}
\to
x(y^j)
\to
Ax(y^j)-b
\to
\lambda^{j+1}.
]

For Algorithm 2:

[
\lambda^j
\to
\boxed{\text{PAVA state}}
\to
x(\lambda^j),\ J
]

[
\to
R(\lambda^j),\ V
\to
d
\to
p_N\text{ or }p_G
\to
\boxed{\text{Armijo}}
\to
\lambda^{j+1}.
]

For PDHG:

[
(x^k,y^k)
\to
\boxed{\text{one PAVA primal step}}
\to
x^{k+1}
\to
\text{trial dual updates}
\to
\boxed{\text{Malitsky--Pock test}}
\to
y^{k+1}.
]

---

# 37. What to study first

A sensible order is:

1. Understand
   [
   x(\lambda)
   ==========

   \operatorname{prox}_{\gamma G}
   (v-\gamma A^\top\lambda).
   ]

2. Understand
   [
   \nabla q(\lambda)=Ax(\lambda)-b.
   ]

3. Understand how PAVA produces both:

   * (x(\lambda));
   * a fixed-cell JVP (Jh).

4. Understand FISTA as projected gradient ascent plus momentum.

5. Understand the projected KKT residual
   [
   R(\lambda)
   ==========

   \lambda-[\lambda+c(Ax-b)]_+.
   ]

6. Understand Newton’s matrix
   [
   V=I-E+c\gamma EAJA^\top.
   ]

7. Understand globalization:

   * Newton-versus-gradient direction selection;
   * Armijo backtracking.

8. Treat the PDHG file as a separate algorithm after the first two are clear.

The PDF itself is only two pages, so many implementation details—especially the exact PAVA construction, top-(k) shortcut, certificate code, and line-search safeguards—are supplied by the Python rather than formally derived in the PDF.


## 1. Where does the dual function (q_v(\lambda)) come from?

The primal constrained proximal problem is

[
\min_{Ax\le b}
\left{
G(x)+\frac{1}{2\gamma}|x-v|^2
\right}.
\tag{P}
]

Introduce a nonnegative Lagrange multiplier

[
\lambda\ge0
]

for (Ax-b\le0). The Lagrangian is

[
\mathcal L(x,\lambda)
=====================

G(x)
+
\frac{1}{2\gamma}|x-v|^2
+
\lambda^\top(Ax-b).
\tag{1}
]

The dual function is defined by minimizing the Lagrangian over (x):

[
q_v(\lambda)
:=
\inf_x \mathcal L(x,\lambda).
\tag{2}
]

The PDF states that this becomes

[
\boxed{
q_v(\lambda)
============

e_{\gamma G}(v-\gamma A^\top\lambda)
+
\lambda^\top(Av-b)
------------------

\frac{\gamma}{2}|A^\top\lambda|^2,
}
\tag{3}
]

and that the dual problem is

[
\max_{\lambda\ge0}q_v(\lambda).
]

The coefficient of the final term is (\gamma/2), not (2\gamma). 

### Derivation by completing the square

Focus on the terms involving (x):

[
\frac{1}{2\gamma}|x-v|^2+\lambda^\top Ax.
]

Let

[
a:=A^\top\lambda.
]

Then

[
\frac{1}{2\gamma}|x-v|^2+a^\top x.
]

We want to express this as a shifted squared distance. Expand:

[
\frac{1}{2\gamma}|x-v|^2
========================

\frac{1}{2\gamma}|x|^2
-\frac1\gamma v^\top x
+\frac{1}{2\gamma}|v|^2.
]

Adding (a^\top x),

[
\frac{1}{2\gamma}|x|^2
-\frac1\gamma (v-\gamma a)^\top x
+\frac{1}{2\gamma}|v|^2.
]

Now complete the square around (v-\gamma a):

[
\frac{1}{2\gamma}
|x-(v-\gamma a)|^2
+
a^\top v
--------

\frac{\gamma}{2}|a|^2.
\tag{4}
]

You can verify this by expansion.

Substitute (a=A^\top\lambda):

[
\frac{1}{2\gamma}|x-v|^2+\lambda^\top Ax
========================================

\frac{1}{2\gamma}
|x-(v-\gamma A^\top\lambda)|^2
+
\lambda^\top Av
---------------

\frac{\gamma}{2}|A^\top\lambda|^2.
]

Therefore,

[
\mathcal L(x,\lambda)
=====================

G(x)
+
\frac{1}{2\gamma}
|x-(v-\gamma A^\top\lambda)|^2
+
\lambda^\top(Av-b)
------------------

\frac{\gamma}{2}|A^\top\lambda|^2.
]

Only the first two terms depend on (x). Their minimum is, by definition, the Moreau envelope:

[
e_{\gamma G}(w)
===============

\min_x
\left{
G(x)+\frac{1}{2\gamma}|x-w|^2
\right}.
]

Set

[
w=v-\gamma A^\top\lambda.
]

This gives equation (3).

### Where does (x(\lambda)) come from?

The minimizer in the definition of that envelope is

[
\boxed{
x(\lambda)
==========

\operatorname{prox}_{\gamma G}
(v-\gamma A^\top\lambda).
}
\tag{5}
]

So the same completion-of-the-square step gives both:

* the dual objective (q_v(\lambda));
* the primal response (x(\lambda)) at a given multiplier.

### Why is the gradient (Ax(\lambda)-b)?

The PDF gives

[
\nabla q_v(\lambda)=Ax(\lambda)-b.
\tag{6}
]

Intuitively, the derivative of the dual value with respect to (\lambda_i) measures how much the corresponding constraint is violated:

[
\frac{\partial q_v}{\partial\lambda_i}
======================================

A_i x(\lambda)-b_i.
]

Thus:

* positive gradient: constraint violated, increase (\lambda_i);
* negative gradient: constraint slack, decrease (\lambda_i);
* zero gradient with (\lambda_i>0): active constraint.

The implementation computes exactly this in `evaluate_dual`. 

---

## 2. In ((V+\sigma_j I)d=-R(\lambda)), what is (d)?

The vector

[
\boxed{d}
]

is the **raw semismooth Newton correction in multiplier space**.

It has the same dimension as (\lambda), namely the number (m) of inequality constraints:

[
\lambda,d,R(\lambda)\in\mathbb R^m.
]

The objective is to solve the projected KKT equation

[
R(\lambda)=0,
]

where

[
R(\lambda)
==========

\lambda-
[\lambda+c(Ax(\lambda)-b)]_+.
\tag{7}
]

At the current (\lambda), linearize this residual:

[
R(\lambda+d)
\approx
R(\lambda)+Vd,
]

where (V) is a generalized Jacobian of (R).

To make the linearized residual zero, require

[
R(\lambda)+Vd=0,
]

which gives the Newton system

[
Vd=-R(\lambda).
]

The implemented system adds regularization:

[
\boxed{
(V+\sigma_j I)d=-R(\lambda).
}
\tag{8}
]

Here (\sigma_j\ge0) stabilizes the solve if (V) is singular or nearly singular.

### Simple scalar example

Suppose there is one constraint, so (\lambda\in\mathbb R), and at the current point:

[
R(\lambda)=0.12,
\qquad
V=0.8,
\qquad
\sigma=0.
]

Then

[
0.8d=-0.12,
]

so

[
d=-0.15.
]

The raw Newton proposal is

[
\lambda+d.
]

If (\lambda=0.5), this gives

[
0.5-0.15=0.35.
]

The interpretation is: according to the local affine model of the KKT residual, reducing the multiplier by (0.15) should drive the residual to zero.

### Why (d) is not directly used as the final step

Because (\lambda) must satisfy

[
\lambda\ge0.
]

The raw Newton proposal (\lambda+d) might be negative in some coordinates. Therefore the algorithm constructs the feasible projected Newton direction

[
\boxed{
p_N=[\lambda+d]_+-\lambda.
}
\tag{9}
]

Then

[
\lambda+p_N=[\lambda+d]_+\ge0.
]

So distinguish:

* (d): raw Newton correction obtained from the linear system;
* (p_N): projected feasible Newton direction;
* (\alpha p_N): actual step after line search.

The update is ultimately

[
\lambda^+
=========

\lambda+\alpha p.
]

---

## 3. Newton direction versus projected-gradient direction

The algorithm constructs two candidate directions.

### Projected Newton direction

First solve

[
(V+\sigma I)d=-R(\lambda).
]

Then project:

[
\boxed{
p_N=[\lambda+d]_+-\lambda.
}
]

It uses curvature and active-cell information through

[
V=I-E+c\gamma EAJA^\top.
]

Here:

* (J) comes from the current PAVA pools;
* (E) identifies the active part of the positive-part map;
* (AJA^\top) describes how the primal prox response changes when multipliers change.

Thus Newton asks:

> Using the local piecewise-affine model of the entire KKT system, what multiplier change would drive the residual directly to zero?

### Projected-gradient direction

The safe first-order direction is

[
\boxed{
p_G
===

\left[
\lambda+\frac1{L_d}\nabla q_v(\lambda)
\right]_+
-\lambda,
}
\tag{10}
]

where

[
\nabla q_v(\lambda)=Ax(\lambda)-b
]

and

[
L_d=\gamma|A|^2.
]

This asks:

> Move uphill in the dual objective using only its gradient, with a conservative Lipschitz-based step, then project onto (\lambda\ge0).

### Main conceptual difference

| Direction | Information used                                     | Typical behavior                |
| --------- | ---------------------------------------------------- | ------------------------------- |
| (p_G)     | Gradient only                                        | Robust but slower               |
| (p_N)     | Gradient/KKT residual plus generalized curvature (V) | Potentially much faster locally |
| (d)       | Raw Newton system solution                           | May violate (\lambda\ge0)       |
| (p_N)     | Projection of raw Newton proposal                    | Feasible multiplier direction   |

### Numerical comparison

Suppose

[
R(\lambda)=
\begin{pmatrix}
0.20\
-0.05
\end{pmatrix},
\qquad
V=
\begin{pmatrix}
1&0\
0&0.5
\end{pmatrix}.
]

Newton solves

[
Vd=-R,
]

giving

[
d=
\begin{pmatrix}
-0.20\
0.10
\end{pmatrix}.
]

If

[
\lambda=
\begin{pmatrix}
0.10\
0.20
\end{pmatrix},
]

then

[
\lambda+d=
\begin{pmatrix}
-0.10\
0.30
\end{pmatrix}.
]

Projection gives

[
[\lambda+d]_+
=============

\begin{pmatrix}
0\
0.30
\end{pmatrix},
]

hence

[
p_N=
\begin{pmatrix}
-0.10\
0.10
\end{pmatrix}.
]

Now suppose the dual gradient and Lipschitz constant are

[
\nabla q(\lambda)
=================

\begin{pmatrix}
-0.08\
0.04
\end{pmatrix},
\qquad
L_d=2.
]

Then

[
\lambda+\frac1{L_d}\nabla q
===========================

\begin{pmatrix}
0.10\
0.20
\end{pmatrix}
+
\frac12
\begin{pmatrix}
-0.08\
0.04
\end{pmatrix}
=============

\begin{pmatrix}
0.06\
0.22
\end{pmatrix}.
]

This is already nonnegative, so

[
p_G=
\begin{pmatrix}
-0.04\
0.02
\end{pmatrix}.
]

The Newton direction is larger and attempts to remove the residual much more aggressively:

[
p_N=(-0.10,0.10),
]

while gradient uses the conservative move

[
p_G=(-0.04,0.02).
]

### Why the algorithm sometimes rejects (p_N)

A local Newton model can be inaccurate when:

* the PAVA partition is about to change;
* the active constraint set is about to change;
* the Newton matrix is ill-conditioned;
* the raw step crosses several piecewise-affine cells.

The code checks whether (p_N) is a sufficiently good ascent direction:

[
\nabla q(\lambda)^\top p_N
\ge
10^{-4}L_d|p_G|^2.
]

If this fails, it uses (p_G). 

Then Armijo backtracking determines how far to move along the selected direction.

---

## 4. In PDHG, what are (y_f,y_c,\alpha_f,\alpha_c)?

The code names these variables:

* `factor_dual`;
* `constraint_dual`;
* `alpha_factor`;
* `alpha_constraint`.

I will denote them by

[
y_f,\quad y_c,\quad \alpha_f,\quad\alpha_c.
]

The uploaded PDHG script does not contain the full model decomposition; it imports several definitions from `markowitz_pdhg_benchmark`, which was not included in the three files. However, the roles of these quantities are clear from the update equations in the uploaded script. 

### (y_f): factor-risk dual variable

The primal portfolio (x) is mapped to factor exposures by

[
F^\top x,
]

where `instance.factor_loadings` acts as (F).

The code stores

```python
factor_image = instance.factor_loadings.T @ x
```

so

[
K_f x=F^\top x.
]

The dual variable associated with this factor block is

[
\boxed{y_f=\texttt{factor_dual}.}
]

Its update is

[
y_f^+
=====

\frac{
y_f+\sigma_f\bar f
}{
1+\sigma_f
},
\tag{11}
]

where (\bar f) is the extrapolated factor exposure.

This division by (1+\sigma_f) is characteristic of the proximal map of a quadratic conjugate. Thus (y_f) carries the dual information associated with factor-risk exposure.

### (y_c): linear-constraint dual variable

The normalized constraint matrix is denoted here by (C), and the code computes

```python
constraint_image = scaled_c @ x
```

so

[
K_cx=Cx.
]

The dual variable for the interval constraints is

[
\boxed{y_c=\texttt{constraint_dual}.}
]

It is updated using

```python
constraint_trial = common.interval_dual_prox(...)
```

which enforces the dual prox corresponding to interval bounds such as

[
l\le Cx\le u.
]

Thus:

* (y_f): dual variable for factor-risk block;
* (y_c): dual variable for the explicit interval/portfolio-constraint block.

The combined adjoint contribution to the primal update is

[
\boxed{
K^\top y
========

Fy_f+C^\top y_c.
}
\tag{12}
]

The code stores this as `adjoint_dual`.

---

## What are (\alpha_f) and (\alpha_c)?

They are block scaling parameters that tie the two dual step sizes to the primal step (\tau).

The code defines

```python
alpha_factor = step_ratio * step_ratio
alpha_constraint = (
    alpha_factor * constraint_dual_weight
)
```

and then, for each trial (\tau_{\text{trial}}),

```python
sigma_factor_trial = (
    tau_trial * alpha_factor
)
sigma_constraint_trial = (
    tau_trial * alpha_constraint
)
```

Thus

[
\boxed{
\sigma_f=\tau\alpha_f,
\qquad
\sigma_c=\tau\alpha_c.
}
\tag{13}
]

They are not dual variables. They determine the relative sizes of the dual steps.

Concretely,

[
\alpha_f=(\text{step ratio})^2,
]

and

[
\alpha_c
========

\alpha_f
\times
(\text{constraint dual weight}).
]

### Why use different block scalings?

The factor block and constraint block may have very different numerical scales.

For example:

* factor exposures may be naturally moderate;
* normalized constraint rows may produce larger or smaller dual responses;
* one block may dominate the line-search norm.

Using separate (\alpha_f,\alpha_c) is a form of block preconditioning.

A larger (\alpha_c) means

[
\sigma_c=\tau\alpha_c
]

is larger, so the constraint dual variables respond more aggressively relative to the factor dual variables.

### Numerical example

Suppose

[
\text{step ratio}=0.5,
]

so

[
\alpha_f=0.5^2=0.25.
]

Suppose

[
\text{constraint dual weight}=4.
]

Then

[
\alpha_c=0.25\cdot4=1.
]

If the current primal step is

[
\tau=0.2,
]

then

[
\sigma_f=\tau\alpha_f=0.2(0.25)=0.05,
]

while

[
\sigma_c=\tau\alpha_c=0.2(1)=0.2.
]

Thus the constraint dual block moves with a step four times as large as the factor block.

---

## How (\alpha_f,\alpha_c) enter the line-search condition

The line search accepts (\tau_{\text{trial}}) when

[
\tau_{\text{trial}}^2
\left|
F\Delta y_f+C^\top\Delta y_c
\right|^2
\le
\delta^2
\left(
\frac{|\Delta y_f|^2}{\alpha_f}
+
\frac{|\Delta y_c|^2}{\alpha_c}
\right).
\tag{14}
]

The right side is a block-weighted dual norm.

If (\alpha_c) is larger, then

[
\frac{|\Delta y_c|^2}{\alpha_c}
]

is smaller, making large constraint-dual changes harder to accept unless their adjoint effect is controlled. At the same time, the actual dual step (\sigma_c=\tau\alpha_c) is larger. The combined design balances update aggressiveness with the line-search metric.

---

## Compact notation map

[
\boxed{
x(\lambda)
==========

\operatorname{prox}_{\gamma G}
(v-\gamma A^\top\lambda)
}
]

is the primal response in Algorithms 1 and 2.

[
\boxed{
q_v(\lambda)
}
]

is the concave multiplier-dual objective.

[
\boxed{
R(\lambda)
==========

\lambda-[\lambda+c(Ax(\lambda)-b)]_+
}
]

is the projected KKT residual.

[
\boxed{
d
}
]

is the raw Newton correction solving the regularized linearized KKT system.

[
\boxed{
p_N=[\lambda+d]_+-\lambda
}
]

is the projected Newton direction.

[
\boxed{
p_G=
[\lambda+L_d^{-1}\nabla q(\lambda)]_+-\lambda
}
]

is the projected-gradient direction.

In PDHG:

[
\boxed{
y_f
}
]

is the factor-risk dual variable,

[
\boxed{
y_c
}
]

is the interval-constraint dual variable, and

[
\boxed{
\sigma_f=\tau\alpha_f,\quad
\sigma_c=\tau\alpha_c
}
]

are the blockwise dual step sizes.

