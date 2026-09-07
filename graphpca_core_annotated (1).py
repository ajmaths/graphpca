# %% [markdown]
# # GraphPCA for weighted simple graphs
#
# Let $G=(V,E)$ be a finite weighted graph with $n$ vertices and $m$ edges.
# The edge lengths are collected in
#
# $$
# \ell=(\ell_e)_{e\in E},
# \qquad C=\operatorname{diag}(\ell),
# $$
#
# and the edge-flow inner product is
#
# $$
# \langle a,b\rangle_C=a^TCb
# =\sum_{e\in E}\ell_ea_eb_e.
# $$
#
# Each vertex $y$ is assigned a scalar field $S_t(\cdot,y)$ at diffusion time
# $t$. The current experiments use either the heat-kernel column
# $H_t(\cdot,y)$ or the diffusion-distance potential
# $d_t(\cdot,y)^2/(2n)$. If $K$ is the length-scaled incidence operator, the
# corresponding edge flow is
#
# $$
# g_y=K^TS_t(\cdot,y)\in\mathbb R^m.
# $$
#
# For a probability measure $\mu$ on the vertices, GraphPCA first centers the
# flows:
#
# $$
# \bar g=\sum_y\mu_y g_y,
# \qquad \widetilde g_y=g_y-\bar g.
# $$
#
# Put the centered flows into the weighted matrix
#
# $$
# U=C^{1/2}
# \begin{bmatrix}
# \sqrt{\mu_1}\,\widetilde g_1&\cdots&
# \sqrt{\mu_n}\,\widetilde g_n
# \end{bmatrix}.
# $$
#
# If
#
# $$
# U=P\Sigma Q^T,
# $$
#
# then the GraphPCA eigenvalues and principal edge fields are
#
# $$
# \lambda_j=\sigma_j^2,
# \qquad v_j=C^{-1/2}P_j.
# $$
#
# Thus the left singular vectors are the eigenvectors of the whitened
# edge-space covariance $UU^T$. Dividing by $C^{1/2}$ returns to the original
# edge metric, where
#
# $$
# \langle v_j,v_k\rangle_C=\delta_{jk}.
# $$
#
# The score of vertex $y$ on the $j$th principal edge field is
#
# $$
# \alpha_j(y)=\langle\widetilde g_y,v_j\rangle_C
# =\widetilde g_y^TCv_j.
# $$
#
# The implementation below follows this construction in stages: Laplacian
# primitives, scalar potentials, edge flows, graph preparation, centered PCA,
# mathematical checks, heat-time sweeps, degenerate eigenspaces, and plotting.

# %%
"""A readable, self-contained implementation of GraphPCA.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "GraphPCAResult", "graphpca", "check_graphpca",
    "graph_heat_kernel", "diffusion_distance_potential", "laplacian_eigh",
    "edge_flows", "rank_capacity",
    "GraphSetup", "prepare_graph", "edge_length_affinities",
    "AffinityConnectivity", "affinity_connectivity_report", "WEAK_GAP_RTOL",
    "fit_graphpca", "GraphPCAFit",
    "degenerate_blocks", "block_magnitude", "block_angle", "block_summary",
    "check_block_invariance", "align_pc_signs",
    "plot_eigenvalues", "plot_variance_explained", "plot_cumulative_variance",
    "plot_spectrum_across_t", "plot_block_magnitude",
]

# Palette for PC index: all six colour checks pass on a light background.
# Marker shapes give a second encoding so figures survive greyscale printing.
PC_COLORS = ["#3B6FD4", "#E4572E", "#1B9E77", "#9467BD"]
PC_MARKERS = ["o", "s", "^", "D"]
INK, MUTED = "#1a1a1a", "#6b6b6b"
SEQUENTIAL_CMAP = "viridis"        # diffusion time is ordered, so: a ramp

# Once the heat kernel has equilibrated (H_t -> constant), subtracting the
# mean leaves nothing but round-off. Below this ratio the answer is noise.
# No eigenvalue threshold can catch this: when the whole spectrum is noise,
# the noise still clears a threshold measured relative to itself.
CANCELLATION_RTOL = 1e7 * np.finfo(float).eps


# %% [markdown]
# ## 1. Laplacian eigendecomposition
#
# The graph Laplacian is a symmetric positive-semidefinite matrix
#
# $$
# L=\Phi\Lambda\Phi^T,
# $$
#
# with eigenvalues returned in ascending order. A connected graph has one
# zero eigenvalue. Numerically, that zero may appear as a tiny negative value,
# so `laplacian_eigh` clips only negative values consistent with roundoff. It
# does not erase small positive eigenvalues, because a small spectral gap can
# describe a genuinely weakly connected graph.
#
# `_dense` lets the same validation work for NumPy arrays and SciPy sparse
# matrices.

# %%
def _dense(A):
    """Accept a SciPy sparse matrix wherever a dense one is expected.

    np.asarray on a sparse matrix quietly makes a 0-d object array, which
    then fails a long way from the cause.
    """
    if hasattr(A, "toarray") and not isinstance(A, np.ndarray):
        return np.asarray(A.toarray(), float)
    return np.asarray(A, float)


def laplacian_eigh(laplacian, zero_tol=1e-12):
    """Eigendecompose a symmetric positive-semidefinite graph Laplacian.

    Eigenvalues are returned in ascending order. A Laplacian is positive
    semidefinite in exact arithmetic, but its zero mode may be reported as a
    tiny negative number by floating-point eigensolvers. We clip only those
    round-off negatives; we do NOT erase small positive eigenvalues. That
    distinction matters for a weakly connected graph, whose genuine spectral
    gap can be much smaller than 1e-12 in the units used for ``L``.

    A negative eigenvalue that is LARGE relative to the spectrum is not
    round-off -- it means the input was not a Laplacian -- and raises
    ``ValueError`` rather than being clipped. The tolerance is relative
    (``-1e-12 * max(|lambda|)``), so it does not move when ``L`` is rescaled.

    ``zero_tol`` is retained and validated for API compatibility, but it no
    longer controls the symmetry tolerance: ``zero_tol=0`` must remain a
    usable setting in floating-point arithmetic. Positive eigenvalues are
    never removed. This follows GraphPCA 0.8.0's
    ``laplacian_eigendecomposition`` convention, with the added non-PSD guard.
    """
    matrix = _dense(laplacian)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("laplacian must be a square matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("laplacian must contain only finite values.")
    zero_tol = float(zero_tol)
    if not np.isfinite(zero_tol) or zero_tol < 0:
        raise ValueError("zero_tol must be finite and nonnegative.")

    # Symmetry tolerance is FIXED (the package's 1e-12/1e-14), not zero_tol:
    # zero_tol=0 is a legal argument value and must not reject a Laplacian
    # whose asymmetry is one ulp of round-off.
    if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-14):
        raise ValueError("laplacian must be symmetric.")

    try:                                   # scipy gives better residuals
        from scipy.linalg import eigh
        values, vectors = eigh(matrix)
    except ImportError:
        values, vectors = np.linalg.eigh(matrix)

    # We only clip eigensolver round-off here. A negative eigenvalue LARGE 
    # relative to the spectrum is not round-off: it means the input was not a 
    # Laplacian at all. The usual causes are passing the adjacency W where L 
    # was meant, passing -L, or a weight matrix with genuinely signed entries. 
    # Clipping those to zero produces a plausible-looking spectrum from a 
    # meaningless input.
    #
    # So: tolerate round-off scaled to the spectrum, reject anything bigger.
    # The threshold is RELATIVE, for the same reason the zero-mode rule above
    # is: scaling L is only a change of time units and must not change the
    # verdict.
    # Do not floor this at 1.0: doing so makes the verdict depend on the units
    # of L and lets a rescaled adjacency matrix or -L slip through.
    scale = max(float(np.max(np.abs(values))) if values.size else 0.0,
                np.finfo(float).tiny)
    if np.any(values < -1e-12 * scale):
        raise ValueError(
            f"laplacian has eigenvalue {values.min():.3e}, which is "
            f"{abs(values.min()) / scale:.1e} of the spectrum -- too large to "
            f"be round-off. It is not positive semidefinite, so it is not a "
            f"graph Laplacian. Check you did not pass the adjacency matrix, "
            f"-L, or a signed weight matrix.")
    return np.maximum(values, 0.0), vectors


# %% [markdown]
# ## 2. Scalar fields attached to the vertices
#
# GraphPCA begins with one scalar field for each vertex. This module supplies
# two choices.
#
# The heat-kernel choice is
#
# $$
# H_t=\exp(-tL)=\Phi\exp(-t\Lambda)\Phi^T.
# $$
#
# Its $y$th column is the scalar field $S_t(\cdot,y)=H_t(\cdot,y)$.
#
# The alternative diffusion-distance potential uses the spectral embedding
#
# $$
# X_t=\Phi_+\exp(-t\Lambda_+),
# $$
#
# where the zero modes are omitted, and sets
#
# $$
# S_t(x,y)=\frac{d_t(x,y)^2}{2n},
# \qquad d_t(x,y)=\|X_t(x)-X_t(y)\|.
# $$
#
# The downstream GraphPCA calculation is identical for either choice.

# %%
def graph_heat_kernel(lap_values, lap_vectors, t):
    """Return ``H_t = exp(-t L)`` from a Laplacian eigendecomposition.

    All modes are retained, including the constant zero mode. The final
    symmetrisation removes the one-ulp asymmetry left by matrix multiplication.
    """
    values = np.asarray(lap_values, float).ravel()
    vectors = np.asarray(lap_vectors, float)
    t = float(t)
    if vectors.ndim != 2 or vectors.shape[1] != values.size:
        raise ValueError("lap_vectors must have one column per eigenvalue.")
    if vectors.shape[0] == 0:
        raise ValueError("lap_vectors must contain at least one vertex row.")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(vectors)):
        raise ValueError("the Laplacian eigendecomposition must be finite.")
    if not np.isfinite(t) or t < 0:
        raise ValueError("t must be finite and nonnegative.")
    scale = max(float(np.max(np.abs(values))) if values.size else 0.0,
                np.finfo(float).tiny)
    if np.any(values < -1e-12 * scale):
        raise ValueError("heat-kernel eigenvalues must be nonnegative.")
    values = np.maximum(values, 0.0)
    decay = np.exp(-t * values)
    heat = (vectors * decay) @ vectors.T
    return 0.5 * (heat + heat.T)


def diffusion_distance_potential(lap_values, lap_vectors, t, zero_rtol=1e-12):
    """The OTHER potential: D_t[i,j] = d_t(i,j)^2 / (2n).

    Here d_t is the diffusion distance, ||X_t[i] - X_t[j]|| with
    X_t = Phi exp(-t lam) over the NON-ZERO modes. This is what the math-core
    notebook uses; the current notebooks use `graph_heat_kernel` instead.
    Both are just a choice of scalar field per vertex -- `graphpca()` does not
    care which one you hand it. The package spells the choice
    `potential_function="heat_kernel"` or `"diffusion_potential"`; the bare
    `"diffusion"` is an older spelling of THIS file's, still accepted by
    `fit_graphpca` below and normalised to the package's form.

    The zero modes are dropped: a constant eigenvector contributes the same
    value to every vertex, so it cancels out of a difference anyway.
    """
    values = np.asarray(lap_values, float).ravel()
    vectors = np.asarray(lap_vectors, float)
    t = float(t)
    zero_rtol = float(zero_rtol)
    if vectors.ndim != 2 or vectors.shape[1] != values.size:
        raise ValueError("lap_vectors must have one column per eigenvalue.")
    if vectors.shape[0] == 0:
        raise ValueError("lap_vectors must contain at least one vertex row.")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(vectors)):
        raise ValueError("the Laplacian eigendecomposition must be finite.")
    if not np.isfinite(t) or t < 0:
        raise ValueError("t must be finite and nonnegative.")
    if not np.isfinite(zero_rtol) or zero_rtol < 0:
        raise ValueError("zero_rtol must be finite and nonnegative.")
    n = vectors.shape[0]
    # Purely relative: the old max(lambda_max, 1.0) floor could discard every
    # genuine mode after a harmless rescaling of the Laplacian. The empty
    # guard keeps a zero-mode-only spectrum from crashing max() below.
    scale = max(float(np.abs(values).max()) if values.size else 0.0,
                np.finfo(float).tiny)
    keep = values > zero_rtol * scale
    X = vectors[:, keep] * np.exp(-t * values[keep])
    gram = X @ X.T
    diagonal = np.diag(gram)
    d_squared = np.maximum(diagonal[:, None] + diagonal[None, :] - 2.0 * gram, 0.0)
    np.fill_diagonal(d_squared, 0.0)
    return 0.5 * d_squared / n


# %% [markdown]
# ## 3. Edge flows and the rank ceiling
#
# If `D[:, y]` stores the scalar field associated with vertex $y$, applying
# the graph gradient gives
#
# $$
# g_y=K^TD_{:,y}.
# $$
#
# `edge_flows` evaluates all of these gradients simultaneously, producing an
# $m\times n$ matrix. This matrix may be precomputed and reused when several
# vertex measures are analyzed on the same graph.
#
# If $n_\mu$ vertices have positive mass, centering leaves at most
# $n_\mu-1$ independent columns. Since the edge space has dimension $m$, the
# number of nonzero principal components cannot exceed
#
# $$
# \min(m,n_\mu-1).
# $$

# %%
def edge_flows(D, K):
    """Return all uncentred flows ``g_y = K.T @ S_t[:, y]`` at once.

    The output has shape ``(m, n)``. In experiments such as MNIST, compute it
    once per diffusion time and reuse it for every image measure via the
    ``flows=`` argument of :func:`graphpca`.
    """
    Dmat, Kmat = _dense(D), _dense(K)
    if Kmat.ndim != 2 or not np.all(np.isfinite(Kmat)):
        raise ValueError("K must be a finite array of shape (n, m).")
    n, _ = Kmat.shape
    if Dmat.shape != (n, n) or not np.all(np.isfinite(Dmat)):
        raise ValueError(f"D must be a finite array of shape ({n}, {n}).")
    return Kmat.T @ Dmat


def rank_capacity(n_atoms, m):
    """How many components can exist: min(m, n_atoms - 1).

    Centring costs one dimension, and a measure living on a single vertex
    gives a covariance that is identically zero.
    """
    return int(max(min(int(m), int(n_atoms) - 1), 0))


# %% [markdown]
# ## 4. From a weighted graph to $L$, $K$, and $\ell$
#
# `prepare_graph` converts a simple undirected NetworkX graph into the arrays
# used by GraphPCA. The edge attribute named by `weight` is interpreted as a
# positive edge length $\ell_e$, not as an affinity.
#
# The conversion has four steps:
#
# 1. Fix the vertex and edge orderings and collect the edge lengths.
# 2. Convert lengths to Gaussian affinities
#
#    $$
#    a_e=\exp\!\left(-\frac{\ell_e^2}{2\sigma^2}\right).
#    $$
#
# 3. Form the affinity adjacency matrix $W$ and Laplacian
#    $L=\operatorname{diag}(W\mathbf 1)-W$.
# 4. Form an oriented incidence matrix $B$ and the length-scaled incidence
#    operator
#
#    $$
#    K=B\operatorname{diag}(\ell)^{-1}.
#    $$
#
# The orientation of each edge changes the sign of that coordinate but not
# the GraphPCA geometry. `AffinityConnectivity` separately checks whether
# very small affinities make a topologically connected graph numerically weak
# or disconnected for diffusion.

# %%
@dataclass
class GraphSetup:
    """A weighted graph reduced to the arrays GraphPCA needs.

    `node_list` and `edge_list` fix the row and column orderings that every
    later array follows. Everything downstream indexes by position, so these
    two lists are the single source of truth for what row 7 or column 12 is.
    """

    node_list: list
    edge_list: list
    n: int
    m: int
    ell: np.ndarray          # (m,) edge LENGTHS, from the 'weight' attribute
    sigma: float             # the affinity bandwidth actually used
    affinity: np.ndarray     # (m,) w_e = exp(-ell_e^2 / 2 sigma^2)
    W: np.ndarray            # (n, n) affinity-weighted adjacency
    L: np.ndarray            # (n, n) combinatorial Laplacian, diag(d) - W
    B_inc: np.ndarray        # (n, m) oriented incidence, -1 at u, +1 at v
    K: np.ndarray            # (n, m) length-scaled incidence, B_inc / ell


def edge_length_affinities(ell, sigma=None, eps=1e-20):
    """Edge lengths -> Gaussian affinities  w_e = exp(-ell_e^2 / 2 sigma^2).

    `sigma` defaults to the median edge length. Affinities are clipped to
    (eps, 1] so a very long edge does not underflow to exactly zero and
    disconnect the graph by accident. Returns (affinity, sigma).
    """
    ell = np.asarray(ell, float).ravel()
    if ell.size == 0 or not np.all(np.isfinite(ell)) or np.any(ell <= 0):
        raise ValueError("ell must be positive finite edge lengths.")
    eps = float(eps)
    if not np.isfinite(eps) or not 0 < eps <= 1:
        raise ValueError("eps must be finite and satisfy 0 < eps <= 1.")
    if sigma is None:
        sigma = float(np.median(ell))
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive.")
    return np.clip(np.exp(-ell ** 2 / (2.0 * sigma ** 2)), eps, 1.0), sigma


#: Below this, lambda2 / lambda_max means the graph is held together by a
#: thread. Chosen well above the 1e-12 zero-mode tolerance so the two checks
#: overlap instead of leaving a gap. Note the healthy-side margin depends on
#: SHAPE, not just health: compact graphs sit at 1e-1..1 but extended ones
#: decay like 1/diameter^2 (a unit-weight path of N vertices has gap_ratio
#: ~ pi^2/(4 N^2): N=100 -> 2.5e-4, N=2000 -> 6e-7 -- below this threshold
#: while perfectly connected). Treat a "weak" verdict on a very elongated
#: graph as a scale warning about choosing t, not as proof of a defect, and
#: raise `weak_gap_rtol` down or use on_weak="ignore" if it is expected.
WEAK_GAP_RTOL = 1e-6


@dataclass
class AffinityConnectivity:
    """Is the graph connected in a way the arithmetic can actually SEE?

    `require_connected` in `prepare_graph` tests the TOPOLOGY: does an edge
    exist. What the heat kernel sees is the affinity Laplacian, whose weights
    exp(-ell^2 / 2 sigma^2) shrink very fast with length. A graph joined by
    one long edge is numerically two graphs, and no topological check can
    tell. Two 6-cliques joined by one bridge, sigma = 1:

        bridge length   w_bridge    at floor?   zero modes   lambda2
                  1     6.07e-01      no            1        1.57e-01
                  5     3.73e-06      no            1        1.24e-06   <--
                 12     1.00e-20      yes           2        3.64e+00

    At length 5 the halves are joined by a thread: lambda2 has fallen five
    orders of magnitude, so a heat time chosen as t = tau / lambda2 would be
    800000x too large. Counting zero modes does not catch it, because that
    compares against 1e-12 of the LARGEST eigenvalue while the affinity floor
    is 1e-20 -- the whole range between is unpoliced. `gap_ratio` is the
    continuous measure that does catch it.
    """

    verdict: str                       # "connected" | "weak" | "disconnected"
    n_zero_modes: int
    lambda2: float
    lambda_max: float
    gap_ratio: float                   # lambda2 / lambda_max
    n_floor_edges: int
    floor_edges: list                  # the (u, v) pairs pinned at the floor
    components_without_floor_edges: int
    sigma: float = None
    detail: str = ""

    @property
    def ok(self):
        return self.verdict == "connected"

    def __str__(self):
        return (f"affinity connectivity: {self.verdict.upper()}   "
                f"lambda2/lambda_max = {self.gap_ratio:.3e}   "
                f"zero modes = {self.n_zero_modes}   "
                f"floor edges = {self.n_floor_edges}")


def affinity_connectivity_report(setup, evals_L, zero_rtol=1e-12,
                                 weak_gap_rtol=WEAK_GAP_RTOL, eps=1e-20,
                                 on_weak="warn"):
    """Judge the affinity graph, three ways, and say which edges are to blame.

      1. which edges sit at the affinity floor, and whether the graph falls
         apart without them  -- the combinatorial view, which names culprits;
      2. how many Laplacian eigenvalues are numerically zero;
      3. the normalised spectral gap lambda2 / lambda_max -- the continuous
         measure, and the one that actually degrades.

    `on_weak` is "warn" (default), "raise" or "ignore". Returns an
    AffinityConnectivity; changes nothing about the fit.
    """
    if on_weak not in {"warn", "raise", "ignore"}:
        raise ValueError("on_weak must be 'warn', 'raise' or 'ignore'.")
    evals = np.asarray(evals_L, float).ravel()
    if evals.size == 0 or not np.all(np.isfinite(evals)):
        raise ValueError("evals_L must be a nonempty vector of finite values.")

    lambda_max = float(np.max(evals))
    cutoff = zero_rtol * lambda_max
    n_zero = int(np.sum(evals <= cutoff))
    positive = evals[evals > cutoff]
    lambda2 = float(positive.min()) if positive.size else 0.0
    gap_ratio = lambda2 / lambda_max if lambda_max > 0 else 0.0

    # which edges the floor swallowed, and what breaks without them
    pinned = np.asarray(setup.affinity, float) <= eps * (1.0 + 1e-9)
    floor_edges = [tuple(setup.edge_list[i]) for i in np.flatnonzero(pinned)]
    # Computed unconditionally: with no floor edges this is simply the
    # component count of the graph itself, which is NOT always 1 when
    # require_connected=False was used. The old hard default of 1 reported a
    # disconnected two-piece graph as one piece in this field.
    import networkx as nx
    survivor = nx.Graph()
    survivor.add_nodes_from(setup.node_list)
    dropped = {frozenset(e) for e in floor_edges}
    survivor.add_edges_from(e for e in setup.edge_list
                            if frozenset(e) not in dropped)
    components_without = nx.number_connected_components(survivor)

    if n_zero > 1 or components_without > 1:
        verdict = "disconnected"
        detail = (
            f"The affinity graph is numerically disconnected: {n_zero} "
            f"Laplacian eigenvalues sit at or below {zero_rtol:g} of the "
            f"largest, so lambda2={lambda2:.6g} is the gap WITHIN one "
            f"component, not of the whole graph"
            + (f". Removing the {len(floor_edges)} edge(s) pinned at the "
               f"affinity floor leaves {components_without} components"
               if components_without > 1 else "") + ".")
    elif gap_ratio < weak_gap_rtol:
        verdict = "weak"
        detail = (
            f"The affinity graph is connected only by a thread: "
            f"lambda2/lambda_max = {gap_ratio:.3e}, below {weak_gap_rtol:g}. "
            f"The zero-mode count does not fire here -- it compares against "
            f"{zero_rtol:g} of the largest eigenvalue -- but the second mode "
            f"is so slow that any usable heat time already sees two pieces.")
    else:
        verdict, detail = "connected", ""

    report = AffinityConnectivity(
        verdict=verdict, n_zero_modes=n_zero, lambda2=lambda2,
        lambda_max=lambda_max, gap_ratio=gap_ratio,
        n_floor_edges=len(floor_edges), floor_edges=floor_edges,
        components_without_floor_edges=components_without,
        sigma=float(setup.sigma), detail=detail)

    if verdict != "connected" and on_weak != "ignore":
        message = detail + (
            f" Long edges make the affinities exp(-ell^2 / 2 sigma^2) "
            f"collapse, so this happens even when the graph is topologically "
            f"connected and require_connected=True passed -- that check tests "
            f"the topology, not the arithmetic. Either raise sigma (currently "
            f"{setup.sigma:.6g}) so long edges keep some weight, choose your "
            f"t_values without reference to lambda2, or analyse the "
            f"components separately.")
        if on_weak == "raise":
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return report


def prepare_graph(graph, node_list=None, edge_list=None, weight="weight",
                  sigma=None, eps=1e-20, require_connected=True):
    """A networkx graph with metric edge LENGTHS in `weight` -> a GraphSetup.

    The four steps, in order:
      1. fix the node and edge orderings, and read the edge lengths ell
      2. ell -> affinities w_e = exp(-ell^2 / 2 sigma^2)
      3. affinities -> adjacency W -> Laplacian L = diag(d) - W
      4. orderings -> oriented incidence B_inc, and K = B_inc / ell

    Nothing here is GraphPCA-specific; it is the graph plumbing that the
    heat kernel and the edge flows both need.
    """
    import networkx as nx

    if not isinstance(graph, nx.Graph) or isinstance(
            graph, (nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError("graph must be a simple undirected networkx.Graph.")

    node_list = list(graph.nodes()) if node_list is None else list(node_list)
    if edge_list is None:
        edge_list = [tuple(e) for e in graph.edges()]
    else:
        normalized = []
        for edge in edge_list:
            pair = tuple(edge)
            if len(pair) != 2:
                raise ValueError("Each edge_list entry must be a (u, v) pair; "
                                 f"got {edge!r}.")
            normalized.append(pair)
        edge_list = normalized
    n, m = len(node_list), len(edge_list)
    if n < 2 or m < 1:
        raise ValueError("Need at least 2 nodes and 1 edge.")
    if (len(node_list) != graph.number_of_nodes()
            or set(node_list) != set(graph.nodes())):
        raise ValueError("node_list must contain every graph vertex exactly once.")
    if any(u == v for u, v in edge_list):
        raise ValueError("Self-loops are not supported.")
    edge_keys = [frozenset(e) for e in edge_list]
    graph_keys = {frozenset(e) for e in graph.edges()}
    if len(set(edge_keys)) != len(edge_keys) or set(edge_keys) != graph_keys:
        raise ValueError("edge_list must contain every graph edge exactly once.")
    if require_connected:
        if not nx.is_connected(graph):
            raise ValueError("Graph must be connected. Pass "
                             "require_connected=False to override.")

    # 1. edge lengths, in edge_list order
    ell = np.empty(m, float)
    for i, (u, v) in enumerate(edge_list):
        data = graph[u][v]
        if weight not in data:
            raise ValueError(f"Edge ({u!r}, {v!r}) has no {weight!r} attribute; "
                             f"it should hold the edge LENGTH.")
        ell[i] = float(data[weight])
    if not np.all(np.isfinite(ell)) or np.any(ell <= 0):
        raise ValueError("All edge lengths must be finite and positive.")

    # 2. lengths -> affinities
    affinity, sigma = edge_length_affinities(ell, sigma=sigma, eps=eps)

    # 3. affinities -> adjacency -> Laplacian
    position = {u: i for i, u in enumerate(node_list)}
    W = np.zeros((n, n), float)
    for k, (u, v) in enumerate(edge_list):
        i, j = position[u], position[v]
        W[i, j] = W[j, i] = affinity[k]
    L = np.diag(W.sum(axis=1)) - W

    # 4. oriented incidence: -1 at the first endpoint, +1 at the second.
    #    (This is networkx's oriented=True convention. The sign is a choice
    #    of edge direction and cancels out of everything GraphPCA computes.)
    B_inc = np.zeros((n, m), float)
    for k, (u, v) in enumerate(edge_list):
        B_inc[position[u], k] = -1.0
        B_inc[position[v], k] = +1.0
    K = B_inc * (1.0 / ell)[None, :]

    return GraphSetup(node_list, edge_list, n, m, ell, sigma, affinity,
                      W, L, B_inc, K)

# %% [markdown]
# ## 5. Output and input validation
#
# `GraphPCAResult` stores the retained eigenvalues in descending order, the
# principal edge fields as columns of `basis`, and one score vector for every
# graph vertex. It also stores the centered-flow mean, total variance, vertex
# measure, edge lengths, and numerical diagnostics.
#
# `_prepare` accepts either:
#
# - a scalar-field matrix `D` together with the incidence operator `K`; or
# - a precomputed $m\times n$ edge-flow matrix through `flows`.
#
# The edge lengths must be strictly positive. The vertex weights must be
# finite and nonnegative and are normalized to sum to one.

# %%
@dataclass
class GraphPCAResult:
    """One GraphPCA fit. Eigenvalues are DESCENDING, so index 0 is PC1."""

    t: float
    eigenvalues: np.ndarray          # (k,)   lambda_j
    basis: np.ndarray                # (m, k) the PCs v_j, C-orthonormal
    scores: np.ndarray               # (n, k) alpha_j(y), for ALL n vertices
    mean_flow: np.ndarray            # (m,)   gbar
    total_variance: float
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance_ratio: np.ndarray
    all_eigenvalues: np.ndarray      # available spectrum before filtering;
                                     # randomized solver returns only k_keep
    rank_capacity: int
    n_atoms: int                     # number of vertices with mu > 0
    ell: np.ndarray
    mu: np.ndarray
    diagnostics: dict = field(default_factory=dict)

    @property
    def n_components(self):
        return self.scores.shape[1]

    @property
    def k_over_capacity(self):
        """Fraction of the available rank used. 1.0 means nothing was cut."""
        return (self.n_components / self.rank_capacity
                if self.rank_capacity else float("nan"))

    @property
    def eigenvalues_ascending(self):
        """The `graphpca` package stores eigenvalues smallest-first."""
        return self.eigenvalues[::-1]

    @property
    def basis_ascending(self):
        """The package's `evecs_dict[t]` column order."""
        return self.basis[:, ::-1]


def _prepare(D, K, flows, ell, mu):
    """Validate the inputs. Returns (flows or None, D, K, ell, mu, m, n).

    All the defensive checking lives here so `graphpca` below stays a
    readable transcription of the formulas.
    """
    if flows is not None:
        G = _dense(flows)
        if G.ndim != 2 or not np.all(np.isfinite(G)):
            raise ValueError("flows must be a finite array of shape (m, n).")
        m, n = G.shape
    else:
        if D is None or K is None:
            raise ValueError("Give either (D, K) or flows=(m, n).")
        G, D, K = None, _dense(D), _dense(K)
        if K.ndim != 2 or not np.all(np.isfinite(K)):
            raise ValueError("K must be a finite array of shape (n, m).")
        n, m = K.shape
        if D.shape != (n, n):
            raise ValueError(f"D must be ({n}, {n}) to match K; got {D.shape}.")
        if not np.all(np.isfinite(D)):
            raise ValueError("D must contain only finite values.")

    if n < 2 or m < 1:
        raise ValueError("The inputs must have n >= 2 vertices and m >= 1 edge.")

    ell = np.ones(m) if ell is None else np.asarray(ell, float).ravel()
    if ell.shape[0] != m or not np.all(np.isfinite(ell)) or not np.all(ell > 0):
        raise ValueError(f"ell must be {m} positive finite edge lengths.")

    mu = np.full(n, 1.0 / n) if mu is None else np.asarray(mu, float).ravel()
    if mu.shape[0] != n:
        raise ValueError(f"mu must have length n={n}; got {mu.shape[0]}.")
    if not np.all(np.isfinite(mu)) or np.any(mu < 0) or mu.sum() <= 0:
        raise ValueError("mu must be finite, nonnegative, with positive mass.")

    return G, D, K, ell, mu / mu.sum(), m, n


# %% [markdown]
# ## 6. GraphPCA at one diffusion time
#
# Let $F$ be the matrix whose $y$th column is the centered edge flow
# $\widetilde g_y$. With
#
# $$
# C=\operatorname{diag}(\ell),
# \qquad M_\mu=\operatorname{diag}(\mu),
# $$
#
# the weighted design matrix is
#
# $$
# U=C^{1/2}FM_\mu^{1/2}.
# $$
#
# The exact solver computes
#
# $$
# U=P\Sigma Q^T.
# $$
#
# Since
#
# $$
# UU^T=P\Sigma^2P^T,
# $$
#
# the left singular vector $P_j$ is an eigenvector of the whitened
# edge-space covariance, with eigenvalue $\lambda_j=\sigma_j^2$. The
# principal edge field is obtained by undoing the metric whitening:
#
# $$
# v_j=C^{-1/2}P_j.
# $$
#
# Its score at vertex $y$ is
#
# $$
# \alpha_j(y)=\widetilde g_y^TCv_j.
# $$
#
# `solver="randomized"` approximates the leading singular triplets for large
# problems. `solver="direct"` explicitly forms the $m\times m$ whitened
# covariance and diagonalizes it; this is useful as an independent check but
# squares the singular values and is less accurate in the small spectrum.

# %%
def graphpca(D=None, K=None, ell=None, mu=None, t=None, k_keep=40,
             solver="exact", eig_rtol=1e-10, eig_atol=0.0, random_state=0,
             n_iter=3, cancellation_rtol=CANCELLATION_RTOL, warn=True,
             flows=None, pin_signs=True):
    """GraphPCA at one diffusion time. Section 6 above gives the mathematics.

    D, K       the scalar-field matrix (n, n) and the incidence operator (n, m)
    flows      (m, n) precomputed uncentred flows, INSTEAD of (D, K). Use this
               when the graph is fixed and only mu changes, so K.T @ S_t is
               built once and reused.
               CAVEAT: the large-t cancellation guard cannot fire on this
               path. It compares the centred FIELD against the raw field, and
               here only the flows are visible -- once the field has
               equilibrated the flows ARE the round-off, so centring them
               cancels nothing relative to themselves and the ratio never
               drops below the threshold. Keep t in a sensible range yourself
               (tau / lambda_2 with tau of order 1), or probe each t once
               through the (D, K) path first.
    ell, mu    edge lengths (default: all ones) and vertex measure (default:
               uniform). Both are validated and mu is normalised for you.
    k_keep     how many components to return, capped at the rank ceiling.
    solver     "exact"      -- one SVD of U. The safe default: no squaring.
               "randomized" -- sklearn's randomized SVD, for large problems.
               "direct"     -- build the (m, m) covariance and eigendecompose
                               it. This is the DEFINITION written out
                               literally, kept as an independent check. It is
                               slower, uses O(m^2) memory, and loses accuracy
                               in the small eigenvalues, so use it to verify
                               "exact", not to produce results.
    pin_signs  make each PC's largest entry positive, so a PC does not flip
               sign between diffusion times. Set False to reproduce a raw SVD
               bit for bit -- for instance to keep an existing cache of fitted
               bases valid. Subspace quantities are identical either way.
    """
    if not isinstance(k_keep, (int, np.integer)) or int(k_keep) < 1:
        raise ValueError("k_keep must be a positive integer.")
    if solver not in {"exact", "randomized", "direct"}:
        raise ValueError("solver must be 'exact', 'randomized' or 'direct'.")
    eig_rtol, eig_atol = float(eig_rtol), float(eig_atol)
    if (not np.isfinite(eig_rtol) or eig_rtol < 0
            or not np.isfinite(eig_atol) or eig_atol < 0):
        raise ValueError("eig_rtol and eig_atol must be finite and nonnegative.")
    cancellation_rtol = float(cancellation_rtol)
    if not np.isfinite(cancellation_rtol) or cancellation_rtol < 0:
        raise ValueError("cancellation_rtol must be finite and nonnegative.")
    if (not isinstance(n_iter, (int, np.integer)) or int(n_iter) < 0):
        raise ValueError("n_iter must be a nonnegative integer.")

    G, D, K, ell, mu, m, n = _prepare(D, K, flows, ell, mu)

    n_atoms = int(np.count_nonzero(mu))
    capacity = rank_capacity(n_atoms, m)
    if warn and n_atoms < 2:
        warnings.warn(f"mu lives on {n_atoms} vertex/vertices, so the centred "
                      f"covariance is identically zero and there are no "
                      f"principal components.", RuntimeWarning, stacklevel=2)

    # --- 1. the flows, and their mu-weighted mean -------------------------
    if G is None:
        mean_field = D @ mu                        # centre the field first,
        centred_field = D - mean_field[:, None]    # then take gradients --
        F = K.T @ centred_field                    # this is what the package
        mean_flow = K.T @ mean_field               # does, to the last bit
        raw_size, centred_size = np.abs(D).max(), np.abs(centred_field).max()
    else:
        mean_flow = G @ mu
        F = G - mean_flow[:, None]
        raw_size, centred_size = np.abs(G).max(), np.abs(F).max()

    if warn and raw_size == 0:
        # An all-zero diffusion potential is usually large-t underflow, but a
        # custom field can be zero deliberately and precomputed flows do not
        # expose their source. Warn about the observable fact without claiming
        # more than this route can prove.
        what = "potential" if G is None else "precomputed flow matrix"
        warnings.warn(
            f"t={t!r}: the {what} is identically zero. If it came from a "
            f"diffusion potential, it may have underflowed. Everything "
            f"returned for this time is empty; check the input or use a "
            f"smaller t.", RuntimeWarning, stacklevel=2)
    if (warn and cancellation_rtol > 0 and raw_size > 0
            and centred_size <= cancellation_rtol * raw_size):
        warnings.warn(f"t={t!r}: centring left only round-off (ratio "
                      f"{centred_size / raw_size:.2e}). Everything returned "
                      f"for this time is noise, not signal. Use a smaller t.",
                      RuntimeWarning, stacklevel=2)

    # --- 2. U = sqrt(ell) * (centred flows) * sqrt(mu) --------------------
    # Only the mu > 0 columns matter -- the rest are multiplied by zero --
    # and dropping them is a large saving when mu is sparse, as it is for a
    # digit image where most pixels are empty.
    active = mu > 0
    sqrt_ell = np.sqrt(ell)[:, None]
    U = sqrt_ell * (F[:, active] * np.sqrt(mu[active])[None, :])
    total_variance = float(np.sum(U ** 2))         # = trace of the covariance

    k_want = min(int(k_keep), capacity)
    if k_want < 1:
        return _empty_result(t, m, n, mean_flow, total_variance, capacity,
                             n_atoms, ell, mu, solver, k_keep, pin_signs)

    # --- 3. one SVD of U (never U @ U.T; see the module docstring) --------
    if solver == "exact":
        try:
            U_left, singular, _ = np.linalg.svd(U, full_matrices=False)
        except np.linalg.LinAlgError:              # rare non-convergence
            from scipy.linalg import svd
            U_left, singular, _ = svd(U, full_matrices=False,
                                      lapack_driver="gesvd")
        full_spectrum = singular ** 2
        U_left, singular = U_left[:, :k_want], singular[:k_want]

    elif solver == "randomized":
        from sklearn.utils.extmath import randomized_svd
        U_left, singular, _ = randomized_svd(U, n_components=k_want,
                                             n_iter=n_iter,
                                             random_state=random_state)
        full_spectrum = singular ** 2

    elif solver == "direct":
        # The DEFINITION, written out literally, as an independent check.
        # It builds the (m, m) edge-space covariance and eigendecomposes it:
        #
        #     M_cov = sum_y mu_y (g_y - gbar)(g_y - gbar)^T
        #           = K.T (D diag(mu) D.T) K  -  gbar gbar.T
        #     A     = C^{1/2} M_cov C^{1/2},    A u = lambda u,   v = C^{-1/2} u
        #
        # Slower, and O(m^2) memory, and less accurate in the small
        # eigenvalues because it forms the square this file otherwise avoids.
        # Use it to confirm the SVD path, not to produce results.
        if G is None:
            second_moment = K.T @ ((D * mu[None, :]) @ D.T) @ K
        else:
            second_moment = (G * mu[None, :]) @ G.T
        M_cov = second_moment - np.outer(mean_flow, mean_flow)
        M_cov = 0.5 * (M_cov + M_cov.T)            # kill asymmetric round-off
        root_ell = np.sqrt(ell)
        A = (root_ell[:, None] * M_cov) * root_ell[None, :]
        eigenvalues_all, u_vectors = np.linalg.eigh(A)
        order = np.argsort(eigenvalues_all)[::-1]  # -> descending, like the SVD
        full_spectrum = np.clip(eigenvalues_all[order], 0.0, None)
        U_left = u_vectors[:, order][:, :k_want]
        singular = np.sqrt(full_spectrum[:k_want])

    # --- 4. drop the components that are numerically zero -----------------
    eigenvalues = singular ** 2
    largest = float(eigenvalues.max()) if eigenvalues.size else 0.0
    keep = eigenvalues > max(eig_rtol * largest, eig_atol)
    eigenvalues, U_left = eigenvalues[keep], U_left[:, keep]

    if pin_signs and U_left.shape[1]:
        sign = np.sign(U_left[np.argmax(np.abs(U_left), axis=0),
                              np.arange(U_left.shape[1])])
        sign[sign == 0] = 1.0
        U_left = U_left * sign[None, :]

    # --- 5. back into the edge metric, then the vertex scores -------------
    basis = U_left / sqrt_ell                      # v_j = U_left_j / sqrt(ell)
    scores = F.T @ (ell[:, None] * basis)          # ALL n vertices, not just mu>0
    ratio = (eigenvalues / total_variance if total_variance > 0
             else np.full(eigenvalues.shape, np.nan))

    return GraphPCAResult(
        t=t, eigenvalues=eigenvalues, basis=basis, scores=scores,
        mean_flow=mean_flow, total_variance=total_variance,
        explained_variance_ratio=ratio,
        cumulative_explained_variance_ratio=np.cumsum(ratio),
        all_eigenvalues=full_spectrum, rank_capacity=capacity,
        n_atoms=n_atoms, ell=ell, mu=mu,
        diagnostics={"solver": solver, "n": n, "m": m,
                     "k_requested": int(k_keep), "pin_signs": bool(pin_signs),
                     "spectrum_complete": solver != "randomized"})


def _empty_result(t, m, n, mean_flow, total_variance, capacity, n_atoms,
                  ell, mu, solver, k_keep, pin_signs):
    """Return the correctly shaped empty result when no component can exist.

    Carries the SAME diagnostics keys as a populated result, so
    ``result.diagnostics["k_requested"]`` never raises on the empty path.
    """
    return GraphPCAResult(
        t=t, eigenvalues=np.zeros(0), basis=np.zeros((m, 0)),
        scores=np.zeros((n, 0)), mean_flow=mean_flow,
        total_variance=total_variance,
        explained_variance_ratio=np.zeros(0),
        cumulative_explained_variance_ratio=np.zeros(0),
        all_eigenvalues=np.zeros(0), rank_capacity=capacity, n_atoms=n_atoms,
        ell=ell, mu=mu,
        diagnostics={"solver": solver, "n": n, "m": m,
                     "k_requested": int(k_keep), "pin_signs": bool(pin_signs),
                     "spectrum_complete": solver != "randomized"})


# %% [markdown]
# ## 7. Mathematical checks and sign alignment
#
# A valid GraphPCA result satisfies three identities:
#
# $$
# V^TCV=I,
# $$
#
# $$
# \sum_y\mu_y\alpha_j(y)=0,
# $$
#
# and
#
# $$
# \sum_y\mu_y\alpha_j(y)\alpha_k(y)
# =\lambda_j\delta_{jk}.
# $$
#
# `check_graphpca` verifies these identities numerically. They establish
# internal consistency among the basis, scores, metric, and eigenvalues; they
# do not prove that an approximate solver found the exact leading subspace.
#
# A principal component is defined only up to sign. `align_pc_signs` compares
# score columns with a reference fit and flips a column when their weighted
# overlap is negative. This changes only the orientation of the axis, not the
# represented principal direction.

# %%
def check_graphpca(result, rtol=1e-9, atol=1e-12, verbose=False):
    """Three identities a correct fit must satisfy. Raises if one fails.

    1. the PCs are C-orthonormal          V.T diag(ell) V = I
    2. the scores are mu-centred          sum_y mu_y alpha_j(y) = 0
    3. the score covariance is diagonal   sum_y mu_y alpha_j alpha_k
                                              = lambda_j if j == k else 0

    The third is the substantive one: it ties the vertex scores back to the
    eigenvalues, so it fails if the basis, the metric or the centring is
    scrambled. Tolerances are relative TO THE LEADING EIGENVALUE (and to
    max|scores| for identity 2), because lambda spans many decades as t
    sweeps and a fixed absolute tolerance would be meaningless at one end.

    Know what that buys and what it does not. Because identity 3 is measured
    at the scale of lambda_1, an error confined to a trailing component whose
    own lambda_j is far below lambda_1 * rtol passes unseen -- zeroing the
    last scores column of a wide-spectrum fit still passes. And all three
    identities test INTERNAL CONSISTENCY, not optimality: a self-consistent
    subset of the wrong components passes, as does a randomized-SVD fit whose
    eigenvalues are off by percents. This is a scramble detector, not a
    certificate that these are the true leading components.
    """
    V, S, lam = result.basis, result.scores, result.eigenvalues
    mu, ell = result.mu, result.ell
    k = S.shape[1]

    measured = {
        "c_orthonormality_error": (
            float(np.max(np.abs(V.T @ (ell[:, None] * V) - np.eye(k))))
            if k else 0.0, 1.0),
        "score_mean_error": (
            float(np.max(np.abs(mu @ S))) if k else 0.0,
            float(np.max(np.abs(S))) if k and S.size else 1.0),
        "score_covariance_error": (
            float(np.max(np.abs((S * mu[:, None]).T @ S - np.diag(lam))))
            if k else 0.0, float(lam[0]) if k else 1.0),
    }

    report = {}
    for name, (value, scale) in measured.items():
        report[name] = value
        report[name + "_relative"] = value / scale
        if value >= atol + rtol * scale:
            raise AssertionError(f"{name} = {value:.3e} exceeds "
                                 f"{atol + rtol * scale:.3e}")
    if verbose:
        for name, (value, scale) in measured.items():
            print(f"  {name:<26} {value:.3e}   relative {value / scale:.3e}")
        print(f"  all three identities hold at t={result.t}")
    return report


def align_pc_signs(scores, reference_scores, mu=None, n_components=None):
    """Flip PC columns to agree in sign with a reference fit.

    PC signs are arbitrary, so when sweeping t, aligning against one reference
    time stops the colour maps reversing for no reason.
    """
    S = np.asarray(scores, float).copy()
    R = np.asarray(reference_scores, float)
    k = min(S.shape[1], R.shape[1])
    if n_components is not None:
        k = min(k, int(n_components))
    weight = np.ones(len(S)) if mu is None else np.asarray(mu, float)
    for j in range(k):
        if float(np.sum(weight * R[:, j] * S[:, j])) < 0:
            S[:, j] *= -1
    return S


# %% [markdown]
# ## 8. Complete GraphPCA sweep
#
# `fit_graphpca` combines the preceding steps:
#
# 1. Convert the input graph into $\ell$, $W$, $L$, $B$, and $K$.
# 2. Diagonalize the graph Laplacian once.
# 3. Diagnose the affinity connectivity.
# 4. At each requested heat time, construct either the heat kernel or the
#    diffusion-distance potential.
# 5. Run the one-time GraphPCA calculation and store its output.
#
# `GraphPCAFit.by_t[t]` contains a `GraphPCAResult` with eigenvalues in
# descending order. For compatibility with the external package,
# `evals_dict[t]` and `evecs_dict[t]` use ascending order, whereas
# `coeffs_dict[t]` and the variance-ratio dictionaries remain PC1-first.
#
# Deterministic sign pinning reduces arbitrary sign reversals across runs.
# When exact agreement with an earlier figure is needed, `sign_reference`
# aligns each score column with the supplied reference scores.

# %%
@dataclass
class GraphPCAFit:
    """Everything one sweep produces. Field names match the package's result.

    NOTE the two conventions, inherited so downstream code is unchanged:
      * `evals_dict[t]` and `evecs_dict[t]` are ASCENDING (smallest first)
      * `coeffs_dict[t]` and `pc_var_ratio_dict[t]` are PC1 FIRST
    `by_t[t]` is the GraphPCAResult, which is descending throughout.
    """

    setup: GraphSetup
    t_values: list
    potential_function: str
    mu: np.ndarray
    by_t: dict                       # t -> GraphPCAResult (descending)
    potential_dict: dict             # t -> the (n, n) scalar field used
    evals_dict: dict                 # t -> ASCENDING eigenvalues
    evecs_dict: dict                 # t -> (m, k) PCs, matching order
    coeffs_dict: dict                # t -> (n, k_pcs) scores, PC1 first
    pc_var_ratio_dict: dict          # t -> PC1 first
    pc_cum_ratio_dict: dict
    total_var_dict: dict
    gbar_dict: dict
    evals_L: np.ndarray              # Laplacian spectrum
    evecs_L: np.ndarray
    connectivity: object = None      # AffinityConnectivity; see below

    # convenience passthroughs, so `fit.K` works like the package's
    @property
    def K(self):
        return self.setup.K

    @property
    def ell(self):
        return self.setup.ell

    @property
    def L(self):
        return self.setup.L

    @property
    def node_list(self):
        return self.setup.node_list

    @property
    def edge_list(self):
        return self.setup.edge_list

    @property
    def sigma(self):
        return self.setup.sigma


def fit_graphpca(graph, potential_function="heat_kernel", t_values=(1.0,),
                 mu=None, k_keep=40, k_pcs=None, sigma=None, eig_rtol=1e-10,
                 solver="exact", require_connected=True, verbose=False,
                 pin_signs=True, sign_reference=None,
                 on_weak_connectivity="warn", weak_gap_rtol=WEAK_GAP_RTOL,
                 eig_atol=0.0, random_state=0, n_iter=3,
                 cancellation_rtol=CANCELLATION_RTOL,
                 **prepare_kwargs):
    """Graph in, GraphPCA out. Replaces the `graphpca` package's fit_graphpca.

    potential_function : "heat_kernel"         -> H_t = exp(-t L)
                         "diffusion_potential" -> d_t(i,j)^2 / (2n)
                         ("diffusion" remains accepted as a legacy alias.)

    ON PC SIGNS -- read this if a figure came out mirrored.

    An eigenvector is only defined up to sign: v and -v describe the same
    principal direction, and every solver picks one arbitrarily. The package's
    `fit_graphpca` leaves whatever its solver returned (it ships
    `align_pc_signs` but does not call it), so its signs vary with the solver
    and with t. This function pins each PC's largest-magnitude entry positive
    by default, which is reproducible but need not agree with the package's
    arbitrary choice -- so a scatter plot can come out MIRRORED even though
    the eigenvalues, the subspace and every distance between points are
    identical.

    pin_signs      : False leaves the raw SVD signs. Note this still need not
                     match the package, whose default solver is a different
                     code path with its own arbitrary signs.
    sign_reference : dict {t: (n, k) coefficients} to match. Each PC is
                     flipped to agree with the reference in the mu-weighted
                     inner product. THIS is how to reproduce an earlier
                     figure's orientation exactly.
    """
    if potential_function == "diffusion":               # legacy spelling
        potential_function = "diffusion_potential"
    if potential_function not in {"heat_kernel", "diffusion_potential"}:
        raise ValueError("potential_function must be 'heat_kernel' or "
                         "'diffusion_potential'.")
    if k_pcs is not None and (
            not isinstance(k_pcs, (int, np.integer)) or int(k_pcs) < 0):
        raise ValueError("k_pcs must be a nonnegative integer or None.")

    t_values = [float(t) for t in t_values]
    if not t_values:
        raise ValueError("t_values must contain at least one diffusion time.")
    if any(not np.isfinite(t) or t <= 0 for t in t_values):
        raise ValueError("Every diffusion time must be finite and positive.")
    if len(set(t_values)) != len(t_values):
        raise ValueError("t_values must not contain duplicates.")

    setup = prepare_graph(graph, sigma=sigma,
                          require_connected=require_connected, **prepare_kwargs)
    lap_values, lap_vectors = laplacian_eigh(setup.L)

    # `require_connected` above tested the topology. This tests what the heat
    # kernel actually sees. It computes nothing the fit uses -- every array
    # below is bit-identical with or without it -- it only looks and reports.
    connectivity = affinity_connectivity_report(
        setup, lap_values, weak_gap_rtol=weak_gap_rtol,
        eps=prepare_kwargs.get("eps", 1e-20), on_weak=on_weak_connectivity)

    build = (graph_heat_kernel if potential_function == "heat_kernel"
             else diffusion_distance_potential)
    mu = (np.full(setup.n, 1.0 / setup.n) if mu is None
          else np.asarray(mu, float).ravel())
    if mu.shape != (setup.n,):
        raise ValueError(f"mu must have length n={setup.n}; got {mu.shape}.")
    if not np.all(np.isfinite(mu)) or np.any(mu < 0) or mu.sum() <= 0:
        raise ValueError("mu must be finite, nonnegative, with positive mass.")
    mu = mu / mu.sum()

    fit = GraphPCAFit(
        setup=setup, t_values=t_values, potential_function=potential_function,
        mu=mu, by_t={}, potential_dict={}, evals_dict={},
        evecs_dict={}, coeffs_dict={}, pc_var_ratio_dict={},
        pc_cum_ratio_dict={}, total_var_dict={}, gbar_dict={},
        evals_L=lap_values, evecs_L=lap_vectors, connectivity=connectivity)

    for t in t_values:
        D = build(lap_values, lap_vectors, t)
        result = graphpca(D, setup.K, ell=setup.ell, mu=mu, t=t,
                          k_keep=k_keep, eig_rtol=eig_rtol,
                          eig_atol=eig_atol, solver=solver,
                          random_state=random_state, n_iter=n_iter,
                          cancellation_rtol=cancellation_rtol,
                          pin_signs=pin_signs)
        if sign_reference is not None and t in sign_reference:
            reference = np.asarray(sign_reference[t], float)
            if (reference.ndim != 2 or reference.shape[0] != setup.n
                    or not np.all(np.isfinite(reference))):
                raise ValueError(f"sign_reference[{t!r}] must be a finite "
                                 f"array with {setup.n} rows.")
            width = min(reference.shape[1], result.scores.shape[1])
            flip = np.ones(result.scores.shape[1])
            overlap = np.einsum("i,ij,ij->j", fit.mu,
                                reference[:, :width],
                                result.scores[:, :width])
            flip[:width] = np.where(overlap < 0, -1.0, 1.0)
            result.scores = result.scores * flip[None, :]
            result.basis = result.basis * flip[None, :]
        k = result.n_components if k_pcs is None else min(int(k_pcs),
                                                          result.n_components)
        fit.by_t[t] = result
        fit.potential_dict[t] = D
        fit.evals_dict[t] = result.eigenvalues_ascending       # package order
        fit.evecs_dict[t] = result.basis_ascending
        fit.coeffs_dict[t] = result.scores[:, :k]              # PC1 first
        fit.pc_var_ratio_dict[t] = result.explained_variance_ratio
        fit.pc_cum_ratio_dict[t] = result.cumulative_explained_variance_ratio
        fit.total_var_dict[t] = result.total_variance
        fit.gbar_dict[t] = result.mean_flow
        if verbose:
            print(f"t={t:g}: {result.n_components} components, "
                  f"total variance {result.total_variance:.6e}")
    return fit

# %% [markdown]
# ## 9. Nearly degenerate eigenspaces
#
# When several neighboring eigenvalues are nearly equal, their individual
# eigenvectors are not intrinsically identified: an orthogonal rotation within
# the common eigenspace gives an equally valid basis. `degenerate_blocks`
# groups consecutive eigenvalues whose complete block remains within a chosen
# ratio.
#
# For a block $B$, the score magnitude
#
# $$
# m_B(y)=\sqrt{\sum_{j\in B}\alpha_j(y)^2}
# $$
#
# is invariant under such a rotation. For a two-dimensional block,
# `block_angle` retains the complementary angular coordinate, although that
# angle is defined only up to a global rotation and reflection.
#
# `check_block_invariance` demonstrates the invariance by applying random
# orthogonal changes of basis to the selected score columns.

# %%
def degenerate_blocks(eigenvalues, ratio_threshold=1.2, max_size=None):
    """Group eigenvalues that are too close together to separate.

    lambda_{j+1} joins the current block only if the WHOLE block still spans
    less than `ratio_threshold`, i.e. lambda_first / lambda_last < threshold.

    The stricter rule matters. Chaining on neighbours alone would merge a
    slowly decaying tail into one meaningless block: [0.790, 0.713, 0.598,
    0.502] has neighbour ratios 1.11, 1.19, 1.19 -- all under 1.2 -- and yet
    lambda_1 / lambda_4 = 1.58, so PC1 and PC4 are clearly separate.

    Expects DESCENDING eigenvalues. Returns index lists, e.g. [[0,1],[2,3],[4]].
    """
    lam = np.asarray(eigenvalues, float)
    if lam.size == 0:
        return []
    # Ascending input -- the package's own evals_dict[t] order -- produces
    # nonsense blocks silently, so it is rejected. The scale is RELATIVE to
    # the largest |eigenvalue| (with a tiny floor), never an absolute 1.0:
    # a large-t spectrum sits entirely below 1e-12, and an absolute floor
    # would wave the wrong order straight through for exactly those fits.
    # (GraphPCA 0.8.0's own guard has that flaw; this one does not.)
    if lam.size > 1:
        scale = max(float(np.abs(lam).max()), np.finfo(float).tiny)
        if np.any(np.diff(lam) > 1e-12 * scale):
            raise ValueError(
                "eigenvalues must be DESCENDING (PC1 first). If these came "
                "from the package's evals_dict[t], pass evals_dict[t][::-1].")
    tiny = np.finfo(float).tiny
    blocks, current = [], [0]
    for j in range(1, len(lam)):
        span = lam[current[0]] / max(lam[j], tiny)
        room = max_size is None or len(current) < max_size
        if span < ratio_threshold and room:
            current.append(j)
        else:
            blocks.append(current)
            current = [j]
    blocks.append(current)
    return blocks


def block_magnitude(scores, block, rms=False):
    """sqrt( sum_{j in B} alpha_j(y)^2 ) -- for a pair, sqrt(PC1^2 + PC2^2).

    The rotation-invariant summary of a block: unlike the individual scores it
    does not change when the solver picks a different basis inside the block.
    `rms=True` divides by sqrt(|B|), so blocks of different sizes compare.
    """
    idx = list(block)
    magnitude = np.sqrt(np.sum(np.asarray(scores, float)[:, idx] ** 2, axis=1))
    return magnitude / np.sqrt(len(idx)) if rms else magnitude


def block_angle(scores, block):
    """atan2 inside a 2-D block -- the information the magnitude discards.

    Defined only up to a global rotation and reflection of the block, so
    compare it with a rotation-invariant statistic, never a plain correlation.
    """
    idx = list(block)
    if len(idx) != 2:
        raise ValueError(f"block_angle needs a 2-D block, got {len(idx)}")
    S = np.asarray(scores, float)
    return np.arctan2(S[:, idx[1]], S[:, idx[0]])


def block_summary(result, ratio_threshold=1.2):
    """One dict per block: indices, size, eigenvalues, share, magnitude, angle."""
    lam = result.eigenvalues
    out = []
    for b in degenerate_blocks(lam, ratio_threshold):
        label = ("PC" + "-PC".join(str(i + 1) for i in (b[0], b[-1]))
                 if len(b) > 1 else f"PC{b[0] + 1}")
        out.append({
            "indices": b, "pcs": [i + 1 for i in b], "size": len(b),
            "eigenvalues": lam[b], "label": label,
            "variance_share": float(np.sum(result.explained_variance_ratio[b])),
            "magnitude": block_magnitude(result.scores, b),
            "angle": block_angle(result.scores, b) if len(b) == 2 else None,
        })
    return out


def check_block_invariance(result, block, n_trials=5, seed=0, atol=1e-10):
    """Show the block magnitude does not depend on the basis.

    Rotate the scores inside the block at random -- exactly the freedom a
    degenerate eigenvalue leaves the solver -- and confirm the magnitude stays
    put while the individual axes move.
    """
    rng = np.random.default_rng(seed)
    idx = list(block)
    baseline = block_magnitude(result.scores, idx)
    scale = max(float(np.max(np.abs(baseline))), 1e-300)
    worst_magnitude, worst_axis = 0.0, 0.0
    for _ in range(n_trials):
        Q, _ = np.linalg.qr(rng.normal(size=(len(idx), len(idx))))
        rotated = result.scores.copy()
        rotated[:, idx] = result.scores[:, idx] @ Q
        worst_magnitude = max(worst_magnitude, float(np.max(np.abs(
            block_magnitude(rotated, idx) - baseline))))
        worst_axis = max(worst_axis, float(np.max(np.abs(
            rotated[:, idx] - result.scores[:, idx]))))
    if worst_magnitude >= atol * scale:
        raise AssertionError(f"block magnitude moved by {worst_magnitude:.3e}")
    return {"magnitude_change": worst_magnitude, "axis_change": worst_axis,
            "relative_magnitude_change": worst_magnitude / scale}


# %% [markdown]
# ## 10. Optional plotting helpers
#
# Matplotlib is imported only inside these functions, so it is not required
# for the numerical calculation. The helpers display the retained spectrum,
# explained variance, cumulative variance, spectra across heat times, and the
# rotation-invariant magnitude of a nearly degenerate block.

# %%
def _finish(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_title(title, color=INK, fontsize=11)
    ax.grid(alpha=0.2, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)


def _t_label(result):
    return f", $t={result.t:g}$" if isinstance(result.t, (int, float)) else ""


def plot_eigenvalues(result, ax=None, top_k=None, log=True, mark_blocks=True,
                     ratio_threshold=1.2, title=None):
    """The spectrum, with degenerate blocks shaded so the pairing shows."""
    import matplotlib.pyplot as plt

    lam = np.asarray(result.eigenvalues, float)
    if log:
        lam = lam[lam > 0]
    if top_k:
        lam = lam[:top_k]
    index = np.arange(1, len(lam) + 1)
    ax = ax or plt.subplots(figsize=(6.4, 4.2))[1]

    if mark_blocks:
        for b in degenerate_blocks(result.eigenvalues, ratio_threshold):
            b = [i for i in b if i < len(lam)]
            if len(b) > 1:
                ax.axvspan(b[0] + 0.6, b[-1] + 1.4, color=MUTED, alpha=0.10,
                           lw=0, zorder=0)
    ax.plot(index, lam, "-", color=MUTED, lw=1.2, zorder=1)
    ax.scatter(index, lam, s=42, color=PC_COLORS[0], marker=PC_MARKERS[0],
               zorder=3, linewidths=0)
    if log:
        ax.set_yscale("log")
    if len(index) <= 25:
        ax.set_xticks(index)
    _finish(ax, "component $j$", r"$\lambda_j$",
            title or f"eigenvalues{_t_label(result)}"
            + ("   (shaded = degenerate block)" if mark_blocks else ""))
    return ax


def plot_variance_explained(result, ax=None, top_k=10, annotate=True,
                            title=None):
    import matplotlib.pyplot as plt

    ratio = np.asarray(result.explained_variance_ratio, float)[:top_k]
    index = np.arange(1, len(ratio) + 1)
    ax = ax or plt.subplots(figsize=(6.4, 4.2))[1]
    ax.bar(index, ratio, color=PC_COLORS[0], width=0.68, linewidth=0)
    if annotate:
        for j, value in zip(index, ratio):
            if np.isfinite(value) and value >= 0.01:
                ax.annotate(f"{value:.1%}", (j, value), fontsize=8,
                            textcoords="offset points", xytext=(0, 3),
                            ha="center", color=MUTED)
    ax.set_xticks(index)
    top = float(np.nanmax(ratio)) if ratio.size else 1.0
    ax.set_ylim(0, min(1.0, top * 1.2) if np.isfinite(top) and top > 0 else 1.0)
    _finish(ax, "component $j$", "explained variance ratio",
            title or f"variance explained{_t_label(result)}")
    return ax


def plot_cumulative_variance(result, ax=None, top_k=10, threshold=0.9,
                             title=None):
    import matplotlib.pyplot as plt

    cumulative = np.asarray(
        result.cumulative_explained_variance_ratio, float)[:top_k]
    index = np.arange(1, len(cumulative) + 1)
    ax = ax or plt.subplots(figsize=(6.4, 4.2))[1]
    ax.axhline(1.0, color=MUTED, ls="--", lw=0.8)
    if threshold:
        ax.axhline(threshold, color=PC_COLORS[1], ls=":", lw=1.2,
                   label=f"{threshold:.0%}")
        ax.legend(fontsize=9, frameon=False, loc="lower right", labelcolor=INK)
    ax.plot(index, cumulative, "-", color=MUTED, lw=1.2, zorder=1)
    ax.scatter(index, cumulative, s=42, color=PC_COLORS[2],
               marker=PC_MARKERS[2], zorder=3, linewidths=0)
    ax.set_xticks(index)
    ax.set_ylim(0, 1.08)
    if title is None:
        title = f"cumulative variance{_t_label(result)}"
        if result.rank_capacity:
            title += f"   (rank ceiling {result.rank_capacity})"
    _finish(ax, "number of components", "cumulative variance", title)
    return ax


def plot_spectrum_across_t(results, ax=None, top_k=10, title=None):
    """One curve per diffusion time, coloured by a sequential ramp."""
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors as mcolors

    items = list(results.items()) if isinstance(results, dict) else \
        [(r.t, r) for r in results]
    items.sort(key=lambda kv: float(kv[0]))
    ax = ax or plt.subplots(figsize=(7.0, 4.6))[1]

    times = [float(t) for t, _ in items]
    low, high = min(times), max(times)
    if high <= low:
        high = low * 1.001 + 1e-12
    norm = (mcolors.LogNorm(low, high) if low > 0 and high / low > 5
            else mcolors.Normalize(low, high))
    cmap = plt.get_cmap(SEQUENTIAL_CMAP)       # cm.get_cmap was removed in 3.9

    for t, result in items:
        lam = np.asarray(result.eigenvalues, float)
        lam = lam[lam > 0][:top_k]
        if lam.size:
            ax.plot(np.arange(1, lam.size + 1), lam, "o-", ms=4, lw=1.6,
                    color=cmap(0.12 + 0.78 * float(norm(float(t)))))
    ax.set_yscale("log")
    bar = ax.figure.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                             pad=0.02)
    bar.set_label("diffusion time $t$", color=INK)
    bar.ax.tick_params(colors=MUTED, labelsize=8)
    _finish(ax, "component $j$", r"$\lambda_j$",
            title or "spectrum across diffusion times")
    return ax


def plot_block_magnitude(result, target=None, ratio_threshold=1.2, ax=None,
                         target_label="", title=None):
    """The averaged block coordinate sqrt(sum_j PC_j^2), one panel per block.

    Give `target` (a known latent -- a label, an angle, a radius) to plot each
    block magnitude against it, which shows which block carries what.
    """
    import matplotlib.pyplot as plt

    blocks = [b for b in block_summary(result, ratio_threshold) if b["size"] > 1]
    if not blocks:
        blocks = block_summary(result, ratio_threshold)[:2]
    if not blocks:
        raise ValueError("no components to plot")

    if ax is None:
        _, ax = plt.subplots(1, len(blocks), figsize=(4.4 * len(blocks), 3.9),
                             squeeze=False)
        ax = ax.ravel()
    ax = np.atleast_1d(ax)

    for j, b in enumerate(blocks[:len(ax)]):
        panel = ax[j]
        colour = PC_COLORS[j % len(PC_COLORS)]
        caption = f"{b['label']}   {b['variance_share']:.1%} of variance"
        if target is None:
            panel.hist(b["magnitude"], bins=40, color=colour, alpha=0.85, lw=0)
            _finish(panel, rf"$\|\alpha_{{{b['label']}}}\|$", "count", caption)
        else:
            panel.scatter(np.asarray(target, float), b["magnitude"], s=9,
                          color=colour, marker=PC_MARKERS[j % len(PC_MARKERS)],
                          alpha=0.6, linewidths=0)
            _finish(panel, target_label or "target",
                    r"$\sqrt{\sum_j \alpha_j^2}$", caption)
    if title:
        ax[0].figure.suptitle(title, color=INK)
    return ax