"""Core graph-Laplacian utilities for the theory-verification experiments.

The functions here use a reduced grounded Laplacian, so no pseudoinverse of the
full singular graph Laplacian is required.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import networkx as nx
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class GroundedNetwork:
    """A connected undirected resistor network with one or more grounded nodes."""

    graph: nx.Graph
    edges: tuple[tuple[int, int], ...]
    grounded_nodes: tuple[int, ...]
    free_nodes: tuple[int, ...]
    node_to_free: dict[int, int]
    incidence_reduced: Array

    @property
    def n_free(self) -> int:
        return len(self.free_nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)


def build_grounded_network(
    graph: nx.Graph,
    grounded_nodes: Sequence[int] = (0,),
) -> GroundedNetwork:
    """Orient edges deterministically and build the reduced incidence matrix."""
    if not nx.is_connected(graph):
        raise ValueError("The graph must be connected.")

    nodes = tuple(sorted(graph.nodes()))
    grounded = tuple(sorted(set(grounded_nodes)))
    if not grounded:
        raise ValueError("At least one grounded node is required.")
    if any(node not in graph for node in grounded):
        raise ValueError("Every grounded node must belong to the graph.")

    free = tuple(node for node in nodes if node not in grounded)
    node_to_free = {node: index for index, node in enumerate(free)}

    # Deterministic orientation: smaller node index -> larger node index.
    edges = tuple((min(i, j), max(i, j)) for i, j in graph.edges())
    edges = tuple(sorted(edges))

    b_reduced = np.zeros((len(free), len(edges)), dtype=float)
    for edge_index, (tail, head) in enumerate(edges):
        if tail in node_to_free:
            b_reduced[node_to_free[tail], edge_index] = -1.0
        if head in node_to_free:
            b_reduced[node_to_free[head], edge_index] = 1.0

    return GroundedNetwork(
        graph=graph.copy(),
        edges=edges,
        grounded_nodes=grounded,
        free_nodes=free,
        node_to_free=node_to_free,
        incidence_reduced=b_reduced,
    )


def reduced_vector(network: GroundedNetwork, node_values: dict[int, float]) -> Array:
    """Build a free-node vector from a sparse node:value dictionary."""
    vector = np.zeros(network.n_free, dtype=float)
    for node, value in node_values.items():
        if node in network.grounded_nodes:
            if abs(value) > 0.0:
                raise ValueError("A grounded-node component cannot appear in a reduced vector.")
            continue
        vector[network.node_to_free[node]] = float(value)
    return vector


def laplacian(network: GroundedNetwork, conductance: Array) -> Array:
    """Return the symmetric positive-definite reduced weighted Laplacian."""
    g = np.asarray(conductance, dtype=float)
    if g.shape != (network.n_edges,):
        raise ValueError(f"Expected conductance shape {(network.n_edges,)}, got {g.shape}.")
    if np.any(g <= 0.0):
        raise ValueError("All conductances must be strictly positive.")
    b = network.incidence_reduced
    return (b * g[np.newaxis, :]) @ b.T


def solve_state(network: GroundedNetwork, conductance: Array, forcing: Array) -> Array:
    """Solve L(g) u = f for the free-node potentials."""
    f = np.asarray(forcing, dtype=float)
    if f.shape != (network.n_free,):
        raise ValueError(f"Expected forcing shape {(network.n_free,)}, got {f.shape}.")
    return np.linalg.solve(laplacian(network, conductance), f)


def task_output(
    network: GroundedNetwork,
    conductance: Array,
    forcing: Array,
    observation: Array,
) -> Array:
    """Return y(g)=C L(g)^{-1} f."""
    c = np.atleast_2d(np.asarray(observation, dtype=float))
    if c.shape[1] != network.n_free:
        raise ValueError("Observation matrix has an incompatible number of columns.")
    return c @ solve_state(network, conductance, forcing)


def analytical_jacobian(
    network: GroundedNetwork,
    conductance: Array,
    forcing: Array,
    observation: Array,
) -> Array:
    """Exact Jacobian dy/dg from the graph-Laplacian sensitivity identity."""
    c = np.atleast_2d(np.asarray(observation, dtype=float))
    lmat = laplacian(network, conductance)
    u = np.linalg.solve(lmat, forcing)
    b = network.incidence_reduced
    theta = b.T @ u
    transfer = c @ np.linalg.solve(lmat, b)
    return -transfer * theta[np.newaxis, :]


def finite_difference_jacobian(
    network: GroundedNetwork,
    conductance: Array,
    forcing: Array,
    observation: Array,
    step: float,
) -> Array:
    """Centred finite-difference Jacobian."""
    g = np.asarray(conductance, dtype=float)
    if step <= 0.0:
        raise ValueError("The finite-difference step must be positive.")
    result = np.empty((np.atleast_2d(observation).shape[0], network.n_edges), dtype=float)
    for edge in range(network.n_edges):
        plus = g.copy()
        minus = g.copy()
        plus[edge] += step
        minus[edge] -= step
        if minus[edge] <= 0.0:
            raise ValueError("The finite-difference step makes a conductance non-positive.")
        result[:, edge] = (
            task_output(network, plus, forcing, observation)
            - task_output(network, minus, forcing, observation)
        ) / (2.0 * step)
    return result


def torch_autograd_jacobian(
    network: GroundedNetwork,
    conductance: Array,
    forcing: Array,
    observation: Array,
) -> Array:
    """Independent Jacobian computed with PyTorch automatic differentiation."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch is required for the AD verification.") from exc

    dtype = torch.float64
    b = torch.as_tensor(network.incidence_reduced, dtype=dtype)
    f = torch.as_tensor(forcing, dtype=dtype)
    c = torch.as_tensor(np.atleast_2d(observation), dtype=dtype)
    g = torch.tensor(np.asarray(conductance), dtype=dtype, requires_grad=True)

    def response(g_tensor: "torch.Tensor") -> "torch.Tensor":
        lmat = (b * g_tensor.unsqueeze(0)) @ b.T
        u = torch.linalg.solve(lmat, f)
        return c @ u

    jac = torch.autograd.functional.jacobian(response, g, create_graph=False)
    return jac.detach().cpu().numpy()


def exact_task_loss(
    network: GroundedNetwork,
    conductance: Array,
    forcing: Array,
    observation: Array,
    target: Array,
) -> float:
    """Return 0.5*||y(g)-target||^2."""
    residual = task_output(network, conductance, forcing, observation) - np.asarray(target)
    return 0.5 * float(residual @ residual)


def orthogonal_projector_onto_nullspace(matrix: Array, rtol: float = 1e-11) -> Array:
    """Orthogonal projector onto ker(matrix)."""
    a = np.atleast_2d(np.asarray(matrix, dtype=float))
    if a.size == 0:
        return np.eye(a.shape[1], dtype=float)
    u, singular_values, vt = np.linalg.svd(a, full_matrices=True)
    del u
    if singular_values.size == 0:
        rank = 0
    else:
        rank = int(np.sum(singular_values > rtol * singular_values[0]))
    null_basis = vt[rank:].T
    return null_basis @ null_basis.T


def numerical_rank(matrix: Array, rtol: float = 1e-10) -> int:
    """Rank with an explicit relative singular-value threshold."""
    singular_values = np.linalg.svd(np.asarray(matrix), compute_uv=False)
    if singular_values.size == 0 or singular_values[0] == 0.0:
        return 0
    return int(np.sum(singular_values > rtol * singular_values[0]))


def minimum_norm_anchored_update(
    jacobian_new: Array,
    desired_correction: Array,
    anchoring_matrix_factor: Array,
    anchoring_strength: float,
) -> Array:
    """Solve min 0.5||J_B dg-d||^2 + 0.5*lambda||R dg||^2.

    The input anchoring_matrix_factor is R, so M=R.T@R. This formulation
    handles positive-semidefinite and rank-deficient penalties robustly.
    """
    jb = np.atleast_2d(np.asarray(jacobian_new, dtype=float))
    d = np.asarray(desired_correction, dtype=float)
    r = np.atleast_2d(np.asarray(anchoring_matrix_factor, dtype=float))
    if anchoring_strength < 0.0:
        raise ValueError("Anchoring strength must be non-negative.")

    if anchoring_strength == 0.0 or r.shape[0] == 0:
        augmented_matrix = jb
        augmented_rhs = d
    else:
        augmented_matrix = np.vstack((jb, np.sqrt(anchoring_strength) * r))
        augmented_rhs = np.concatenate((d, np.zeros(r.shape[0], dtype=float)))

    update, *_ = np.linalg.lstsq(augmented_matrix, augmented_rhs, rcond=None)
    return update


def normalise(vector: Array, *, name: str = "vector") -> Array:
    """Return a unit-norm copy and fail explicitly on a zero vector."""
    v = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(v)
    if norm == 0.0:
        raise ValueError(f"Cannot normalise the zero {name}.")
    return v / norm


def ensure_inside_bounds(
    conductance: Array,
    update: Array,
    lower: float,
    upper: float,
    safety: float = 0.95,
) -> Array:
    """Scale an update, if necessary, so g+update stays strictly inside bounds."""
    g = np.asarray(conductance, dtype=float)
    delta = np.asarray(update, dtype=float)
    scales: list[float] = [1.0]
    for g_e, d_e in zip(g, delta, strict=True):
        if d_e > 0.0:
            scales.append(safety * (upper - g_e) / d_e)
        elif d_e < 0.0:
            scales.append(safety * (lower - g_e) / d_e)
    scale = min(scales)
    if scale < 0.0:
        raise ValueError("The supplied conductance vector is already outside the bounds.")
    return min(1.0, scale) * delta
