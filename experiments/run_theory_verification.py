"""Generate numerical verification figures for the graph-Laplacian theory paper.

Run from the repository root:

    python -m experiments.run_theory_verification

Outputs are written to results/theory_verification/.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from continual.resistor_theory import (  # noqa: E402
    GroundedNetwork,
    analytical_jacobian,
    build_grounded_network,
    ensure_inside_bounds,
    exact_task_loss,
    finite_difference_jacobian,
    laplacian,
    minimum_norm_anchored_update,
    normalise,
    numerical_rank,
    orthogonal_projector_onto_nullspace,
    reduced_vector,
    solve_state,
    task_output,
    torch_autograd_jacobian,
)

SEED = 260613
G_MIN = 0.35
G_MAX = 2.50

OUTPUT_DIR = ROOT / "results" / "theory_verification"
FIGURE_DIR = OUTPUT_DIR / "figures"
DATA_DIR = OUTPUT_DIR / "data"


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Save a figure as both PDF and PNG."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_csv(filename: str, header: list[str], rows: list[list[float | int | str]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with (DATA_DIR / filename).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def build_grid_network() -> tuple[GroundedNetwork, dict[int, tuple[float, float]]]:
    """Create a 5x5 grid with node 0 grounded."""
    base = nx.grid_2d_graph(5, 5)
    mapping = {(row, col): 5 * row + col for row, col in base.nodes()}
    graph = nx.relabel_nodes(base, mapping)
    positions = {
        mapping[(row, col)]: (float(col), float(-row))
        for row, col in base.nodes()
    }
    return build_grounded_network(graph, grounded_nodes=(0,)), positions


def physical_tasks(network: GroundedNetwork) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one scalar old task and a three-output new task.

    Current is injected at node 24 and extracted through the grounded node 0.
    The old task observes node 4. The new task includes that same output plus
    observations at nodes 20 and 12, creating mixed conflict and safe plasticity.
    """
    forcing = reduced_vector(network, {24: 1.0})
    c_old = np.atleast_2d(reduced_vector(network, {4: 1.0}))
    c_new = np.vstack(
        (
            reduced_vector(network, {4: 1.0}),
            reduced_vector(network, {20: 1.0}),
            reduced_vector(network, {12: 1.0}),
        )
    )
    return forcing, c_old, c_new


def relative_error(reference: np.ndarray, approximation: np.ndarray) -> float:
    denominator = max(np.linalg.norm(reference), np.finfo(float).eps)
    return float(np.linalg.norm(reference - approximation) / denominator)


def fitted_loglog_slope(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & np.isfinite(y) & (y > 0.0)
    if np.count_nonzero(valid) < 3:
        return float("nan")
    slope, _ = np.polyfit(np.log(x[valid]), np.log(y[valid]), deg=1)
    return float(slope)


def experiment_sensitivity_identity(
    network: GroundedNetwork,
    conductance: np.ndarray,
    forcing: np.ndarray,
    observation: np.ndarray,
) -> dict[str, float]:
    analytical = analytical_jacobian(network, conductance, forcing, observation)
    automatic = torch_autograd_jacobian(network, conductance, forcing, observation)
    ad_error = relative_error(analytical, automatic)

    steps = np.geomspace(1.0e-8, 1.0e-2, 19)
    fd_errors: list[float] = []
    rows: list[list[float]] = []
    for step in steps:
        finite = finite_difference_jacobian(
            network, conductance, forcing, observation, float(step)
        )
        error = relative_error(analytical, finite)
        fd_errors.append(error)
        rows.append([float(step), error])

    fd_errors_array = np.asarray(fd_errors)
    best_index = int(np.argmin(fd_errors_array))

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.loglog(steps, fd_errors_array, marker="o", label="Centred finite difference")
    ax.axhline(ad_error, linestyle="--", label="Automatic differentiation error")
    ax.set_xlabel(r"Finite-difference step $\varepsilon$")
    ax.set_ylabel("Relative Jacobian error")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "figure_sensitivity_identity")

    write_csv(
        "sensitivity_identity.csv",
        ["finite_difference_step", "relative_error"],
        rows,
    )

    return {
        "automatic_differentiation_relative_error": ad_error,
        "best_finite_difference_step": float(steps[best_index]),
        "best_finite_difference_relative_error": float(fd_errors_array[best_index]),
    }


def experiment_quadratic_and_nullspace(
    network: GroundedNetwork,
    conductance: np.ndarray,
    forcing: np.ndarray,
    observation: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float]:
    target = task_output(network, conductance, forcing, observation)
    jacobian = analytical_jacobian(network, conductance, forcing, observation)

    sensitive_direction = normalise(jacobian.T[:, 0], name="sensitive direction")
    null_projector = orthogonal_projector_onto_nullspace(jacobian)
    raw_null = null_projector @ rng.normal(size=network.n_edges)
    null_direction = normalise(raw_null, name="null-space direction")

    epsilons = np.geomspace(2.0e-4, 1.0e-1, 20)
    actual_sensitive: list[float] = []
    predicted_sensitive: list[float] = []
    remainder_sensitive: list[float] = []
    actual_null: list[float] = []

    rows: list[list[float]] = []
    for epsilon in epsilons:
        ds = ensure_inside_bounds(
            conductance,
            float(epsilon) * sensitive_direction,
            G_MIN,
            G_MAX,
        )
        dn = ensure_inside_bounds(
            conductance,
            float(epsilon) * null_direction,
            G_MIN,
            G_MAX,
        )

        f_sensitive = exact_task_loss(
            network, conductance + ds, forcing, observation, target
        )
        f_predicted = 0.5 * float(np.linalg.norm(jacobian @ ds) ** 2)
        f_null = exact_task_loss(network, conductance + dn, forcing, observation, target)

        actual_sensitive.append(f_sensitive)
        predicted_sensitive.append(f_predicted)
        remainder_sensitive.append(abs(f_sensitive - f_predicted))
        actual_null.append(f_null)
        rows.append(
            [
                float(epsilon),
                f_sensitive,
                f_predicted,
                abs(f_sensitive - f_predicted),
                f_null,
            ]
        )

    actual_sensitive_array = np.asarray(actual_sensitive)
    predicted_sensitive_array = np.asarray(predicted_sensitive)
    remainder_array = np.asarray(remainder_sensitive)
    actual_null_array = np.asarray(actual_null)

    quadratic_mask = (epsilons >= 1.0e-3) & (epsilons <= 3.0e-2)
    cubic_mask = (epsilons >= 1.0e-3) & (epsilons <= 5.0e-2)
    quartic_mask = (epsilons >= 1.5e-3) & (epsilons <= 7.0e-2)

    sensitive_slope = fitted_loglog_slope(
        epsilons, actual_sensitive_array, quadratic_mask
    )
    remainder_slope = fitted_loglog_slope(epsilons, remainder_array, cubic_mask)
    null_slope = fitted_loglog_slope(epsilons, actual_null_array, quartic_mask)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.loglog(epsilons, actual_sensitive_array, marker="o", label="Actual forgetting")
    ax.loglog(
        epsilons,
        predicted_sensitive_array,
        linestyle="--",
        label=r"Quadratic prediction $\frac{1}{2}\|J_A\Delta g\|^2$",
    )
    ax.set_xlabel(r"Perturbation size $\varepsilon$")
    ax.set_ylabel("Old-task forgetting")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "figure_quadratic_forgetting")

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.loglog(epsilons, remainder_array, marker="o")
    ax.set_xlabel(r"Perturbation size $\varepsilon$")
    ax.set_ylabel("Absolute Taylor remainder")
    fig.tight_layout()
    save_figure(fig, "figure_quadratic_remainder")

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.loglog(
        epsilons,
        actual_sensitive_array,
        marker="o",
        label="Sensitive-subspace update",
    )
    ax.loglog(
        epsilons,
        actual_null_array,
        marker="s",
        label="Null-space update",
    )
    ax.set_xlabel(r"Perturbation size $\varepsilon$")
    ax.set_ylabel("Old-task forgetting")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "figure_null_vs_sensitive")

    write_csv(
        "quadratic_and_nullspace.csv",
        [
            "epsilon",
            "actual_sensitive_forgetting",
            "quadratic_prediction",
            "absolute_remainder",
            "actual_nullspace_forgetting",
        ],
        rows,
    )

    return {
        "sensitive_forgetting_loglog_slope": sensitive_slope,
        "quadratic_remainder_loglog_slope": remainder_slope,
        "nullspace_forgetting_loglog_slope": null_slope,
        "nullspace_projection_residual": float(np.linalg.norm(jacobian @ null_direction)),
    }


def random_scalar_task_jacobian(
    network: GroundedNetwork,
    conductance: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    forcing = normalise(rng.normal(size=network.n_free), name="random forcing")
    observation = np.atleast_2d(
        normalise(rng.normal(size=network.n_free), name="random observation")
    )
    row = analytical_jacobian(network, conductance, forcing, observation)[0]
    return normalise(row, name="random task Jacobian row")


def experiment_rank_capacity(
    network: GroundedNetwork,
    conductance: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, int]:
    n_tasks = network.n_edges + 20
    independent_rows = [
        random_scalar_task_jacobian(network, conductance, rng)
        for _ in range(n_tasks)
    ]

    redundant_bases = independent_rows[:5]
    redundant_rows = [redundant_bases[index % len(redundant_bases)] for index in range(n_tasks)]

    independent_rank: list[int] = []
    redundant_rank: list[int] = []
    independent_dimension: list[int] = []
    redundant_dimension: list[int] = []
    rows: list[list[int]] = []

    for count in range(1, n_tasks + 1):
        rank_ind = numerical_rank(np.vstack(independent_rows[:count]), rtol=1.0e-9)
        rank_red = numerical_rank(np.vstack(redundant_rows[:count]), rtol=1.0e-9)
        independent_rank.append(rank_ind)
        redundant_rank.append(rank_red)
        independent_dimension.append(network.n_edges - rank_ind)
        redundant_dimension.append(network.n_edges - rank_red)
        rows.append(
            [
                count,
                rank_ind,
                network.n_edges - rank_ind,
                rank_red,
                network.n_edges - rank_red,
            ]
        )

    task_counts = np.arange(1, n_tasks + 1)
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot(task_counts, independent_rank, label="Independent-task rank")
    ax.plot(task_counts, independent_dimension, label="Independent-task plastic dimension")
    ax.plot(task_counts, redundant_rank, linestyle="--", label="Redundant-task rank")
    ax.plot(
        task_counts,
        redundant_dimension,
        linestyle="--",
        label="Redundant-task plastic dimension",
    )
    ax.set_xlabel("Number of retained scalar tasks")
    ax.set_ylabel("Dimension")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "figure_rank_capacity")

    write_csv(
        "rank_capacity.csv",
        [
            "number_of_tasks",
            "independent_rank",
            "independent_plastic_dimension",
            "redundant_rank",
            "redundant_plastic_dimension",
        ],
        rows,
    )

    first_full_rank = next(
        (index + 1 for index, rank in enumerate(independent_rank) if rank == network.n_edges),
        -1,
    )
    return {
        "number_of_edges": network.n_edges,
        "first_task_count_with_full_independent_rank": first_full_rank,
        "final_redundant_rank": redundant_rank[-1],
    }



def dominant_safe_direction(
    jacobian_old: np.ndarray,
    jacobian_new: np.ndarray,
    *,
    tolerance: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Construct the most effective unit direction in the old-task null space.

    The direction solves

        maximise ||J_B v||
        subject to J_A v = 0 and ||v|| = 1.

    It is obtained from the dominant right singular vector of

        J_B P_A,

    where P_A is the orthogonal projector onto ker(J_A).

    Parameters
    ----------
    jacobian_old
        Old-task Jacobian J_A with shape (p_A, M).
    jacobian_new
        New-task Jacobian J_B with shape (p_B, M).
    tolerance
        Numerical tolerance used in rank and null-space checks.

    Returns
    -------
    safe_direction
        Unit conductance direction in ker(J_A).
    safe_output_direction
        Unit new-task output direction generated by the safe direction.
        Its first component is set exactly to zero because the first
        new-task output is assumed to be identical to the old-task output.
    safe_gain
        Norm of the new-task output response J_B v_s before normalisation.
    """
    projector_old_null = orthogonal_projector_onto_nullspace(jacobian_old)
    projected_new_jacobian = jacobian_new @ projector_old_null

    _, singular_values, right_vectors_t = np.linalg.svd(
        projected_new_jacobian,
        full_matrices=False,
    )

    if singular_values.size == 0 or singular_values[0] <= tolerance:
        raise RuntimeError(
            "The new task has no numerically useful response in ker(J_A)."
        )

    # The dominant right singular vector of J_B P_A gives the direction
    # of maximal new-task response within the old-task null space.
    safe_direction = projector_old_null @ right_vectors_t[0, :]

    safe_direction_norm = float(np.linalg.norm(safe_direction))
    if safe_direction_norm <= tolerance:
        raise RuntimeError(
            "Failed to construct a nonzero old-task-safe direction."
        )

    safe_direction /= safe_direction_norm

    old_task_sensitivity = float(
        np.linalg.norm(jacobian_old @ safe_direction)
    )
    if old_task_sensitivity > 100.0 * tolerance:
        raise RuntimeError(
            "The constructed direction is not sufficiently close to ker(J_A). "
            f"Residual sensitivity: {old_task_sensitivity:.3e}"
        )

    safe_response = jacobian_new @ safe_direction

    # The first new-task observation is required to be exactly the old-task
    # observation. Therefore, its response must vanish for a safe direction.
    if abs(float(safe_response[0])) > 100.0 * tolerance:
        raise RuntimeError(
            "The first new-task output is not numerically preserved by the "
            "constructed safe direction."
        )

    safe_response[0] = 0.0
    safe_gain = float(np.linalg.norm(safe_response[1:]))

    if safe_gain <= tolerance:
        raise RuntimeError(
            "The safe direction does not influence the additional new-task "
            "outputs."
        )

    # Fix the sign ambiguity of singular vectors deterministically.
    accessible_response = safe_response[1:]
    pivot_local = int(np.argmax(np.abs(accessible_response)))
    pivot_global = pivot_local + 1

    if safe_response[pivot_global] < 0.0:
        safe_direction = -safe_direction
        safe_response = -safe_response

    safe_response[0] = 0.0
    safe_output_direction = safe_response / safe_gain

    return safe_direction, safe_output_direction, safe_gain


def experiment_anchoring(
    network: GroundedNetwork,
    conductance: np.ndarray,
    forcing: np.ndarray,
    observation_old: np.ndarray,
    observation_new: np.ndarray,
) -> dict[str, object]:
    """Verify uniform and sensitivity anchoring using manufactured tasks.

    The supplied conductance vector is treated as the exact-fit old-task
    solution g_A. The old target is defined by y_A^* = y_A(g_A).

    The new target correction has the form

        d_B = delta * (e_1 + q_safe),

    where e_1 changes the output shared with the old task and is therefore
    incompatible with exact first-order retention, while q_safe is generated
    by a conductance direction in ker(J_A).
    """
    tolerance = 1.0e-12
    target_scale = 5.0e-3

    # ------------------------------------------------------------------
    # Validate the task-observation structure.
    # ------------------------------------------------------------------
    if observation_old.ndim != 2 or observation_old.shape[0] != 1:
        raise ValueError(
            "observation_old must contain exactly one scalar-output row."
        )

    if observation_new.ndim != 2 or observation_new.shape[0] < 2:
        raise ValueError(
            "observation_new must contain the old output followed by at "
            "least one additional output."
        )

    if observation_old.shape[1] != observation_new.shape[1]:
        raise ValueError(
            "Old and new observation matrices must act on the same node space."
        )

    if not np.allclose(
        observation_new[0, :],
        observation_old[0, :],
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError(
            "The first row of observation_new must be exactly the old-task "
            "observation row."
        )

    # ------------------------------------------------------------------
    # Compute exact Jacobians at the reference conductance g_A.
    # ------------------------------------------------------------------
    jacobian_old = analytical_jacobian(
        network,
        conductance,
        forcing,
        observation_old,
    )
    jacobian_new = analytical_jacobian(
        network,
        conductance,
        forcing,
        observation_new,
    )

    if not np.allclose(
        jacobian_new[0, :],
        jacobian_old[0, :],
        rtol=1.0e-11,
        atol=1.0e-13,
    ):
        raise RuntimeError(
            "The first row of J_B must equal J_A because the first new-task "
            "output is the old-task output."
        )

    # ------------------------------------------------------------------
    # Construct the deterministic safe conductance and output directions.
    # ------------------------------------------------------------------
    (
        safe_direction,
        safe_output_direction,
        safe_gain,
    ) = dominant_safe_direction(
        jacobian_old,
        jacobian_new,
        tolerance=tolerance,
    )

    output_dimension_new = observation_new.shape[0]

    incompatible_output_direction = np.zeros(
        output_dimension_new,
        dtype=float,
    )
    incompatible_output_direction[0] = 1.0

    # q_safe has first component zero and unit norm on the additional
    # new-task outputs.
    desired_correction = target_scale * (
        incompatible_output_direction + safe_output_direction
    )

    # ------------------------------------------------------------------
    # Manufacture exact old and new targets.
    # ------------------------------------------------------------------
    old_target = task_output(
        network,
        conductance,
        forcing,
        observation_old,
    )

    new_output_at_old = task_output(
        network,
        conductance,
        forcing,
        observation_new,
    )

    new_target = new_output_at_old + desired_correction

    old_residual_at_reference = task_output(
        network,
        conductance,
        forcing,
        observation_old,
    ) - old_target

    if np.linalg.norm(old_residual_at_reference) > tolerance:
        raise RuntimeError(
            "The manufactured old task is not fitted exactly at g_A."
        )

    # A safe linearised update that generates the accessible component.
    safe_linearised_update = (
        target_scale / safe_gain
    ) * safe_direction

    safe_linearised_response = (
        jacobian_new @ safe_linearised_update
    )

    required_safe_response = (
        target_scale * safe_output_direction
    )

    safe_response_error = float(
        np.linalg.norm(
            safe_linearised_response - required_safe_response
        )
    )

    if safe_response_error > 100.0 * tolerance:
        raise RuntimeError(
            "The explicitly constructed safe update does not generate the "
            "required safe output correction."
        )

    # The no-adaptation loss is half the squared correction norm.
    unadapted_new_loss = 0.5 * float(
        np.linalg.norm(desired_correction) ** 2
    )

    # Only the incompatible first coordinate remains at the ideal
    # first-order constrained optimum.
    safe_linearised_loss_floor = 0.5 * target_scale**2

    # ------------------------------------------------------------------
    # Anchoring sweep.
    # ------------------------------------------------------------------
    lambdas = np.concatenate(
        (
            np.array([0.0]),
            np.geomspace(1.0e-6, 1.0e5, 60),
        )
    )

    # minimum_norm_anchored_update is assumed to use the penalty
    #
    #     (lambda / 2) ||factor @ Delta g||^2.
    #
    # Therefore:
    #   factor = I       gives uniform anchoring;
    #   factor = J_A     gives J_A^T J_A sensitivity anchoring.
    methods = {
        "Uniform anchoring": np.eye(network.n_edges),
        "Sensitivity anchoring": jacobian_old,
    }

    records: list[list[float | str]] = []
    summary: dict[str, object] = {}

    normalized_new_loss_curves: dict[str, list[float]] = {}

    for method, factor in methods.items():
        actual_old: list[float] = []
        predicted_old: list[float] = []
        actual_new: list[float] = []
        predicted_new: list[float] = []
        normalized_new: list[float] = []
        update_norms: list[float] = []
        bound_scales: list[float] = []

        for anchoring_strength in lambdas:
            raw_update = minimum_norm_anchored_update(
                jacobian_new,
                desired_correction,
                factor,
                float(anchoring_strength),
            )

            bounded_update = ensure_inside_bounds(
                conductance,
                raw_update,
                G_MIN,
                G_MAX,
                safety=0.98,
            )

            raw_norm = float(np.linalg.norm(raw_update))
            bounded_norm = float(np.linalg.norm(bounded_update))

            if raw_norm > 0.0:
                bound_scale = bounded_norm / raw_norm
            else:
                bound_scale = 1.0

            # Bound rescaling would mean that the applied update is no longer
            # the exact minimiser of the local anchored quadratic problem.
            if bound_scale < 1.0 - 1.0e-11:
                raise RuntimeError(
                    "The analytical anchored update reached the conductance "
                    "bounds and was rescaled. Reduce target_scale to preserve "
                    "the exact local verification. "
                    f"Method={method}, lambda={anchoring_strength:.3e}, "
                    f"scale={bound_scale:.6f}"
                )

            update = bounded_update
            updated_conductance = conductance + update

            old_actual = exact_task_loss(
                network,
                updated_conductance,
                forcing,
                observation_old,
                old_target,
            )

            old_predicted = 0.5 * float(
                np.linalg.norm(jacobian_old @ update) ** 2
            )

            new_actual = exact_task_loss(
                network,
                updated_conductance,
                forcing,
                observation_new,
                new_target,
            )

            new_predicted = 0.5 * float(
                np.linalg.norm(
                    jacobian_new @ update - desired_correction
                ) ** 2
            )

            normalized_loss = (
                new_actual / unadapted_new_loss
                if unadapted_new_loss > 0.0
                else np.nan
            )

            actual_old.append(old_actual)
            predicted_old.append(old_predicted)
            actual_new.append(new_actual)
            predicted_new.append(new_predicted)
            normalized_new.append(float(normalized_loss))
            update_norms.append(bounded_norm)
            bound_scales.append(float(bound_scale))

            records.append(
                [
                    method,
                    float(anchoring_strength),
                    old_actual,
                    old_predicted,
                    new_actual,
                    new_predicted,
                    float(normalized_loss),
                    bounded_norm,
                    float(bound_scale),
                ]
            )

        positive_lambda = lambdas[1:]
        positive_old = np.asarray(actual_old[1:], dtype=float)
        positive_update_norms = np.asarray(
            update_norms[1:],
            dtype=float,
        )

        large_lambda_mask = positive_lambda >= 1.0e2

        method_key = method.lower().replace(" ", "_")

        if method == "Uniform anchoring":
            summary["uniform_large_lambda_update_slope"] = (
                fitted_loglog_slope(
                    positive_lambda,
                    positive_update_norms,
                    large_lambda_mask,
                )
            )
            summary["uniform_large_lambda_forgetting_slope"] = (
                fitted_loglog_slope(
                    positive_lambda,
                    positive_old,
                    large_lambda_mask,
                )
            )

        summary[f"{method_key}_terminal_update_norm"] = float(
            update_norms[-1]
        )
        summary[f"{method_key}_terminal_old_forgetting"] = float(
            actual_old[-1]
        )
        summary[f"{method_key}_terminal_new_loss"] = float(
            actual_new[-1]
        )
        summary[f"{method_key}_terminal_normalized_new_loss"] = float(
            normalized_new[-1]
        )
        summary[f"{method_key}_minimum_bound_scale"] = float(
            min(bound_scales)
        )

        normalized_new_loss_curves[method] = normalized_new

        # --------------------------------------------------------------
        # Actual versus local predicted forgetting.
        # --------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(7.0, 5.0))

        positive_actual_old = np.maximum(
            np.asarray(actual_old[1:], dtype=float),
            1.0e-30,
        )
        positive_predicted_old = np.maximum(
            np.asarray(predicted_old[1:], dtype=float),
            1.0e-30,
        )

        ax.loglog(
            positive_lambda,
            positive_actual_old,
            label="Actual old-task forgetting",
        )
        ax.loglog(
            positive_lambda,
            positive_predicted_old,
            linestyle="--",
            label="Local forgetting prediction",
        )
        ax.set_xlabel(r"Anchoring strength $\lambda$")
        ax.set_ylabel("Old-task forgetting")
        ax.legend()
        fig.tight_layout()

        stem = (
            "figure_anchoring_formula_uniform"
            if method == "Uniform anchoring"
            else "figure_anchoring_formula_sensitivity"
        )
        save_figure(fig, stem)

    # ------------------------------------------------------------------
    # Clear adaptation plot: normalized new-task loss versus lambda.
    # Uniform anchoring should tend to 1, while sensitivity anchoring
    # should approach the constrained safe-learning level, approximately
    # 1/2 for the manufactured target used here.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    for method, normalized_losses in normalized_new_loss_curves.items():
        ax.semilogx(
            lambdas[1:],
            np.asarray(normalized_losses[1:], dtype=float),
            marker="o",
            markevery=5,
            label=method,
        )

    normalized_safe_floor = (
        safe_linearised_loss_floor / unadapted_new_loss
    )

    ax.axhline(
        1.0,
        linestyle=":",
        label="No-adaptation level",
    )
    ax.axhline(
        normalized_safe_floor,
        linestyle="--",
        label="First-order safe-learning limit",
    )
    ax.set_xlabel(r"Anchoring strength $\lambda$")
    ax.set_ylabel(
        r"Normalized new-task loss "
        r"$\mathcal{L}_B(\lambda)/\mathcal{L}_B(g_A)$"
    )
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "figure_anchoring_tradeoff")

    # ------------------------------------------------------------------
    # Save machine-readable sweep and exact target construction.
    # ------------------------------------------------------------------
    write_csv(
        "anchoring.csv",
        [
            "method",
            "lambda",
            "actual_old_forgetting",
            "predicted_old_forgetting",
            "actual_new_loss",
            "predicted_new_loss",
            "normalized_new_loss",
            "update_norm",
            "bound_scale",
        ],
        records,
    )

    np.save(
        DATA_DIR / "anchoring_reference_conductance.npy",
        conductance,
    )
    np.save(
        DATA_DIR / "anchoring_old_target.npy",
        old_target,
    )
    np.save(
        DATA_DIR / "anchoring_new_output_at_old.npy",
        new_output_at_old,
    )
    np.save(
        DATA_DIR / "anchoring_new_target.npy",
        new_target,
    )
    np.save(
        DATA_DIR / "anchoring_desired_correction.npy",
        desired_correction,
    )
    np.save(
        DATA_DIR / "anchoring_safe_direction.npy",
        safe_direction,
    )
    np.save(
        DATA_DIR / "anchoring_safe_output_direction.npy",
        safe_output_direction,
    )
    np.save(
        DATA_DIR / "anchoring_safe_linearised_update.npy",
        safe_linearised_update,
    )

    summary["target_scale"] = float(target_scale)
    summary["old_target"] = old_target.tolist()
    summary["new_output_at_old"] = new_output_at_old.tolist()
    summary["new_target"] = new_target.tolist()
    summary["desired_new_output_correction"] = (
        desired_correction.tolist()
    )
    summary["desired_new_output_correction_norm"] = float(
        np.linalg.norm(desired_correction)
    )
    summary["safe_output_direction"] = (
        safe_output_direction.tolist()
    )
    summary["safe_gain"] = float(safe_gain)
    summary["safe_direction_old_sensitivity"] = float(
        np.linalg.norm(jacobian_old @ safe_direction)
    )
    summary["safe_linearised_update_norm"] = float(
        np.linalg.norm(safe_linearised_update)
    )
    summary["safe_linearised_response_error"] = float(
        safe_response_error
    )
    summary["unadapted_new_loss"] = float(unadapted_new_loss)
    summary["safe_linearised_new_loss_floor"] = float(
        safe_linearised_loss_floor
    )
    summary["normalized_safe_linearised_loss_floor"] = float(
        normalized_safe_floor
    )

    return summary





def scaled_widths(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=float)
    maximum = float(np.max(values))
    if maximum <= 0.0:
        return [1.0] * values.size
    return list(0.5 + 5.5 * values / maximum)


def draw_edge_metric(
    network: GroundedNetwork,
    positions: dict[int, tuple[float, float]],
    values: np.ndarray,
    title: str,
    filename: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    nx.draw_networkx_nodes(network.graph, positions, node_size=180, ax=ax)
    nx.draw_networkx_edges(
        network.graph,
        positions,
        edgelist=list(network.edges),
        width=scaled_widths(values),
        ax=ax,
    )
    nx.draw_networkx_labels(network.graph, positions, font_size=7, ax=ax)
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    save_figure(fig, filename)


def experiment_edge_maps(
    network: GroundedNetwork,
    positions: dict[int, tuple[float, float]],
    conductance: np.ndarray,
    forcing: np.ndarray,
    observation: np.ndarray,
) -> dict[str, float]:
    lmat = laplacian(network, conductance)
    u = solve_state(network, conductance, forcing)
    c = observation[0]
    w = np.linalg.solve(lmat, c)
    b = network.incidence_reduced

    forward_drop = np.abs(b.T @ u)
    current = np.abs(conductance * (b.T @ u))
    adjoint_drop = np.abs(b.T @ w)
    sensitivity = forward_drop * adjoint_drop

    draw_edge_metric(
        network,
        positions,
        forward_drop,
        "Forward voltage-drop magnitude",
        "figure_edge_forward_voltage",
    )
    draw_edge_metric(
        network,
        positions,
        current,
        "Forward current magnitude",
        "figure_edge_current",
    )
    draw_edge_metric(
        network,
        positions,
        adjoint_drop,
        "Adjoint output-transfer magnitude",
        "figure_edge_adjoint_voltage",
    )
    draw_edge_metric(
        network,
        positions,
        sensitivity,
        "Edge sensitivity magnitude",
        "figure_edge_sensitivity_map",
    )

    correlation = float(np.corrcoef(current, sensitivity)[0, 1])
    write_csv(
        "edge_metrics.csv",
        [
            "edge_index",
            "tail",
            "head",
            "absolute_forward_voltage_drop",
            "absolute_current",
            "absolute_adjoint_voltage_drop",
            "sensitivity",
        ],
        [
            [
                index,
                tail,
                head,
                float(forward_drop[index]),
                float(current[index]),
                float(adjoint_drop[index]),
                float(sensitivity[index]),
            ]
            for index, (tail, head) in enumerate(network.edges)
        ],
    )
    return {"current_sensitivity_correlation": correlation}


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    network, positions = build_grid_network()
    conductance = rng.uniform(0.75, 1.25, size=network.n_edges)
    forcing, observation_old, observation_new = physical_tasks(network)

    summary: dict[str, object] = {
        "seed": SEED,
        "number_of_free_nodes": network.n_free,
        "number_of_edges": network.n_edges,
        "grounded_nodes": list(network.grounded_nodes),
        "conductance_minimum": float(np.min(conductance)),
        "conductance_maximum": float(np.max(conductance)),
    }

    summary["sensitivity_identity"] = experiment_sensitivity_identity(
        network, conductance, forcing, observation_old
    )
    summary["quadratic_and_nullspace"] = experiment_quadratic_and_nullspace(
        network, conductance, forcing, observation_old, rng
    )
    summary["rank_capacity"] = experiment_rank_capacity(network, conductance, rng)
    summary["anchoring"] = experiment_anchoring(
        network,
        conductance,
        forcing,
        observation_old,
        observation_new,
    )
    summary["edge_maps"] = experiment_edge_maps(
        network, positions, conductance, forcing, observation_old
    )

    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    np.save(DATA_DIR / "conductance.npy", conductance)

    print(json.dumps(summary, indent=2))
    print(f"\nFigures: {FIGURE_DIR}")
    print(f"Data:    {DATA_DIR}")


if __name__ == "__main__":
    main()
