"""Class-balanced optimal-transport mass reweighting for DIAGVI.

The alignment term of DIAGVI is an unbalanced optimal transport (OT) problem between the
latent point clouds of two modalities. By default every cell carries uniform mass, so the
two empirical marginals reflect each modality's native cell-type composition. When those
compositions disagree (label/target shift), mass conservation forces the surplus mass of a
class that is abundant in one modality but rare in the other to be transported *across*
cell-type boundaries, collapsing rare types onto abundant ones.

This module reweights the OT input masses so that the two marginals agree on the shared
cell types, while leaving the reconstruction and graph losses untouched. The construction
follows GLUE's ``estimate_balancing_weight`` (Cao & Gao, 2022), specialized to the case
where cell-type labels are available in both modalities.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
import torch

from scvi import REGISTRY_KEYS, settings

if TYPE_CHECKING:
    from scvi.data import AnnDataManager

logger = logging.getLogger(__name__)

# Lower bound on the shared-labeled fraction, guarding against division by zero.
_PHI_EPS = 1e-8


class OTClassBalancer:
    """Per-cell optimal-transport masses that balance shared cell types across modalities.

    Holds, for each modality, a lookup from that modality's integer label code to a per-cell
    base weight, together with the fraction of the modality that carries a shared-class
    label. Use :meth:`masses` to turn a minibatch's label codes into OT sample masses.

    Instances are normally created via :func:`build_ot_class_balancer` rather than directly.

    Parameters
    ----------
    class_weights
        Mapping from modality name to a 1D array of length ``n_classes`` for that modality,
        indexed by its integer label code. Entries are the per-class weight ``w_m(c)`` for
        shared classes and ``1.0`` (natural mass) for unlabeled and non-shared classes.
    phi
        Mapping from modality name to the fraction of its cells carrying a shared-class
        label.
    shared_classes
        Names of the shared classes, in a fixed order used by ``compositions``,
        ``reference``, and ``shared_weights``.
    compositions
        Mapping from modality name to that modality's composition ``p_m(c)`` over
        ``shared_classes``.
    reference
        The consensus reference ``r(c)`` over ``shared_classes``.
    shared_weights
        Mapping from modality name to the per-class weights ``w_m(c)`` over
        ``shared_classes``.
    """

    def __init__(
        self,
        class_weights: dict[str, np.ndarray],
        phi: dict[str, float],
        shared_classes: list[str],
        compositions: dict[str, np.ndarray],
        reference: np.ndarray,
        shared_weights: dict[str, np.ndarray],
    ):
        self.class_weights = class_weights
        self.phi = phi
        self.shared_classes = shared_classes
        self.compositions = compositions
        self.reference = reference
        self.shared_weights = shared_weights
        self._tensor_cache: dict[tuple, torch.Tensor] = {}

    def _weight_tensor(self, mode: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return the class-weight lookup for ``mode``, cached per device and dtype."""
        key = (mode, str(device), dtype)
        cached = self._tensor_cache.get(key)
        if cached is None:
            cached = torch.as_tensor(self.class_weights[mode], device=device, dtype=dtype)
            self._tensor_cache[key] = cached
        return cached

    def masses(
        self,
        labels: torch.Tensor,
        mode: str,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Return per-cell optimal-transport masses for one modality's minibatch.

        Each cell receives a base weight ``v_i`` -- the balancing weight of its class if that
        class is shared, else ``1.0`` -- rescaled by the expected number of shared-labeled
        cells in the minibatch, ``a_i = v_i / (B * phi_m)``. This anchors the shared block to
        a total mass of approximately one in both modalities, so their shared marginals agree.
        Cells that are unlabeled or belong to a non-shared class contribute additional mass on
        top, which the unbalanced OT ``reach`` is free to discard.

        Parameters
        ----------
        labels
            Integer label codes for this modality's minibatch, of shape ``(B,)`` or ``(B, 1)``.
        mode
            Modality name, matching the keys used to build the balancer.
        device
            Device for the returned tensor. Defaults to the device of ``labels``.
        dtype
            Floating dtype for the returned tensor. Defaults to the current torch default.

        Returns
        -------
        Strictly positive 1D tensor of shape ``(B,)``.
        """
        if mode not in self.class_weights:
            raise KeyError(
                f"Unknown modality '{mode}'. Expected one of {list(self.class_weights)}."
            )

        codes = labels.reshape(-1).long()
        device = codes.device if device is None else device
        dtype = torch.get_default_dtype() if dtype is None else dtype

        weights = self._weight_tensor(mode, device, dtype)
        v = weights[codes.to(weights.device)]
        n_cells = max(codes.shape[0], 1)
        return v / (n_cells * self.phi[mode])


def build_ot_class_balancer(
    adata_managers: dict[str, AnnDataManager],
    input_names: list[str],
    n_min: int = 20,
    kappa: float = 10.0,
) -> OTClassBalancer | None:
    """Build class-balancing optimal-transport masses from two registered modalities.

    Shared classes are matched by *name* across the two modalities (their integer codes are
    registered independently and generally differ), and admitted only if well populated in
    both. Both modalities are then reweighted toward a consensus reference -- the normalized
    geometric mean of their compositions -- which removes the proportion mismatch while
    preserving each type's relative abundance.

    Parameters
    ----------
    adata_managers
        Mapping from modality name to its :class:`~scvi.data.AnnDataManager`.
    input_names
        The two modality names, in model order.
    n_min
        Minimum number of cells of a class in *each* modality for it to be treated as shared.
        Keeps sparsely populated classes from receiving large, high-variance weights.
    kappa
        Cap on the per-class weight, which is clipped to ``[1 / kappa, kappa]``. A safety rail
        for classes whose abundance differs extremely between modalities.

    Returns
    -------
    An :class:`OTClassBalancer`, or ``None`` if no shared classes qualify, in which case the
    caller should fall back to uniform masses.

    Notes
    -----
    Counts are taken over the full registered data rather than per minibatch, so the weights
    are low variance and match the marginals in expectation.
    """
    if len(input_names) != 2:
        raise ValueError(
            f"DIAGVI class balancing requires exactly two modalities, got {len(input_names)}: "
            f"{input_names}."
        )
    if n_min < 1:
        raise ValueError(f"`n_min` must be at least 1, got {n_min}.")
    if kappa < 1.0:
        raise ValueError(f"`kappa` must be at least 1.0, got {kappa}.")

    per_mode = {}
    for name in input_names:
        manager = adata_managers[name]
        state_registry = manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY)
        mapping = np.asarray(state_registry["categorical_mapping"])
        # Only present for LabelsWithUnlabeledObsField; absent for a plain CategoricalObsField.
        unlabeled = state_registry.get("unlabeled_category", None)
        codes = np.asarray(manager.get_from_registry(REGISTRY_KEYS.LABELS_KEY)).ravel().astype(int)
        per_mode[name] = {
            "mapping": mapping,
            "unlabeled": unlabeled,
            "counts": np.bincount(codes, minlength=len(mapping)),
            "n_total": int(codes.shape[0]),
        }

    eligible = {}
    for name, info in per_mode.items():
        names_ok = set()
        for code, class_name in enumerate(info["mapping"]):
            if info["unlabeled"] is not None and class_name == info["unlabeled"]:
                continue
            if info["counts"][code] >= n_min:
                names_ok.add(class_name)
        eligible[name] = names_ok

    mode_a, mode_b = input_names
    shared_classes = sorted(eligible[mode_a] & eligible[mode_b])

    if len(shared_classes) == 0:
        warnings.warn(
            "No cell types are shared between the two modalities with at least "
            f"`n_min`={n_min} cells in each, so alignment cannot be class-balanced. "
            "Falling back to uniform optimal-transport masses. Check that both modalities "
            "were set up with a `labels_key` whose categories overlap, or lower `n_min`.",
            UserWarning,
            stacklevel=settings.warnings_stacklevel,
        )
        return None

    shared_codes, counts_shared = {}, {}
    for name in input_names:
        info = per_mode[name]
        name_to_code = {class_name: code for code, class_name in enumerate(info["mapping"])}
        shared_codes[name] = np.array([name_to_code[c] for c in shared_classes], dtype=int)
        counts_shared[name] = info["counts"][shared_codes[name]].astype(float)

    # Composition over the shared classes, with +1 smoothing.
    compositions = {}
    for name in input_names:
        smoothed = counts_shared[name] + 1.0
        compositions[name] = smoothed / smoothed.sum()

    # Consensus reference: normalized geometric mean of the two compositions.
    reference = np.sqrt(compositions[mode_a] * compositions[mode_b])
    reference = reference / reference.sum()

    class_weights, phi, shared_weights = {}, {}, {}
    for name in input_names:
        # r / p is automatically mean-one over the shared block; the clip is a safety rail.
        weights = np.clip(reference / compositions[name], 1.0 / kappa, kappa)
        shared_weights[name] = weights

        lookup = np.ones(len(per_mode[name]["mapping"]), dtype=np.float64)
        lookup[shared_codes[name]] = weights
        class_weights[name] = lookup

        n_total = max(per_mode[name]["n_total"], 1)
        phi[name] = max(float(counts_shared[name].sum()) / n_total, _PHI_EPS)

    logger.info(
        f"DIAGVI class balancing over {len(shared_classes)} shared cell types: {shared_classes}. "
        f"Shared-labeled fraction per modality: "
        f"{ {name: round(phi[name], 3) for name in input_names} }."
    )

    return OTClassBalancer(
        class_weights=class_weights,
        phi=phi,
        shared_classes=shared_classes,
        compositions=compositions,
        reference=reference,
        shared_weights=shared_weights,
    )
