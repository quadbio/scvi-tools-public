"""Utility functions for the DIAGVI model."""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from scvi.data._download import _download
from scvi.utils import dependencies

if TYPE_CHECKING:
    from typing import Literal

    from anndata import AnnData
    from torch_geometric.data import Data

logger = logging.getLogger(__name__)

# Regex for parsing GTF attribute column
_GTF_ATTR_RE = re.compile(r'(\S+)\s+"([^"]+)"')


def _read_gtf_genes(gtf_path: str) -> pd.DataFrame:
    """Read gene records from GTF file.

    GTF is 1-based inclusive. Converts to 0-based half-open (BED-style).

    Parameters
    ----------
    gtf_path
        Path to GTF file (supports .gz compression).

    Returns
    -------
    DataFrame with chrom, strand, chromStart, chromEnd, attribute columns.
    """
    cols = ["chrom", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"]
    gtf = pd.read_csv(
        gtf_path,
        sep="\t",
        comment="#",
        header=None,
        names=cols,
        compression="infer",
        low_memory=False,
    )
    gtf = gtf.query("feature == 'gene'").copy()
    gtf["chromStart"] = gtf["start"] - 1  # Convert to 0-based
    gtf["chromEnd"] = gtf["end"]  # Keep end (BED-style half-open)
    return gtf[["chrom", "strand", "chromStart", "chromEnd", "attribute"]]


def _extract_gtf_attr(attr_series: pd.Series, key: str) -> pd.Series:
    """Extract a specific attribute from GTF attribute column.

    Parameters
    ----------
    attr_series
        Series containing GTF attribute strings.
    key
        Attribute key to extract (e.g., "gene_name", "gene_id").

    Returns
    -------
    Series with extracted attribute values.
    """

    def get_key(s):
        if pd.isna(s):
            return np.nan
        for k, v in _GTF_ATTR_RE.findall(s):
            if k == key:
                return v
        return np.nan

    return attr_series.map(get_key)


def add_gene_coords_from_gtf(
    adata: AnnData,
    gtf_path: str,
    gtf_key: str = "gene_name",
    var_key: str | None = None,
) -> AnnData:
    """Add genomic coordinates from GTF file to adata.var.

    Parameters
    ----------
    adata
        AnnData object to annotate.
    gtf_path
        Path to GTF file (supports .gz compression).
    gtf_key
        GTF attribute to match against var (default: "gene_name").
        Common values: "gene_name", "gene_id".
    var_key
        Column in adata.var to match. If None, uses var_names.

    Returns
    -------
    AnnData with chrom, chromStart, chromEnd added to var. strand is added if available in GTF.

    Examples
    --------
    >>> add_gene_coords_from_gtf(rna_adata, "gencode.vM25.gtf.gz")
    >>> # Or match on gene_id instead of gene_name:
    >>> add_gene_coords_from_gtf(rna_adata, "gencode.gtf.gz", gtf_key="gene_id")
    """
    # Minimum required columns
    required_cols = ["chrom", "chromStart", "chromEnd"]

    # Read GTF first to check what's available
    gtf = _read_gtf_genes(gtf_path)

    # Check if strand is available (non-null values exist)
    has_strand = gtf["strand"].notna().any()
    cols_to_add = required_cols + (["strand"] if has_strand else [])

    # Skip if already annotated
    if all(c in adata.var.columns for c in cols_to_add):
        logger.info(f"adata.var already has columns {cols_to_add} - skipping GTF annotation.")
        return adata
    gtf[gtf_key] = _extract_gtf_attr(gtf["attribute"], gtf_key)

    # De-duplicate (keep last occurrence, matching GLUE behavior)
    gtf = gtf.dropna(subset=[gtf_key]).drop_duplicates(subset=[gtf_key], keep="last")

    # Get keys to match from AnnData
    if var_key is None:
        keys = adata.var_names
    else:
        keys = adata.var[var_key]

    # Left-join, preserving adata.var index
    merge_df = gtf.set_index(gtf_key)[cols_to_add].reindex(keys).set_index(adata.var.index)

    missing = merge_df["chrom"].isna().sum()
    logger.info(f"Added gene coordinates from GTF. Missing annotations: {missing}/{len(merge_df)}")

    adata.var = adata.var.join(merge_df)
    return adata


def _dist_power_decay(d: np.ndarray | float) -> np.ndarray | float:
    """GLUE's distance-based power decay weight.

    Computes weight as: ((|d| + 1000) / 1000)^(-0.75)

    Parameters
    ----------
    d
        Distance in base pairs (can be array or scalar).

    Returns
    -------
    Decaying weight (1.0 at distance 0, ~0.19 at 10kb, ~0.03 at 150kb).
    """
    return ((np.abs(d) + 1000) / 1000) ** (-0.75)


def _interval_distance_scalar(
    gene_start: int, gene_end: int, peak_start: int, peak_end: int
) -> int:
    """Compute signed distance between two genomic intervals (scalar version).

    Parameters
    ----------
    gene_start
        Start position of gene interval.
    gene_end
        End position of gene interval.
    peak_start
        Start position of peak interval.
    peak_end
        End position of peak interval.

    Returns
    -------
    Signed distance:
    - 0 if intervals overlap
    - Negative if peak is before (upstream of) gene
    - Positive if peak is after (downstream of) gene
    """
    if gene_start < peak_end and peak_start < gene_end:
        return 0  # Overlap
    elif peak_end <= gene_start:
        return peak_end - gene_start - 1  # Negative (peak before gene)
    else:
        return peak_start - gene_end + 1  # Positive (peak after gene)


def construct_peak_gene_mapping(
    rna_adata: AnnData,
    atac_adata: AnnData,
    gene_region: str = "auto",
    promoter_len: int = 2000,
    extend_range: int = 150000,
    rna_key: str = "rna",
    atac_key: str = "atac",
) -> pd.DataFrame:
    """Build gene-peak mapping DataFrame using genomic coordinates.

    Creates a mapping between genes and peaks based on genomic proximity,
    following the GLUE approach. Peaks within extend_range are linked with
    distance-decaying weights.

    Parameters
    ----------
    rna_adata
        RNA AnnData object. Must have var columns: chrom, chromStart, chromEnd.
        If gene_region is "promoter" or "combined", also requires "strand".
    atac_adata
        ATAC AnnData object. Must have var columns: chrom, chromStart, chromEnd.
    gene_region
        Defines the genomic region of genes:
        - "gene_body": Use gene coordinates as-is (no strand required)
        - "promoter": TSS region only, extended by promoter_len (requires strand)
        - "combined": Gene body + promoter upstream (requires strand)
        - "auto": Use "combined" if strand available, else "gene_body" (default)
    promoter_len
        Length of promoter region upstream of TSS to include (default 2000 bp).
    extend_range
        Maximum distance beyond gene region to link peaks (default 150000 bp).
    rna_key
        Column name for RNA features in output DataFrame.
    atac_key
        Column name for ATAC features in output DataFrame.

    Returns
    -------
    DataFrame with columns: [rna_key, atac_key, weight, sign]
    suitable for use as mapping_df in _construct_guidance_graph().

    Raises
    ------
    ValueError
        If required var columns are missing from either AnnData.
    """
    # Auto-detect gene_region based on strand availability
    has_strand = "strand" in rna_adata.var.columns
    if gene_region == "auto":
        gene_region = "combined" if has_strand else "gene_body"
        logger.info(
            f"Auto-detected gene_region='{gene_region}' "
            f"(strand column {'found' if has_strand else 'not found'})."
        )

    # Validate gene_region value
    valid_regions = ("gene_body", "promoter", "combined")
    if gene_region not in valid_regions:
        raise ValueError(
            f"gene_region must be one of {valid_regions} or 'auto'. Got: '{gene_region}'"
        )

    # Validate strand requirement for promoter/combined
    if gene_region in ("promoter", "combined") and not has_strand:
        raise ValueError(
            f"gene_region='{gene_region}' requires 'strand' column in rna_adata.var. "
            f"Use gene_region='gene_body' or add strand information."
        )

    # Validate required ATAC columns
    required_atac = ["chrom", "chromStart", "chromEnd"]
    missing_atac = [c for c in required_atac if c not in atac_adata.var.columns]
    if missing_atac:
        raise ValueError(f"ATAC AnnData.var missing required columns: {missing_atac}")

    # Validate required RNA columns
    required_rna = ["chrom", "chromStart", "chromEnd"]
    if gene_region in ("promoter", "combined"):
        required_rna.append("strand")
    missing_rna = [c for c in required_rna if c not in rna_adata.var.columns]
    if missing_rna:
        raise ValueError(f"RNA AnnData.var missing required columns: {missing_rna}")

    # Prepare gene DataFrame
    genes = rna_adata.var[required_rna].copy()
    genes["gene_name"] = rna_adata.var_names

    # Apply gene region transformation
    if gene_region == "gene_body":
        # No modification - use gene coordinates as-is
        pass

    elif gene_region == "promoter":
        # Convert to TSS, then expand by promoter_len
        plus_strand = genes["strand"] == "+"
        minus_strand = genes["strand"] == "-"

        # Set to TSS (strand-specific start site)
        # For + strand: TSS is chromStart
        # For - strand: TSS is chromEnd
        genes.loc[plus_strand, "chromEnd"] = genes.loc[plus_strand, "chromStart"] + 1
        genes.loc[minus_strand, "chromStart"] = genes.loc[minus_strand, "chromEnd"] - 1

        # Expand upstream by promoter_len
        genes.loc[plus_strand, "chromStart"] -= promoter_len
        genes.loc[minus_strand, "chromEnd"] += promoter_len

    elif gene_region == "combined":
        # Expand gene body upstream by promoter_len (strand-aware)
        plus_strand = genes["strand"] == "+"
        minus_strand = genes["strand"] == "-"
        genes.loc[plus_strand, "chromStart"] -= promoter_len
        genes.loc[minus_strand, "chromEnd"] += promoter_len

    # Ensure chromStart is non-negative
    genes["chromStart"] = genes["chromStart"].clip(lower=0)

    # Prepare peak DataFrame
    peaks = atac_adata.var[["chrom", "chromStart", "chromEnd"]].copy()
    peaks["peak_name"] = atac_adata.var_names

    # P3: Sweep-line algorithm (GLUE-style) - O(n log n + edges) instead of O(n*m)
    # Sort genes and peaks by chromosome and position
    genes = genes.sort_values(["chrom", "chromStart"]).reset_index(drop=True)
    peaks = peaks.sort_values(["chrom", "chromStart"]).reset_index(drop=True)

    edges = []  # Collect (gene_name, peak_name, weight, sign) tuples

    # Process each chromosome separately
    for chrom in genes["chrom"].unique():
        g_chrom = genes[genes["chrom"] == chrom].reset_index(drop=True)
        p_chrom = peaks[peaks["chrom"] == chrom].reset_index(drop=True)

        if len(g_chrom) == 0 or len(p_chrom) == 0:
            continue

        # Convert to numpy arrays for speed
        g_starts = g_chrom["chromStart"].values
        g_ends = g_chrom["chromEnd"].values
        g_names = g_chrom["gene_name"].values

        p_starts = p_chrom["chromStart"].values
        p_ends = p_chrom["chromEnd"].values
        p_names = p_chrom["peak_name"].values

        n_peaks = len(p_chrom)

        # Two-pointer sweep-line algorithm
        window_start = 0  # First peak that might still be in range
        window_end = 0  # First peak not yet added to window

        for gi in range(len(g_chrom)):
            gene_start = g_starts[gi]
            gene_end = g_ends[gi]
            gene_name = g_names[gi]

            # Move window_start forward: remove peaks too far behind
            # (peak_end < gene_start - extend_range)
            while window_start < n_peaks and p_ends[window_start] < gene_start - extend_range:
                window_start += 1

            # Move window_end forward: add peaks that start within range
            # (peak_start <= gene_end + extend_range)
            while window_end < n_peaks and p_starts[window_end] <= gene_end + extend_range:
                window_end += 1

            # Check all peaks in [window_start, window_end)
            for pi in range(window_start, window_end):
                d = _interval_distance_scalar(gene_start, gene_end, p_starts[pi], p_ends[pi])
                if abs(d) <= extend_range:
                    w = _dist_power_decay(d)
                    edges.append((gene_name, p_names[pi], w, 1.0))

    if not edges:
        logger.warning("No gene-peak pairs found within extend_range.")
        return pd.DataFrame(columns=[rna_key, atac_key, "weight", "sign"])

    result = pd.DataFrame(edges, columns=[rna_key, atac_key, "weight", "sign"])

    # Calculate peaks-per-gene statistics
    peaks_per_gene = result.groupby(rna_key).size()
    logger.info(
        f"Constructed peak-gene mapping with {len(result)} pairs (gene_region='{gene_region}'). "
        f"Peaks per gene: min={peaks_per_gene.min()}, max={peaks_per_gene.max()}, "
        f"mean={peaks_per_gene.mean():.1f}, median={peaks_per_gene.median():.1f}."
    )
    return result


def propagate_highly_variable(
    rna_adata: AnnData,
    atac_adata: AnnData,
    mapping_df: pd.DataFrame,
    rna_key: str = "rna",
    atac_key: str = "atac",
    subset: bool = True,
) -> None:
    """Propagate highly_variable status from RNA genes to linked ATAC peaks.

    Marks peaks in atac_adata as highly variable if they are linked to at least
    one highly variable gene in rna_adata, based on the peak-gene mapping.
    Optionally subsets both adatas to only highly variable features.

    Parameters
    ----------
    rna_adata
        RNA AnnData with 'highly_variable' column in var.
    atac_adata
        ATAC AnnData to annotate (modified in place).
    mapping_df
        Gene-peak mapping DataFrame from construct_peak_gene_mapping().
    rna_key
        Column name for RNA features in mapping_df.
    atac_key
        Column name for ATAC features in mapping_df.
    subset
        If True (default), subset both adatas in place to only contain
        highly variable genes/peaks.

    Notes
    -----
    Modifies atac_adata.var in place by adding 'highly_variable' column.
    A peak is marked as highly variable if it is linked to at least one
    highly variable gene. If subset=True, both adatas are subsetted in place
    to only contain highly variable features.
    """
    if "highly_variable" not in rna_adata.var.columns:
        logger.warning("No 'highly_variable' column in rna_adata.var. Skipping.")
        return

    # Get set of highly variable gene names
    hvg_mask = rna_adata.var["highly_variable"].astype(bool)
    hvg_names = set(rna_adata.var_names[hvg_mask])

    # Find peaks linked to HVGs
    hvg_peaks = mapping_df.loc[mapping_df[rna_key].isin(hvg_names), atac_key].unique()

    # Mark peaks in atac_adata
    atac_adata.var["highly_variable"] = atac_adata.var_names.isin(hvg_peaks)

    n_hvg = len(hvg_names)
    n_hvp = atac_adata.var["highly_variable"].sum()
    logger.info(f"Propagated highly_variable: {n_hvg} HVGs -> {n_hvp} peaks marked.")

    # Subset both adatas to highly variable features in place
    if subset:
        # Subset RNA to HVGs
        rna_hvg_mask = rna_adata.var["highly_variable"].values
        rna_adata._inplace_subset_var(rna_hvg_mask)

        # Subset ATAC to HV peaks
        atac_hvp_mask = atac_adata.var["highly_variable"].values
        atac_adata._inplace_subset_var(atac_hvp_mask)

        # Subset mapping_df in place to only keep valid mappings
        valid_rna = set(rna_adata.var_names)
        valid_atac = set(atac_adata.var_names)
        invalid_mask = ~(
            mapping_df[rna_key].isin(valid_rna) & mapping_df[atac_key].isin(valid_atac)
        )
        mapping_df.drop(mapping_df.index[invalid_mask], inplace=True)

        logger.info(
            f"Subsetted adatas in place: RNA ({n_hvg} HVGs), ATAC ({n_hvp} HV peaks). "
            f"Mapping reduced to {len(mapping_df)} pairs."
        )


@dependencies("torch_geometric")
def _construct_guidance_graph(
    adatas: dict[str, AnnData],
    mapping_df: pd.DataFrame | None,
    weight: float = 1.0,
    sign: float = 1.0,
) -> Data:
    """Construct a guidance graph linking features across modalities.

    Creates a bipartite graph where nodes represent features from each modality
    and edges connect corresponding features based on the mapping DataFrame or
    shared feature names.

    Parameters
    ----------
    adatas
        Dictionary mapping modality names to AnnData objects.
    mapping_df
        DataFrame with columns matching modality names, containing feature
        mappings. If None, uses shared feature names.
    weight
        Edge weight for cross-modality connections.
    sign
        Edge sign for cross-modality connections.

    Returns
    -------
    PyTorch Geometric Data object with node features, edge indices,
    edge weights, edge signs, and modality index tensors.

    Raises
    ------
    ValueError
        If not exactly two modalities are provided or no overlapping features
        exist when mapping_df is None.
    """
    from torch_geometric.data import Data

    if len(adatas) != 2:
        raise ValueError("Exactly two modalities are required.")
    input_names = list(adatas.keys())
    adata1, adata2 = adatas[input_names[0]], adatas[input_names[1]]

    if mapping_df is not None:
        features1 = list(adata1.var_names)
        features2 = list(adata2.var_names)
    else:
        shared_features = set(adata1.var_names) & set(adata2.var_names)
        if not shared_features:
            raise ValueError("No overlapping features between the two modalities.")

        features1 = [f"{f}_{input_names[0]}" for f in adata1.var_names]
        features2 = [f"{f}_{input_names[1]}" for f in adata2.var_names]

    all_features = features1 + features2
    feature_to_index = {f: i for i, f in enumerate(all_features)}

    if mapping_df is not None:
        # Vectorized edge construction (much faster than row-by-row)
        feat1_col = mapping_df[input_names[0]].values
        feat2_col = mapping_df[input_names[1]].values

        # P2: Vectorized pandas lookup (3-5x faster than list comprehension for 1M+ rows)
        idx_series = pd.Series(feature_to_index)
        idx1 = idx_series.reindex(feat1_col).values.astype(np.int64)
        idx2 = idx_series.reindex(feat2_col).values.astype(np.int64)

        # Get weights and signs from mapping_df if present, else use defaults
        if "weight" in mapping_df.columns:
            weights = mapping_df["weight"].values
        else:
            weights = np.full(len(mapping_df), weight, dtype=np.float32)

        if "sign" in mapping_df.columns:
            signs = mapping_df["sign"].values
        else:
            signs = np.full(len(mapping_df), sign, dtype=np.float32)

        # Build bidirectional edges: (i->j) and (j->i)
        edge_index = np.column_stack([np.concatenate([idx1, idx2]), np.concatenate([idx2, idx1])])
        edge_weight = np.concatenate([weights, weights])
        edge_sign = np.concatenate([signs, signs])

    else:
        shared_list = list(shared_features)
        n_shared = len(shared_list)

        idx1 = np.array(
            [feature_to_index[f"{f}_{input_names[0]}"] for f in shared_list], dtype=np.int64
        )
        idx2 = np.array(
            [feature_to_index[f"{f}_{input_names[1]}"] for f in shared_list], dtype=np.int64
        )

        edge_index = np.column_stack([np.concatenate([idx1, idx2]), np.concatenate([idx2, idx1])])
        edge_weight = np.full(2 * n_shared, weight, dtype=np.float32)
        edge_sign = np.full(2 * n_shared, sign, dtype=np.float32)

    # Add self-loops for all features
    n_features = len(all_features)
    self_loops = np.column_stack([np.arange(n_features), np.arange(n_features)])
    edge_index = np.vstack([edge_index, self_loops])
    edge_weight = np.concatenate([edge_weight, np.full(n_features, weight, dtype=np.float32)])
    edge_sign = np.concatenate([edge_sign, np.full(n_features, sign, dtype=np.float32)])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    edge_sign = torch.tensor(edge_sign, dtype=torch.float)

    # P2: Vectorized index lookup using pandas (faster than list comprehension)
    idx_series = pd.Series(feature_to_index)
    indices1 = torch.tensor(idx_series.reindex(features1).values, dtype=torch.long)
    indices2 = torch.tensor(idx_series.reindex(features2).values, dtype=torch.long)

    logger.info(
        f"Constructed guidance graph with {n_features} nodes and {edge_index.shape[1]} edges "
        f"({len(features1)} {input_names[0]} features, "
        f"{len(features2)} {input_names[1]} features)."
    )

    # Note: x (identity matrix) removed - not used by GraphEncoder which has its own vrepr
    return Data(
        edge_index=edge_index,
        edge_weight=edge_weight,
        edge_sign=edge_sign,
        num_nodes=n_features,
        **{f"{input_names[0]}_indices": indices1, f"{input_names[1]}_indices": indices2},
    )


def _check_guidance_graph_consistency(graph: Data, adatas: dict[str, AnnData]):
    """Validate guidance graph structure and consistency with AnnData objects.

    Performs several consistency checks on the guidance graph:
    1. Node count matches total number of features across modalities
    2. Required edge attributes (edge_weight, edge_sign) are present
    3. Self-loops exist for all nodes
    4. Graph is symmetric (undirected)

    Parameters
    ----------
    graph
        PyTorch Geometric Data object representing the guidance graph.
    adatas
        Dictionary mapping modality names to AnnData objects.

    Raises
    ------
    ValueError
        If any consistency check fails.
    """
    n_expected = sum(adata.shape[1] for adata in adatas.values())

    # 1. Check variable coverage via counts
    if graph.num_nodes != n_expected:
        raise ValueError(
            f"Graph node count {graph.num_nodes} does not match expected {n_expected}."
        )

    # 2. Check edge attributes
    for attr in ["edge_weight", "edge_sign"]:
        if not hasattr(graph, attr):
            raise ValueError(f"Graph missing required edge attribute: {attr}")
        if getattr(graph, attr).shape[0] != graph.edge_index.shape[1]:
            raise ValueError(f"Edge attribute {attr} does not match number of edges.")

    # 3. Check self-loops
    src, tgt = graph.edge_index
    self_loops = src == tgt
    n_self_loops = self_loops.sum().item()
    if n_self_loops < graph.num_nodes:
        raise ValueError("Graph is missing self-loops for some nodes.")

    # 4. Check symmetry (for undirected graphs: For every edge (i, j), check that (j, i) exists)
    # Vectorized check: encode edges as unique integers and use torch.isin
    num_nodes = graph.num_nodes
    edge_ids = src * num_nodes + tgt
    reversed_ids = tgt * num_nodes + src

    has_reverse = torch.isin(edge_ids, reversed_ids)
    if not has_reverse.all():
        idx = (~has_reverse).nonzero(as_tuple=True)[0][0].item()
        raise ValueError(
            f"Graph is not symmetric: edge ({src[idx].item()}, "
            f"{tgt[idx].item()}) has no counterpart."
        )

    # If all checks pass
    logger.info("Guidance graph consistency checks passed.")


def _load_saved_diagvi_files(
    dir_path: str,
    prefix: str | None = None,
    map_location: Literal["cpu", "cuda"] | None = None,
    backup_url: str | None = None,
) -> tuple[dict, dict[str, np.ndarray], dict, dict[str, AnnData | None]]:
    """Loads saved DiagVI model and AnnData files from a directory.

    Parameters
    ----------
    dir_path
        Directory path where the model and AnnData files are stored.
    prefix
        Optional prefix for the file names.
    map_location
        Device mapping for loading the model.
    backup_url
        Optional URL to download the model file if not found locally.

    Returns
    -------
    A tuple containing:
    - attr_dict: Dictionary of model attributes.
    - var_names: Dictionary of variable names for each modality.
    - model_state_dict: State dictionary of the model.
    - adatas: Dictionary of AnnData objects for each modality.

    Raises
    ------
    ValueError
        If model file cannot be loaded.
    """
    file_name_prefix = prefix or ""

    model_file_name = f"{file_name_prefix}model.pt"
    model_path = os.path.join(dir_path, model_file_name)

    try:
        _download(backup_url, dir_path, model_file_name)
        model = torch.load(model_path, map_location=map_location, weights_only=False)
    except FileNotFoundError as exc:
        raise ValueError(f"Failed to load model file at {model_path}. ") from exc

    names = model["names"]

    adatas = {}
    var_names = {}
    for name in names:
        adata_path = os.path.join(dir_path, f"{file_name_prefix}adata_{name}.h5ad")
        if os.path.exists(adata_path):
            adatas[name] = ad.read_h5ad(adata_path)
            var_names[name] = adatas[name].var_names
        else:
            adatas[name] = None

    model_state_dict = model["model_state_dict"]
    attr_dict = model["attr_dict"]

    return (
        attr_dict,
        var_names,
        model_state_dict,
        adatas,
    )


@dependencies("torch_geometric")
def compute_graph_loss(graph: Data, feature_embeddings: torch.Tensor) -> torch.Tensor:
    """Compute graph reconstruction loss using negative sampling.

    Uses structured negative sampling to compute a contrastive loss that
    encourages connected nodes to have similar embeddings and unconnected
    nodes to have dissimilar embeddings.

    Parameters
    ----------
    graph
        PyTorch Geometric Data object with edge_index.
    feature_embeddings
        Tensor of shape (n_features, embedding_dim) containing feature embeddings.

    Returns
    -------
    Scalar tensor containing the graph reconstruction loss.
    """
    import torch_geometric

    edge_index = graph.edge_index
    edge_index_neg = torch_geometric.utils.structured_negative_sampling(edge_index)

    # Use tensor indexing directly (avoid GPU→CPU transfer)
    pos_i = edge_index_neg[0]
    pos_j = edge_index_neg[1]
    neg_j = edge_index_neg[2]

    vi = feature_embeddings[pos_i]
    vj = feature_embeddings[pos_j]
    vj_neg = feature_embeddings[neg_j]

    pos_logits = (vi * vj).sum(dim=1)
    pos_loss = F.logsigmoid(pos_logits).mean()

    neg_logits = (vi * vj_neg).sum(dim=1)
    neg_loss = F.logsigmoid(-neg_logits).mean()

    total_loss = -(pos_loss + neg_loss) / 2

    return total_loss


def kl_divergence_graph(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Computes the KL divergence for graph latent variables.

    Parameters
    ----------
    mu
        Mean tensor of the latent variables.
    logvar
        Log-variance tensor of the latent variables.

    Returns
    -------
    The mean KL divergence as a tensor.
    """
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_mean = kl.mean()
    return kl_mean
