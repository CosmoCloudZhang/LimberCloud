"""
Sample-count controls for spectra runners.

``sample_count`` is the number of non-fiducial sampled rows. ``fiducial_only``
implies zero sampled rows. The production campaign must request 1,000 sampled
rows explicitly; safe defaults do not launch that campaign.
"""

from __future__ import annotations

import argparse
from typing import Sequence

import numpy


def resolve_sample_count(sample_count, fiducial_only):
    """
    Resolve the number of non-fiducial samples to evaluate.

    Args:
        sample_count (int | None): Requested non-fiducial sample rows. When
            ``None`` and ``fiducial_only`` is false, defaults to ``0`` so that
            accidental launches do not evaluate the 1,000-row campaign.
        fiducial_only (bool): When true, forces zero sampled rows.

    Returns:
        int: Non-fiducial sample count (``>= 0``).

    Raises:
        ValueError: If the selection is inconsistent or negative.
    """
    if fiducial_only:
        if sample_count is not None and int(sample_count) != 0:
            raise ValueError(
                "--fiducial-only requires zero sampled rows; "
                f"got --sample-count={sample_count}"
            )
        return 0

    if sample_count is None:
        return 0

    resolved = int(sample_count)
    if resolved < 0:
        raise ValueError(f"--sample-count must be >= 0; got {resolved}")
    return resolved


def build_checkpoint_counts(sample_count, count_size=10):
    """
    Build cumulative checkpoint sample counts for timing products.

    Args:
        sample_count (int): Number of non-fiducial samples to evaluate.
        count_size (int): Preferred number of cumulative checkpoints. Reduced
            automatically when ``sample_count`` is smaller.

    Returns:
        numpy.ndarray: Strictly positive integer checkpoint counts with dtype
        ``int32``, empty when ``sample_count`` is zero.
    """
    sample_count = int(sample_count)
    if sample_count <= 0:
        return numpy.zeros(0, dtype=numpy.int32)

    count_size = max(1, min(int(count_size), sample_count))
    if count_size == 1:
        return numpy.asarray([sample_count], dtype=numpy.int32)

    if sample_count < 100:
        return numpy.unique(
            numpy.linspace(1, sample_count, count_size, dtype=numpy.int32)
        )

    count1 = min(100, sample_count)
    return numpy.unique(
        numpy.linspace(count1, sample_count, count_size, dtype=numpy.int32)
    )


def add_sample_control_arguments(parser):
    """
    Attach ``--sample-count`` and ``--fiducial-only`` to an argument parser.

    Args:
        parser (argparse.ArgumentParser): Parser to extend in place.

    Returns:
        argparse.ArgumentParser: The same parser for chaining.
    """
    parser.add_argument(
        "--sample-count",
        type=int,
        default=None,
        help=(
            "Number of non-fiducial sampled cosmologies. Defaults to 0. "
            "The production campaign must pass 1000 explicitly."
        ),
    )
    parser.add_argument(
        "--fiducial-only",
        action="store_true",
        help="Evaluate no sampled rows (implies --sample-count=0).",
    )
    return parser


def resolve_from_namespace(namespace):
    """
    Resolve sample count from a parsed argparse namespace.

    Args:
        namespace (argparse.Namespace): Parsed arguments containing
            ``sample_count`` and ``fiducial_only``.

    Returns:
        int: Non-fiducial sample count.
    """
    return resolve_sample_count(
        getattr(namespace, "sample_count", None),
        bool(getattr(namespace, "fiducial_only", False)),
    )


def add_evaluation_arguments(parser):
    """Attach shared evaluation flags, including the sample-count controls.

    Existing ``--tag``, ``--label``, ``--folder`` and ``--number`` arguments
    stay on the individual drivers. ``--sample-count`` still counts non-fiducial
    rows. ``--fiducial-only`` still implies zero sampled rows.

    Args:
        parser (argparse.ArgumentParser): Parser to extend in place.

    Returns:
        argparse.ArgumentParser: The same parser for chaining.
    """

    add_sample_control_arguments(parser)
    parser.add_argument(
        "--sample-table",
        default=None,
        help="Directory containing the canonical Cosmologies.npz table.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run-ID subdirectory under the family/survey results root.",
    )
    parser.add_argument(
        "--run-config",
        default=None,
        help="Optional versioned evaluation-configuration JSON.",
    )
    parser.add_argument(
        "--include-fiducial",
        action="store_true",
        help="Request sample ID 0 in addition to --sample-count sampled rows.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume by validated sample ID. Does not redraw missing rows.",
    )
    parser.add_argument(
        "--mode",
        choices=("validation", "benchmark"),
        default="benchmark",
        help="validation defaults are applied by the shared evaluator; benchmark keeps timing outputs.",
    )
    return parser


def parse_sample_controls(argv: Sequence[str] | None = None):
    """
    Parse only the sample-control flags from an argument list.

    Args:
        argv (Sequence[str] | None): Argument list; defaults to ``sys.argv[1:]``
            when used through argparse internals. Prefer passing an explicit
            list from tests.

    Returns:
        int: Resolved non-fiducial sample count.
    """
    parser = argparse.ArgumentParser(add_help=False)
    add_sample_control_arguments(parser)
    namespace, _ = parser.parse_known_args(argv)
    return resolve_from_namespace(namespace)
