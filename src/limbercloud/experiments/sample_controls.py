"""
Sample-count controls for spectra runners.

Every run evaluates the fiducial, sample 0. ``sample_count`` is how many
further cosmologies, IDs 1..N, are evaluated after it. The default is 0, so a
launch without ``--sample-count`` is the fiducial alone. The production
campaign passes 1000 explicitly.
"""

from __future__ import annotations

import argparse
from typing import Sequence

import numpy


def resolve_sample_count(sample_count):
    """
    Resolve how many cosmologies to evaluate after the fiducial.

    Args:
        sample_count (int | None): Requested extra rows. ``None`` and ``0``
            both mean the fiducial alone, so an accidental launch does not
            evaluate the 1,000-row campaign.

    Returns:
        int: Extra sample count (``>= 0``).

    Raises:
        ValueError: When the count is negative.
    """

    if sample_count is None:
        return 0

    resolved = int(sample_count)
    if resolved < 0:
        raise ValueError(f"--sample-count must be >= 0; got {resolved}")
    return resolved


def build_checkpoint_counts(sample_count):
    """Return checkpoints at each 100 samples, including the final count.

    Tiny pilots have one checkpoint at their actual count. Zero requests only
    the fiducial and therefore has no sampled checkpoints.
    """
    sample_count = resolve_sample_count(sample_count)
    if sample_count == 0:
        return numpy.zeros(0, dtype=numpy.int32)
    checkpoints = numpy.arange(100, sample_count + 1, 100, dtype=numpy.int32)
    if checkpoints.size == 0 or checkpoints[-1] != sample_count:
        checkpoints = numpy.append(checkpoints, numpy.int32(sample_count))
    return checkpoints


def add_sample_control_arguments(parser):
    """
    Attach ``--sample-count`` to an argument parser.

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
            "Number of cosmologies after the fiducial. Defaults to 0, which "
            "evaluates sample 0 only. The production campaign must pass 1000."
        ),
    )
    return parser


def resolve_from_namespace(namespace):
    """
    Resolve sample count from a parsed argparse namespace.

    Args:
        namespace (argparse.Namespace): Parsed arguments containing
            ``sample_count``.

    Returns:
        int: Number of cosmologies after the fiducial.
    """
    return resolve_sample_count(getattr(namespace, "sample_count", None))


def add_evaluation_arguments(parser):
    """Attach shared evaluation flags, including the sample-count controls.

    Existing ``--tag``, ``--label`` and ``--folder`` arguments stay on the
    individual drivers. ``--sample-count`` is the number of cosmologies after
    the fiducial. The fiducial always runs.

    Args:
        parser (argparse.ArgumentParser): Parser to extend in place.

    Returns:
        argparse.ArgumentParser: The same parser for chaining.
    """

    add_sample_control_arguments(parser)
    parser.add_argument(
        "--sample-table",
        default=None,
        help=(
            "Directory containing the canonical Cosmologies.npz table. "
            "Sample 0 is the fiducial."
        ),
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
