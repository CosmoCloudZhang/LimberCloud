"""Guards that run before a spectra driver loads data or creates output.

Every launch evaluates the fiducial cosmology, sample 0 of the saved table.
``--sample-count`` adds that many further rows, IDs 1..N. Zero means the
fiducial alone. Timing files are written in the family/survey directory and a
rerun replaces the products it evaluates. A fiducial-only launch leaves
previous Cosmology timings intact. Older names that still contain ``_128`` are left
in place because the new names do not match them.
"""

from __future__ import annotations

from pathlib import Path


class RunControlError(ValueError):
    """Raised when a run has no cosmology table or a negative sample count."""


def require_effective_work(*, sample_count: int, sample_table=None) -> int:
    """Require a saved cosmology table and a non-negative sample count.

    Args:
        sample_count (int): Number of cosmologies after the fiducial.
        sample_table: ``--sample-table`` value. Sample 0 of that table is the
            fiducial, so the table is required even when the count is zero.

    Returns:
        int: The validated extra-sample count.

    Raises:
        RunControlError: When the count is negative or the table is missing.
    """

    count = int(sample_count)
    if count < 0:
        raise RunControlError(f"--sample-count must be >= 0; got {count}")
    if not sample_table:
        raise RunControlError(
            "Pass --sample-table pointing at the canonical Cosmologies.npz "
            "directory. The fiducial is sample 0 of that table, and "
            "--sample-count adds samples 1..N."
        )
    return count


def prepare_result_directory(directory) -> Path:
    """Create the family/survey directory used for timing products.

    Args:
        directory: ``results/spectra/<family>/<survey>`` or the JAX device
            equivalent.

    Returns:
        Path: The directory. An existing ``Time_*.txt`` is left for the driver
        to replace when it saves. This does not delete older ``*_128*`` files.
    """

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path
