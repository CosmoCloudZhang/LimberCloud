"""Write one canonical cosmology table.

The fiducial row is sample 0 and consumes no random draw. Sampled IDs are
1..``--sampled-count``. The paper campaign must pass ``--sampled-count 1000``
explicitly; the default writes the fiducial row only.
"""

from __future__ import annotations

import argparse
import json

from limbercloud import ProjectPaths
from limbercloud.validation.samples import (
    generate_cosmology_table,
    save_cosmology_table,
)


def main(folder, run_id, seed, sampled_count):
    """Generate and publish ``Cosmologies.npz`` plus its manifest.

    Args:
        folder (str): Runtime root that contains ``config/cosmology.json``.
        run_id (str): Shared run identifier.
        seed (int): Seed for ``numpy.random.default_rng``.
        sampled_count (int): Non-fiducial rows. Zero stores the fiducial only.

    Returns:
        str: Content hash of the published table.
    """

    paths = ProjectPaths.from_root(folder)
    with paths.config_file("cosmology").open() as handle:
        fiducial = json.load(handle)
    table = generate_cosmology_table(
        fiducial, seed=int(seed), sampled_count=int(sampled_count)
    )
    destination = paths.spectrum_inputs(run_id)
    save_cosmology_table(destination, table)
    print(f"Wrote {destination} content_hash={table.content_hash}")
    return table.content_hash


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canonical cosmology table")
    parser.add_argument(
        "--folder", required=True, help="Runtime root containing config/cosmology.json"
    )
    parser.add_argument("--run-id", required=True, help="Shared sample-table run ID")
    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        help="default_rng seed. There is no hidden campaign seed.",
    )
    parser.add_argument(
        "--sampled-count",
        type=int,
        default=0,
        help="Non-fiducial rows. The paper campaign must pass 1000 explicitly.",
    )
    args = parser.parse_args()
    main(args.folder, args.run_id, args.seed, args.sampled_count)
