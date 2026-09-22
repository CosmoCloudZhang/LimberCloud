"""Centralized canonical runtime paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_CONFIG_NAMES = {
    "cosmology": "cosmology.json",
    "survey": "survey.json",
    "number_density": "number_density.json",
    "galaxy_bias": "galaxy_bias.json",
    "magnification_bias": "magnification_bias.json",
    "intrinsic_alignment": "intrinsic_alignment.json",
}


@dataclass(frozen=True)
class ProjectPaths:
    """Resolve LimberCloud runtime inputs and outputs."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())

    @classmethod
    def from_root(
        cls,
        root: str | os.PathLike[str],
    ) -> "ProjectPaths":
        """Create paths rooted at the canonical runtime directory."""

        return cls(root=Path(root))

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def config(self) -> Path:
        return self.root / "config"

    @property
    def plots(self) -> Path:
        return self.root / "plots"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def results(self) -> Path:
        return self.root / "results"

    def survey_data(self, survey: str) -> Path:
        """Return the data directory for ``Y1`` or ``Y10``."""

        return self.data / _validate_survey(survey)

    def config_file(self, name: str) -> Path:
        """Return a named configuration JSON path."""

        try:
            filename = _CONFIG_NAMES[name]
        except KeyError as error:
            choices = ", ".join(sorted(_CONFIG_NAMES))
            raise ValueError(f"Unknown configuration file {name!r}; expected: {choices}") from error
        return self.config / filename

    def spectrum_results(
        self,
        backend: str,
        survey: str,
        device: str | None = None,
        *,
        interpolation: str | None = None,
        run_id: str | None = None,
    ) -> Path:
        """Return a spectrum-result directory, optionally inside a run ID.

        The family, device and radial order are validated together by
        :class:`limbercloud.validation.method.MethodIdentity`, so a NUMERIC
        order can never reach a CCL, NUMBA or JAX directory.

        Args:
            backend (str): ``CCL``, ``NUMBA``, ``JAX`` or ``NUMERIC``.
            survey (str): ``Y1`` or ``Y10``.
            device (str | None): ``CPU`` or ``GPU`` for JAX. Other families use CPU.
            interpolation (str | None): NUMERIC order ``linear``, ``quadratic``
                or ``cubic``. Required for NUMERIC and rejected otherwise.
            run_id (str | None): New-run subdirectory. Omitted paths stay at the
                historical family/survey root for legacy readers.

        Returns:
            Path: Family/device/survey directory, plus ``run_id`` when given.
        """

        from limbercloud.validation.method import MethodIdentity

        method = MethodIdentity.create(backend, device, interpolation)
        survey_name = _validate_survey(survey)
        if method.selects_order:
            base = self.results / "spectra" / method.family / method.interpolation / survey_name
        elif method.selects_device:
            base = self.results / "spectra" / method.family / method.device / survey_name
        else:
            base = self.results / "spectra" / method.family / survey_name
        if run_id is None:
            return base
        return base / _validate_run_id(run_id)

    def spectrum_inputs(self, run_id: str) -> Path:
        """Return the shared cosmology-table directory for one run.

        Args:
            run_id (str): Run identifier shared across surveys and backends.

        Returns:
            Path: ``results/spectra/inputs/<run_id>``.
        """

        return self.results / "spectra" / "inputs" / _validate_run_id(run_id)

    def covariance_results(self, survey: str) -> Path:
        """Return the covariance directory for a survey."""

        survey_name = _validate_survey(survey)
        return self.results / "covariance" / survey_name

    def validation_results(self, survey: str) -> Path:
        """Return spectra written by the validation notebooks."""

        survey_name = _validate_survey(survey)
        return self.results / "validation" / "spectra" / survey_name

    def plot_group(self, group: str, survey: str | None = None) -> Path:
        """Return a named plot directory."""

        path = self.plots / group.lower()
        if survey is not None:
            path /= _validate_survey(survey)
        return path


def _validate_survey(survey: str) -> str:
    survey_name = survey.upper()
    if survey_name not in {"Y1", "Y10"}:
        raise ValueError(f"Unknown survey {survey!r}; expected 'Y1' or 'Y10'")
    return survey_name


def _validate_run_id(run_id: str) -> str:
    if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
        raise ValueError(f"Invalid run ID {run_id!r}")
    return run_id
