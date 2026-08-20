"""Centralized legacy and canonical runtime paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from limbercloud.config import RuntimeLayout


_LEGACY_CONFIG_NAMES = {
    "cosmology": "COSMOLOGY.json",
    "survey": "SURVEY.json",
    "number_density": "DENSITY.json",
    "galaxy_bias": "GALAXY.json",
    "magnification_bias": "MAGNIFICATION.json",
    "intrinsic_alignment": "ALIGNMENT.json",
}

_CANONICAL_CONFIG_NAMES = {
    "cosmology": "cosmology.json",
    "survey": "survey.json",
    "number_density": "number_density.json",
    "galaxy_bias": "galaxy_bias.json",
    "magnification_bias": "magnification_bias.json",
    "intrinsic_alignment": "intrinsic_alignment.json",
}


@dataclass(frozen=True)
class ProjectPaths:
    """Resolve LimberCloud runtime inputs and outputs.

    The default remains the legacy NERSC layout during migration. Set
    ``LIMBERCLOUD_LAYOUT=canonical`` after the external data tree has been
    copied and verified.
    """

    root: Path
    layout: RuntimeLayout = RuntimeLayout.LEGACY

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())
        object.__setattr__(self, "layout", RuntimeLayout.parse(self.layout))

    @classmethod
    def from_root(
        cls,
        root: str | os.PathLike[str],
        layout: str | RuntimeLayout | None = None,
    ) -> "ProjectPaths":
        """Create paths using an explicit layout or the environment."""

        selected = layout or os.environ.get("LIMBERCLOUD_LAYOUT", "legacy")
        return cls(root=Path(root), layout=RuntimeLayout.parse(selected))

    @property
    def data(self) -> Path:
        return self.root / ("DATA" if self.layout is RuntimeLayout.LEGACY else "data")

    @property
    def config(self) -> Path:
        return self.root / ("INFO" if self.layout is RuntimeLayout.LEGACY else "config")

    @property
    def plots(self) -> Path:
        return self.root / ("PLOT" if self.layout is RuntimeLayout.LEGACY else "plots")

    @property
    def logs(self) -> Path:
        return self.root / ("LOG" if self.layout is RuntimeLayout.LEGACY else "logs")

    @property
    def results(self) -> Path:
        return self.root if self.layout is RuntimeLayout.LEGACY else self.root / "results"

    def survey_data(self, survey: str) -> Path:
        """Return the data directory for ``Y1`` or ``Y10``."""

        return self.data / _validate_survey(survey)

    def config_file(self, name: str) -> Path:
        """Return a named configuration JSON path."""

        mapping = (
            _LEGACY_CONFIG_NAMES
            if self.layout is RuntimeLayout.LEGACY
            else _CANONICAL_CONFIG_NAMES
        )
        try:
            filename = mapping[name]
        except KeyError as error:
            choices = ", ".join(sorted(mapping))
            raise ValueError(f"Unknown configuration file {name!r}; expected: {choices}") from error
        return self.config / filename

    def spectrum_results(
        self,
        backend: str,
        survey: str,
        device: str | None = None,
    ) -> Path:
        """Return a spectrum-result directory without changing its matrix."""

        backend_name = backend.upper()
        survey_name = _validate_survey(survey)
        if backend_name not in {"CCL", "NUMBA", "JAX"}:
            raise ValueError(f"Unknown backend {backend!r}")

        if backend_name == "JAX":
            if device is None or device.upper() not in {"CPU", "GPU"}:
                raise ValueError("JAX results require device='CPU' or device='GPU'")
            device_name = device.upper()
            if self.layout is RuntimeLayout.LEGACY:
                return self.root / "JAX" / device_name / survey_name
            return self.results / "spectra" / "JAX" / device_name / survey_name

        if device is not None and device.upper() != "CPU":
            raise ValueError(f"{backend_name} only supports the CPU device")
        if self.layout is RuntimeLayout.LEGACY:
            return self.root / "PYTHON" / backend_name / survey_name
        return self.results / "spectra" / backend_name / survey_name

    def covariance_results(self, survey: str) -> Path:
        """Return the covariance directory for a survey."""

        survey_name = _validate_survey(survey)
        if self.layout is RuntimeLayout.LEGACY:
            return self.root / "COVARIANCE" / survey_name
        return self.results / "covariance" / survey_name

    def validation_results(self, survey: str) -> Path:
        """Return spectra written by the validation notebooks."""

        survey_name = _validate_survey(survey)
        if self.layout is RuntimeLayout.LEGACY:
            return self.root / "PYTHON" / "CELL" / survey_name
        return self.results / "validation" / "spectra" / survey_name

    def plot_group(self, group: str, survey: str | None = None) -> Path:
        """Return a named plot directory in either runtime layout."""

        group_name = group.upper() if self.layout is RuntimeLayout.LEGACY else group.lower()
        path = self.plots / group_name
        if survey is not None:
            path /= _validate_survey(survey)
        return path


def _validate_survey(survey: str) -> str:
    survey_name = survey.upper()
    if survey_name not in {"Y1", "Y10"}:
        raise ValueError(f"Unknown survey {survey!r}; expected 'Y1' or 'Y10'")
    return survey_name
