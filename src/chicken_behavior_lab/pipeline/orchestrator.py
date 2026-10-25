
"""Small, dependency-injected orchestration layer for ChickenBehaviorLab.

Stage adapters are intentionally injected as callables. This keeps orchestration
independent of the concrete APIs of detector, tracker, pose, and model classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from time import perf_counter
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence


LOGGER = logging.getLogger(__name__)
StageProcessor = Callable[["PipelineContext"], Any]


class PipelineConfigurationError(ValueError):
    """Raised when pipeline stages are configured inconsistently."""


class PipelineStageError(RuntimeError):
    """Wraps an exception raised by a pipeline stage while preserving its cause."""

    def __init__(self, stage_name: str) -> None:
        self.stage_name = stage_name
        super().__init__(f"Pipeline stage {stage_name!r} failed.")


@dataclass(frozen=True)
class PipelineContext:
    """Inputs and outputs available to the next stage.

    ``outputs`` contains results from earlier stages, keyed by stage name.
    A processor should return its result instead of mutating this mapping.
    """

    video: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)
    outputs: Mapping[str, Any] = field(default_factory=dict)

    def get_output(self, stage_name: str) -> Any:
        """Return an earlier stage result or raise a clear configuration error."""
        if stage_name not in self.outputs:
            raise PipelineConfigurationError(
                f"Required output {stage_name!r} is not available."
            )
        return self.outputs[stage_name]


@dataclass(frozen=True)
class PipelineStage:
    """One named stage and its explicit dependencies."""

    name: str
    processor: StageProcessor
    requires: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise PipelineConfigurationError("Stage name must be non-empty.")

        if not callable(self.processor):
            raise PipelineConfigurationError(
                f"Processor for stage {self.name!r} must be callable."
            )

        if len(set(self.requires)) != len(self.requires):
            raise PipelineConfigurationError(
                f"Stage {self.name!r} has duplicate dependencies."
            )


@dataclass(frozen=True)
class PipelineResult:
    """Outputs and simple timing information for a completed run."""

    outputs: Mapping[str, Any]
    stage_durations_seconds: Mapping[str, float]

    def get(self, stage_name: str) -> Any:
        """Convenience accessor for a named stage output."""
        return self.outputs[stage_name]


class ChickenBehaviorPipeline:
    """Run configured stages sequentially and retain every stage result.

    Recommended stage names are ``detection``, ``tracking``, ``pose``,
    ``skeleton``, ``behavior``, ``events``, and ``analytics``.
    These names are conventions, not hard-coded requirements.
    """

    def __init__(self, stages: Sequence[PipelineStage]) -> None:
        if not stages:
            raise PipelineConfigurationError(
                "At least one pipeline stage must be configured."
            )

        names = [stage.name for stage in stages]

        if len(names) != len(set(names)):
            raise PipelineConfigurationError(
                "Pipeline stage names must be unique."
            )

        available: set[str] = set()

        for stage in stages:
            missing = set(stage.requires) - available

            if missing:
                raise PipelineConfigurationError(
                    f"Stage {stage.name!r} requires outputs that do not "
                    f"come from earlier stages: {sorted(missing)}."
                )

            available.add(stage.name)

        self._stages = tuple(stages)

    @property
    def stage_names(self) -> tuple[str, ...]:
        """Return the configured stage names in execution order."""
        return tuple(stage.name for stage in self._stages)

    def run(
        self,
        video: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> PipelineResult:
        """Execute all stages and return results and per-stage wall time.

        Any stage failure stops the run and raises ``PipelineStageError``.
        The original exception is preserved as ``__cause__``.
        """
        stage_outputs: dict[str, Any] = {}
        timings: dict[str, float] = {}

        context = PipelineContext(
            video=video,
            metadata=MappingProxyType(dict(metadata or {})),
            outputs=MappingProxyType(stage_outputs.copy()),
        )

        for stage in self._stages:
            missing = set(stage.requires) - set(stage_outputs)

            if missing:
                raise PipelineConfigurationError(
                    f"Stage {stage.name!r} is missing outputs: "
                    f"{sorted(missing)}."
                )

            LOGGER.debug("Starting pipeline stage: %s", stage.name)
            started = perf_counter()

            try:
                output = stage.processor(context)

            except Exception as exc:
                elapsed = perf_counter() - started
                timings[stage.name] = elapsed

                LOGGER.exception(
                    "Pipeline stage %s failed after %.4f seconds",
                    stage.name,
                    elapsed,
                )

                error = PipelineStageError(stage.name)
                raise error from exc

            timings[stage.name] = perf_counter() - started
            stage_outputs[stage.name] = output

            context = PipelineContext(
                video=video,
                metadata=context.metadata,
                outputs=MappingProxyType(stage_outputs.copy()),
            )

            LOGGER.debug(
                "Finished pipeline stage: %s (%.4f seconds)",
                stage.name,
                timings[stage.name],
            )

        return PipelineResult(
            outputs=MappingProxyType(stage_outputs.copy()),
            stage_durations_seconds=MappingProxyType(timings.copy()),
        )
