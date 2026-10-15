"""Checkpoint lifecycle for an external orchestrator.

The orchestrator captures and restores the whole process from outside. SGLang
only gets itself safe to capture (prepare), reports that it is (status), and
goes back to serving once told the restore is done (resume).
"""

import asyncio
import logging
from enum import Enum
from typing import Optional

from sglang.srt.managers.io_struct import (
    ContinueGenerationReqInput,
    PauseGenerationReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
)
from sglang.srt.runtime_context import get_exec, get_model, get_parallel, get_serving

logger = logging.getLogger(__name__)


class CheckpointState(str, Enum):
    SERVING = "serving"
    PREPARING = "preparing"
    PREPARED = "prepared"
    RESUMING = "resuming"
    FAILED = "failed"


class CheckpointConflict(Exception):
    pass


def validate_checkpoint_args() -> None:
    features = get_exec().features
    missing = [
        flag
        for flag, enabled in (
            ("--enable-memory-saver", features.enable_memory_saver),
            ("--enable-weights-cpu-backup", features.enable_weights_cpu_backup),
        )
        if not enabled
    ]
    if missing:
        raise ValueError(f"--enable-checkpoint requires {' and '.join(missing)}")
    if get_model().weight_cache_mode != "off":
        raise ValueError("--enable-checkpoint requires --weight-cache-mode off")
    parallel = get_parallel()
    if parallel.tp_size * parallel.pp_size * parallel.num_dp_ranks != 1:
        raise ValueError("--enable-checkpoint supports only TP=1, PP=1 and DP=1")
    if get_serving().tokenizer_worker_num != 1:
        raise ValueError("--enable-checkpoint supports only one tokenizer worker")


class CheckpointLifecycle:
    def __init__(self, tokenizer_manager):
        self.tokenizer_manager = tokenizer_manager
        self.state = CheckpointState.SERVING
        self.error: Optional[str] = None
        self._task: Optional[asyncio.Task] = None

    @property
    def admitting(self) -> bool:
        return self.state == CheckpointState.SERVING

    @property
    def healthy(self) -> bool:
        return self.state != CheckpointState.FAILED

    def status(self) -> dict:
        return {"state": self.state.value, "error": self.error}

    def prepare(self) -> None:
        if self.state in (CheckpointState.PREPARING, CheckpointState.PREPARED):
            return
        if self.state != CheckpointState.SERVING:
            raise CheckpointConflict(f"cannot prepare while {self.state.value}")
        self._start(CheckpointState.PREPARING, self._prepare, CheckpointState.PREPARED)

    def resume(self) -> None:
        if self.state in (CheckpointState.RESUMING, CheckpointState.SERVING):
            return
        if self.state != CheckpointState.PREPARED:
            raise CheckpointConflict(f"cannot resume while {self.state.value}")
        self._start(CheckpointState.RESUMING, self._resume, CheckpointState.SERVING)

    def _start(self, running: CheckpointState, step, done: CheckpointState) -> None:
        self.state = running
        self.error = None
        self._task = asyncio.create_task(self._run(step, done))

    async def _run(self, step, done: CheckpointState) -> None:
        try:
            await step()
        except Exception as e:
            logger.exception("Checkpoint %s failed", self.state.value)
            self.error = f"{type(e).__name__}: {e}"
            self.state = CheckpointState.FAILED
            return
        logger.info("Checkpoint %s", done.value)
        self.state = done

    async def _prepare(self) -> None:
        tm = self.tokenizer_manager
        await tm.pause_generation(PauseGenerationReqInput())
        await tm.release_memory_occupation(ReleaseMemoryOccupationReqInput(), None)

    async def _resume(self) -> None:
        tm = self.tokenizer_manager
        await tm.resume_memory_occupation(ResumeMemoryOccupationReqInput(), None)
        await tm.continue_generation(ContinueGenerationReqInput())
