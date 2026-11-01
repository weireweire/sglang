"""Vocab-parallel DSpark verify (SGLANG_OPT_DSV41_DSPARK_VOCAB_PARALLEL_VERIFY).

The target-verify logits stay the LM head's vocab shard. The static verify graph's
sampled-accept branch rejects on the shards (vocab_parallel_rejection.py); its greedy
branch, and any step whose accept or bookkeeping needs the whole row, gathers them
first, so those steps match a gathered verify bit for bit. The draft logits stay
replicated and fp32.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.kernels.ops.speculative.dspark.blockwise_rejection import RejectionNoise
from sglang.kernels.ops.speculative.dspark.vocab_parallel_rejection import (
    SimulatedBonusArgs,
    VocabParallelExchange,
    make_vocab_parallel_exchange,
    vocab_parallel_rejection,
)
from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessor

logger = logging.getLogger(__name__)


def vocab_parallel_verify_supported(*, tp_size: int, vocab_size: int) -> bool:
    """Static checks; the debug observers that read the whole verify row are off."""
    return (
        envs.SGLANG_OPT_DSV41_DSPARK_VOCAB_PARALLEL_VERIFY.get()
        and tp_size > 1
        and vocab_size % tp_size == 0
        and not envs.SGLANG_DSPARK_DEBUG_CONFIDENCE_METRICS.get()
        and not envs.SGLANG_DSPARK_STS_COLLECT_PATH.get()
    )


class VocabParallelVerify:
    """The target logits processor that defers the gather, and the exchange buffer
    of the sharded rejection (built at the first eager call, a warmup)."""

    def __init__(
        self,
        *,
        logits_processor: LogitsProcessor,
        verify_num_draft_tokens: int,
        max_bs: int,
    ) -> None:
        self._logits_processor = logits_processor
        self._stride = int(verify_num_draft_tokens)
        self._max_bs = int(max_bs)
        self._exchange: Optional[VocabParallelExchange] = None
        logits_processor.vocab_parallel_verify = True

    def is_shard(self, logits: torch.Tensor) -> bool:
        return self._logits_processor.is_deferred_shard(logits)

    def gather(self, shard: torch.Tensor) -> torch.Tensor:
        """The fp32 whole-vocab logits the gathered verify returns."""
        return self._logits_processor.finish_deferred_gather(shard)

    def reject(
        self,
        *,
        candidates: torch.Tensor,
        shard: torch.Tensor,
        draft_logits: Optional[torch.Tensor],
        target_temperatures: torch.Tensor,
        draft_temperatures: torch.Tensor,
        noise: RejectionNoise,
        sim: Optional[SimulatedBonusArgs],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return vocab_parallel_rejection(
            exchange=self._ensure_exchange(shard.device),
            candidates=candidates,
            shard_logits=shard,
            softcap=self._logits_processor.final_logit_softcapping or None,
            draft_logits=draft_logits,
            target_temperatures=target_temperatures,
            draft_temperatures=draft_temperatures,
            noise=noise,
            sim=sim,
        )

    def _ensure_exchange(self, device) -> VocabParallelExchange:
        if self._exchange is not None:
            return self._exchange
        assert not torch.cuda.is_current_stream_capturing(), (
            "the vocab-parallel exchange is built at a warmup, before capture"
        )
        # Lazy import avoids a module-load dependency on the distributed facade.
        from sglang.srt.distributed.parallel_state import get_tp_group

        group = get_tp_group()
        self._exchange = make_vocab_parallel_exchange(
            group=group.device_group,
            rank=group.rank_in_group,
            world=group.world_size,
            vocab=self._logits_processor.vocab_size,
            stride=self._stride,
            max_bs=self._max_bs,
            device=device,
        )
        if group.rank_in_group == 0:
            logger.info(
                "DSpark vocab-parallel verify: sampled accepts reject on %d vocab "
                "shards.",
                group.world_size,
            )
        return self._exchange
