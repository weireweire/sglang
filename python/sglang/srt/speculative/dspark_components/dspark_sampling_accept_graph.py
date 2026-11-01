"""CUDA graphs of the temperature-sampling verify accept of static DSpark decode steps
(SGLANG_OPT_DSV41_SAMPLING_ACCEPT_GRAPH).

The static verify graph folds only the greedy accept (DsparkVerifyEpilogue). On an
all-sampled step the rest ran eagerly after the replay: the draft and target softmax,
the rejection kernel, the bonus gather, the accept-site TP broadcast, the engram
history commit, the future relay and the target-hidden KV commit. This module captures
that region once per verify-graph batch size and replays it right after the verify
graph.

The region reads the verify graph's static inputs and outputs (``static_io``) and the
draft sampler's static buffers, which hold this step's values after both replays, so a
replay stages nothing. Its accept outputs land in the epilogue's static accept buffers,
like the greedy fold. Per key the first step runs eagerly (it compiles kernels and warms
allocations), the second captures, later steps replay. The uniform draws come from
torch's default generator, whose graph-safe offsets make a replay draw what the eager
accept draws at the same generator state; with block-wise rejection they come from the
worker's one RejectionNoise, whose device step counter a replay advances like an eager
call. The DSPARK_ACCEPT_SAMPLE broadcast stays in
the region (pynccl inside the graph), so TP ranks agree as in the eager path. With the
deferred context-KV commit, the verify graph already joined this step's projection, so
the region stages the stores from it.

The caller routes only all-sampled steps without top-k / top-p, logit adjustments or
grammar; everything else keeps the eager accept. Under SGLANG_SIMULATE_ACC_LEN the region
draws the bonus at the simulated row from the buffers the caller stages with
begin_sim_bonus.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Hashable
from functools import partial
from typing import Optional

import msgspec
import torch

from sglang.kernels.ops.speculative.dspark.blockwise_rejection import RejectionNoise
from sglang.kernels.ops.speculative.dspark.sampled_accept_tail import (
    sampled_accept_tail,
)
from sglang.kernels.ops.speculative.dspark.vocab_parallel_rejection import (
    SimulatedBonusArgs,
)
from sglang.srt.distributed.parallel_state import graph_capture
from sglang.srt.environ import envs
from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockResult
from sglang.srt.speculative.dspark_components.dspark_verify import (
    AcceptOuts,
    DsparkVerifyEpilogue,
    StaticVerifyIO,
    accept_draft_tokens,
    build_accept_outs,
)
from sglang.srt.speculative.spec_tp_sync import SpecTpSync, SpecTpSyncSite

logger = logging.getLogger(__name__)


class GraphSamplingInfo(msgspec.Struct, frozen=True):
    """The SamplingBatchInfo fields the sampling accept reads, for an all-sampled
    batch without top-k / top-p."""

    temperatures: torch.Tensor
    is_all_greedy: bool = False
    is_any_greedy: bool = False
    need_top_k_sampling: bool = False
    need_top_p_sampling: bool = False


def sampling_accept_commit_lens(
    io: StaticVerifyIO,
    *,
    verify_lens: torch.Tensor,
    gate: torch.Tensor,
    gated: bool,
    draft_sampler,
    epilogue: DsparkVerifyEpilogue,
    tp_sync: SpecTpSync,
    block_verification: bool,
    rejection_noise: Optional[RejectionNoise],
) -> Optional[torch.Tensor]:
    """The eager sampling accept and finalize into the epilogue's static buffers;
    returns the gated commit lengths when ``gated``."""
    bs, stride = io.candidates.shape
    params = draft_sampler.params
    draft_tokens = io.candidates[:, 1:]
    draft_block = DraftBlockResult(
        draft_tokens=draft_tokens,
        corrected_logits=(
            None if draft_sampler.greedy_draft else draft_sampler.step_logits[:bs]
        ),
        greedy_mask=params.greedy_mask[:bs],
        temperatures=params.temperatures[:bs],
        greedy_draft=draft_sampler.greedy_draft,
    )
    vocab_parallel = epilogue.vocab_parallel
    if vocab_parallel is not None and vocab_parallel.is_shard(io.logits):
        assert rejection_noise is not None and not block_verification
        correct_len, bonus, cap_trim_lens, sim_bonus = vocab_parallel.reject(
            candidates=io.candidates,
            shard=io.logits,
            draft_logits=draft_block.corrected_logits,
            target_temperatures=params.temperatures[:bs],
            draft_temperatures=draft_block.temperatures,
            noise=rejection_noise,
            sim=_simulated_bonus_args(epilogue, bs),
        )
        if sim_bonus is not None:
            bonus = sim_bonus
        bonus_logits = None
    else:
        correct_len, bonus, cap_trim_lens = accept_draft_tokens(
            candidates=io.candidates,
            target_logits=io.logits,
            draft_block=draft_block,
            # params.temperatures is the batch's, clamped at 1e-5 (resolve_draft_params).
            sampling_info=GraphSamplingInfo(temperatures=params.temperatures[:bs]),
            draft_input=None,
            gamma=stride - 1,
            verify_num_draft_tokens=stride,
            block_verification=block_verification,
            rejection_noise=rejection_noise,
        )
        bonus_logits = io.logits
        if epilogue.simulated_correct_len is not None:
            bonus = epilogue.sim_bonus(io.logits, bs).to(bonus.dtype).view_as(bonus)
            bonus_logits = None
    if envs.SGLANG_OPT_DSV41_DSPARK_SAMPLED_ACCEPT_TAIL.get():
        return _sampled_accept_tail(
            io,
            correct_len=correct_len,
            bonus=bonus,
            cap_trim_lens=cap_trim_lens,
            verify_lens=verify_lens,
            gate=gate,
            gated=gated,
            epilogue=epilogue,
            tp_sync=tp_sync,
        )
    commit_lens = build_accept_outs(
        correct_len=correct_len,
        bonus=bonus,
        cap_trim_lens=cap_trim_lens,
        prefix_lens=io.seq_lens[:bs],
        draft_tokens=draft_tokens.contiguous(),
        gamma=stride - 1,
        verify_num_draft_tokens=stride,
        tp_sync=tp_sync,
        sync_site=SpecTpSyncSite.DSPARK_ACCEPT_SAMPLE,
        simulated_correct_len=epilogue.simulated_correct_len,
        target_logits=bonus_logits,
        out=epilogue.read_accept(bs),
        block_accept_lens_out=epilogue.read_block_accept_lens(bs),
    ).commit_lens
    if not gated:
        return None
    return torch.minimum(commit_lens, verify_lens.to(torch.int32)) * gate


def _simulated_bonus_args(
    epilogue: DsparkVerifyEpilogue, bs: int
) -> Optional[SimulatedBonusArgs]:
    """DsparkVerifyEpilogue.sim_bonus's staged inputs, or None without a simulation."""
    if epilogue.simulated_correct_len is None:
        return None
    return SimulatedBonusArgs(
        correct_len=epilogue.simulated_correct_len.expand(bs),
        temperatures=epilogue.sim_temperatures_buf[:bs],
        seed=epilogue.sim_seed_buf,
    )


def _sampled_accept_tail(
    io: StaticVerifyIO,
    *,
    correct_len: torch.Tensor,
    bonus: torch.Tensor,
    cap_trim_lens: torch.Tensor,
    verify_lens: torch.Tensor,
    gate: torch.Tensor,
    gated: bool,
    epilogue: DsparkVerifyEpilogue,
    tp_sync: SpecTpSync,
) -> Optional[torch.Tensor]:
    """build_accept_outs into the static buffers, the gated commit lengths and the
    future relay scatter in one launch; the same values bit for bit."""
    bs = io.candidates.shape[0]
    if epilogue.simulated_correct_len is not None:
        correct_len = epilogue.simulated_correct_len.repeat(bs).to(correct_len.dtype)
    site = SpecTpSyncSite.DSPARK_ACCEPT_SAMPLE
    tp_sync.sync(site, correct_len)
    tp_sync.sync(site, bonus)
    tp_sync.sync(site, cap_trim_lens)
    out = epilogue.read_accept(bs)
    gated_commit_lens = torch.empty((bs,), dtype=torch.int32, device=correct_len.device)
    _log_sampled_accept_tail(bs)
    sampled_accept_tail(
        correct_len=correct_len,
        bonus=bonus,
        cap_trim_lens=cap_trim_lens,
        seq_lens=io.seq_lens[:bs],
        candidates=io.candidates,
        verify_lens=verify_lens,
        gate=gate,
        correct_len_out=out.correct_len,
        bonus_out=out.bonus,
        cap_trim_lens_out=out.cap_trim_lens,
        commit_lens_out=out.commit_lens,
        new_seq_lens_out=out.new_seq_lens,
        out_tokens_out=out.out_tokens,
        gated_commit_lens_out=gated_commit_lens,
        block_accept_lens_out=epilogue.read_block_accept_lens(bs),
        relay=epilogue.take_future_relay(io.req_pool_indices, bs),
        commit=epilogue.take_commit_tail(),
    )
    if envs.SGLANG_DEBUG_DSPARK_SAMPLED_ACCEPT_TAIL_CHECK.get() and not (
        torch.cuda.is_current_stream_capturing()
    ):
        _check_sampled_accept_tail(
            io,
            correct_len=correct_len,
            bonus=bonus,
            cap_trim_lens=cap_trim_lens,
            verify_lens=verify_lens,
            gate=gate,
            epilogue=epilogue,
            tp_sync=tp_sync,
            gated_commit_lens=gated_commit_lens,
        )
    return gated_commit_lens if gated else None


def _check_sampled_accept_tail(
    io, *, correct_len, bonus, cap_trim_lens, verify_lens, gate, epilogue, tp_sync,
    gated_commit_lens,
) -> None:
    bs, stride = io.candidates.shape
    got = epilogue.read_accept(bs)
    fields = AcceptOuts.__struct_fields__
    ref = AcceptOuts(*(getattr(got, f).clone() for f in fields))
    block = epilogue.read_block_accept_lens(bs)
    ref_block = None if block is None else block.clone()
    build_accept_outs(
        correct_len=correct_len,
        bonus=bonus,
        cap_trim_lens=cap_trim_lens,
        prefix_lens=io.seq_lens[:bs],
        draft_tokens=io.candidates[:, 1:].contiguous(),
        gamma=stride - 1,
        verify_num_draft_tokens=stride,
        tp_sync=tp_sync,
        sync_site=SpecTpSyncSite.DSPARK_ACCEPT_SAMPLE,
        out=ref,
        block_accept_lens_out=ref_block,
    )
    ref_gated = torch.minimum(ref.commit_lens, verify_lens.to(torch.int32)) * gate
    pairs = [(f, getattr(got, f), getattr(ref, f)) for f in fields]
    pairs.append(("gated", gated_commit_lens, ref_gated))
    if block is not None:
        pairs.append(("block", block, ref_block))
    relay = epilogue.future_relay
    if relay is not None and epilogue._relay_taken:
        rows = io.req_pool_indices[:bs].to(torch.int64)
        pairs.append(("relay_seq", relay.new_seq_lens[rows], ref.new_seq_lens))
        pairs.append(("relay_tok", relay.output_tokens[rows], ref.bonus))
    bad = [f for f, x, y in pairs if x.dtype != y.dtype or not torch.equal(x, y)]
    logger.info(
        "DSpark sampled accept tail check bs=%d: %s",
        bs,
        "bitwise equal (%d outputs)" % len(pairs) if not bad else f"MISMATCH {bad}",
    )


_sampled_accept_tail_logged: set[int] = set()


def _log_sampled_accept_tail(bs: int) -> None:
    if bs in _sampled_accept_tail_logged or not torch.cuda.is_current_stream_capturing():
        return
    _sampled_accept_tail_logged.add(bs)
    logger.info("DSpark sampling-accept graph bs=%d: fused sampled accept tail", bs)


class DsparkSamplingAcceptGraph:
    """Per-batch-size CUDA graphs of the sampling accept region of static verify."""

    # One key per verify-graph batch size is expected; more means addresses move,
    # and the further keys run eagerly. Arbitrary bound.
    MAX_GRAPHS = 32

    def __init__(
        self,
        *,
        epilogue: DsparkVerifyEpilogue,
        resolve_draft_sampler: Callable[[], Optional[object]],
        tp_sync: SpecTpSync,
        block_verification: bool,
        device,
        rejection_noise: Optional[RejectionNoise] = None,
    ) -> None:
        self._epilogue = epilogue
        self._resolve_draft_sampler = resolve_draft_sampler
        self._tp_sync = tp_sync
        self._block_verification = block_verification
        self._rejection_noise = rejection_noise
        # The region always commits; padded rows are gated by verify_lens.
        self._gate = torch.ones((1,), dtype=torch.int32, device=device)
        self._graphs: dict[Hashable, torch.cuda.CUDAGraph] = {}
        self._warmed_up: set[Hashable] = set()
        self._pool = None
        self.disabled = False
        self.num_replays = 0
        # Set by run(): whether this step's region committed the engram history.
        self.committed_engram = False

    def run(self, *, bs: int, target_logits: torch.Tensor) -> Optional[AcceptOuts]:
        """This step's accept through the region, or None when the step must take the
        eager path. Reads only host state every TP rank holds identically."""
        if self.disabled:
            return None
        draft_sampler = self._resolve_draft_sampler()
        if draft_sampler is None or (
            not draft_sampler.greedy_draft and draft_sampler.step_logits is None
        ):
            return None
        epilogue = self._epilogue
        graph_bs = min((b for b in epilogue.static_io if b >= bs), default=None)
        if graph_bs is None:
            return None
        io = epilogue.static_io[graph_bs]
        # The replayed verify graph must be the one whose buffers were recorded.
        if target_logits.data_ptr() != io.logits.data_ptr():
            return None
        key = (
            graph_bs,
            io.logits.data_ptr(),
            draft_sampler.params.temperatures.data_ptr(),
            0
            if draft_sampler.step_logits is None
            else draft_sampler.step_logits.data_ptr(),
        )
        region = partial(
            epilogue.run_static,
            io,
            accept=partial(
                sampling_accept_commit_lens,
                draft_sampler=draft_sampler,
                epilogue=epilogue,
                tp_sync=self._tp_sync,
                block_verification=self._block_verification,
                rejection_noise=self._rejection_noise,
            ),
            gate=self._gate,
            kv_projected=True,
        )
        self.committed_engram = epilogue.engram_history() is not None
        self._run(key, region)
        return epilogue.read_accept(bs)

    def _run(self, key: Hashable, region: Callable[[], object]) -> None:
        graph = self._graphs.get(key)
        if graph is None:
            if key not in self._warmed_up:
                if len(self._warmed_up) < self.MAX_GRAPHS:
                    self._warmed_up.add(key)
                region()
                return
            graph = self._capture(key, region)
            if graph is None:
                return
        graph.replay()
        self.num_replays += 1

    def _capture(
        self, key: Hashable, region: Callable[[], object]
    ) -> Optional[torch.cuda.CUDAGraph]:
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        try:
            # graph_capture() enables pynccl, which the in-graph accept broadcast needs.
            with (
                graph_capture() as ctx,
                torch.cuda.graph(
                    graph,
                    pool=self._pool,
                    stream=ctx.stream,
                    capture_error_mode="thread_local",
                ),
            ):
                region()
        except Exception:
            logger.warning(
                "DSpark sampling-accept graph capture failed for %s; sampling steps "
                "take the eager accept from now on.",
                key,
                exc_info=True,
            )
            self.disabled = True
            # A failed capture recorded the ops without running them.
            region()
            return None
        logger.info("Captured the DSpark sampling-accept region for %s.", key)
        self._graphs[key] = graph
        return graph
