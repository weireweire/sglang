"""Vocab-parallel block-wise rejection of a DSpark verify chain
(SGLANG_OPT_DSV41_DSPARK_VOCAB_PARALLEL_VERIFY).

Each TP rank holds its vocab shard of the target verify logits (pre-softcap, the LM
head's output before the gather) and runs ``blockwise_rejection_triton(shards=TP)``'s
algorithm on it, so the full-vocab logits all-gather is never needed:

1. per shard-aligned stats block (and, under a simulated acceptance, per split of the
   simulated row's Gumbel-max bonus): the softcap, the block's max / sum-exp, and the
   target logit at every draft token the rank owns, pushed to every rank;
2. on every rank, the same one-warp accept over the gathered block stats;
3. the rank's blocks of the rejected row's residual Gumbel-max, pushed to every rank;
4. on every rank, the same insert over the gathered block maxima.

Every value, its block and its reduction order are those of the gathered call with
``shards=TP``, so the outputs (and the simulated bonus) match it bit for bit.

The pushes are one-shot ``multimem.st`` into a symmetric buffer, data as its own flag
(Lamport): a word is ready when it is no longer the sentinel, its reader resets it,
and three buffer sets rotate on the rejection noise's step counter.
"""

from __future__ import annotations

from typing import Optional

import msgspec
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.kernels.ops.speculative.dspark.blockwise_rejection import (
    _RESAMPLE_BLOCK,
    _STATS_BLOCK,
    _TL_RAND_MIN,
    RejectionNoise,
    _global_lse,
    _log1p_neg_stable,
    _murmur3_uniform32,
    _step_seed,
    shard_aligned_spans,
)

# Bit pattern of a word not yet pushed; a pushed word never has it.
_SENTINEL = tl.constexpr(-1)
# Canonical NaN that replaces a pushed value whose bits equal the sentinel.
_CANONICAL_NAN = tl.constexpr(0x7FFFFFFF)
_SIM_BLOCK = 4096
_PARITIES = 3


@triton.jit
def _push_b32(mc_ptr, value):
    """Thread 0 of the program stores the int32 ``value`` at ``mc_ptr`` of every rank."""
    value = tl.where(value == _SENTINEL, _CANONICAL_NAN, value)
    tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            .reg .u32 %t0;
            mov.u32 %t0, %tid.x;
            setp.eq.u32 %p0, %t0, 0;
            @%p0 multimem.st.relaxed.sys.global.b32 [$1], $2;
            mov.u32 $0, 0;
        }
        """,
        "=r,l,r",
        args=[mc_ptr, value],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _push_f32(mc_ptr, value):
    _push_b32(mc_ptr, value.to(tl.int32, bitcast=True))


@triton.jit
def _wait_words(ptr, offs, mask):
    pending = tl.full((), 1, tl.int32)
    while pending > 0:
        words = tl.load(ptr + offs, mask=mask, other=0, volatile=True)
        pending = tl.sum(tl.where(mask & (words == _SENTINEL), 1, 0), axis=0)


@triton.jit
def _gathered_value(x, mask, softcap, HAS_SOFTCAP: tl.constexpr):
    """The gathered logits' fp32 value of a shard element: the cast, or the softcap of
    gathered_softcap_logits op for op; masked lanes keep -inf."""
    x = x.to(tl.float32)
    if HAS_SOFTCAP:
        capped = libdevice.tanh(x / softcap) * softcap
        x = tl.where(mask, capped, x)
    return x


@triton.jit
def _vp_stats_kernel(
    mc_ptr,
    shard_logits_ptr,
    shard_stride,
    draft_logits_ptr,
    draft_stride_0,
    draft_stride_1,
    candidates_ptr,
    candidates_stride,
    target_temp_ptr,
    draft_temp_ptr,
    sim_len_ptr,
    sim_len_stride,
    sim_temp_ptr,
    sim_seed_ptr,
    counter_ptr,
    softcap,
    col0,
    shard_width,
    vocab,
    RANK: tl.constexpr,
    STRIDE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    LOCAL_BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
    SPAN: tl.constexpr,
    SIM_SPLITS: tl.constexpr,
    SIM_BLOCK: tl.constexpr,
    ROWS_MAX: tl.constexpr,
    PARITY_WORDS: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
    HAS_SIM: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1)
    req = row // STRIDE
    step = row % STRIDE
    base = (tl.load(counter_ptr) % 3) * PARITY_WORDS
    stats = base
    tok = base + 4 * ROWS_MAX * NUM_BLOCKS
    sim = tok + ROWS_MAX
    if step < STRIDE - 1:
        if j < LOCAL_BLOCKS:
            b = RANK * LOCAL_BLOCKS + j
            lane = tl.arange(0, BLOCK)
            local = j * SPAN + lane
            offs = col0 + local
            in_span = lane < SPAN
            t_temp = tl.load(target_temp_ptr + req).to(tl.float32)
            x = tl.load(
                # Four lanes per thread as the gathered fp32 load: the sum-exp's order.
                shard_logits_ptr + row * shard_stride + tl.max_contiguous(local, 4),
                mask=in_span & (offs < vocab),
                other=float("-inf"),
            )
            t = (
                _gathered_value(x, in_span & (offs < vocab), softcap, HAS_SOFTCAP)
                / t_temp
            )
            t_max = tl.max(t, axis=0)
            slot = row * NUM_BLOCKS + b
            _push_f32(mc_ptr + 4 * (stats + slot), t_max)
            _push_f32(
                mc_ptr + 4 * (stats + ROWS_MAX * NUM_BLOCKS + slot),
                tl.where(t_max > float("-inf"), tl.sum(tl.exp(t - t_max)), 0.0),
            )
            if HAS_DRAFT_LOGITS:
                d_temp = tl.load(draft_temp_ptr + req).to(tl.float32)
                d = (
                    tl.load(
                        draft_logits_ptr
                        + req * draft_stride_0
                        + step * draft_stride_1
                        + offs,
                        mask=in_span & (offs < vocab),
                        other=float("-inf"),
                    ).to(tl.float32)
                    / d_temp
                )
                d_max = tl.max(d, axis=0)
                _push_f32(
                    mc_ptr + 4 * (stats + 2 * ROWS_MAX * NUM_BLOCKS + slot), d_max
                )
                _push_f32(
                    mc_ptr + 4 * (stats + 3 * ROWS_MAX * NUM_BLOCKS + slot),
                    tl.where(d_max > float("-inf"), tl.sum(tl.exp(d - d_max)), 0.0),
                )
            if j == 0:
                token = tl.load(candidates_ptr + req * candidates_stride + step + 1).to(
                    tl.int64
                )
                token = tl.minimum(tl.maximum(token, 0), vocab - 1)
                if (token >= col0) & (token < col0 + shard_width):
                    x_tok = tl.load(
                        shard_logits_ptr + row * shard_stride + token - col0
                    )
                    _push_f32(
                        mc_ptr + 4 * (tok + row),
                        _gathered_value(x_tok, True, softcap, HAS_SOFTCAP),
                    )
    elif HAS_SIM:
        if j < SIM_SPLITS:
            # simulated_bonus_sample_graph's partial over this rank's columns.
            sim_row = req * STRIDE + tl.load(sim_len_ptr + req * sim_len_stride)
            t = tl.load(sim_temp_ptr + req).to(tl.float32)
            local = j * SIM_BLOCK + tl.arange(0, SIM_BLOCK)
            offs = col0 + local
            mask = (local < shard_width) & (offs < vocab)
            x = tl.load(
                shard_logits_ptr + sim_row.to(tl.int64) * shard_stride + local,
                mask=mask,
                other=float("-inf"),
            )
            x = _gathered_value(x, mask, softcap, HAS_SOFTCAP)
            seed = tl.load(sim_seed_ptr)
            u = tl.rand(seed, req.to(tl.int32) * vocab + offs)
            noisy = x / tl.maximum(t, 1e-5) - tl.log(-tl.log(u))
            v = tl.where(mask, tl.where(t > 0.0, noisy, x), float("-inf"))
            top = tl.max(v, axis=0)
            idx = tl.min(tl.where(v == top, offs, vocab), axis=0)
            split = RANK * SIM_SPLITS + j
            nsplit: tl.constexpr = NUM_BLOCKS // LOCAL_BLOCKS * SIM_SPLITS
            _push_f32(mc_ptr + 4 * (sim + req * 2 * nsplit + split), top)
            _push_b32(mc_ptr + 4 * (sim + req * 2 * nsplit + nsplit + split), idx)


@triton.jit
def _vp_rejection_kernel(
    correct_len_ptr,
    raw_len_ptr,
    trim_ptr,
    rej_lse_ptr,
    sim_bonus_ptr,
    local_ptr,
    candidates_ptr,
    candidates_stride,
    verify_lens_ptr,
    draft_logits_ptr,
    draft_stride_0,
    draft_stride_1,
    target_temp_ptr,
    draft_temp_ptr,
    vocab,
    bs,
    key,
    counter_ptr,
    STRIDE: tl.constexpr,
    PADDED_ROWS: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    PADDED_BLOCKS: tl.constexpr,
    NSPLIT: tl.constexpr,
    ROWS_MAX: tl.constexpr,
    PARITY_WORDS: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
    HAS_CUTOFF: tl.constexpr,
    HAS_SIM: tl.constexpr,
):
    req = tl.program_id(0).to(tl.int64)
    seed = _step_seed(key, counter_ptr)
    base = (tl.load(counter_ptr) % 3) * PARITY_WORDS
    plane: tl.constexpr = ROWS_MAX * NUM_BLOCKS
    tok = base + 4 * plane
    sim = tok + ROWS_MAX
    # Every word this request reads arrives before any is read or reset.
    cells = tl.arange(0, PADDED_ROWS * NUM_BLOCKS)
    rows_mask = cells < (STRIDE - 1) * NUM_BLOCKS
    row0 = req * STRIDE
    planes = 4 if HAS_DRAFT_LOGITS else 2
    for p in range(planes):
        _wait_words(local_ptr, base + p * plane + row0 * NUM_BLOCKS + cells, rows_mask)
    steps = tl.arange(0, PADDED_ROWS)
    steps_mask = steps < STRIDE - 1
    _wait_words(local_ptr, tok + row0 + steps, steps_mask)
    splits = tl.arange(0, 2 * NSPLIT)
    if HAS_SIM:
        _wait_words(local_ptr, sim + req * 2 * NSPLIT + splits, splits < 2 * NSPLIT)
    stats_f = local_ptr.to(tl.pointer_type(tl.float32))
    t_max_ptr = stats_f + base
    t_sumexp_ptr = stats_f + base + plane
    d_max_ptr = stats_f + base + 2 * plane
    d_sumexp_ptr = stats_f + base + 3 * plane
    t_temp = tl.load(target_temp_ptr + req).to(tl.float32)
    d_temp = 1.0
    if HAS_DRAFT_LOGITS:
        d_temp = tl.load(draft_temp_ptr + req).to(tl.float32)
    accepted = tl.zeros((), tl.int32)
    verifying = True
    for i in range(STRIDE - 1):
        if verifying:
            row = req * STRIDE + i
            token = tl.load(candidates_ptr + req * candidates_stride + i + 1).to(
                tl.int64
            )
            valid = (token >= 0) & (token < vocab)
            token = tl.minimum(tl.maximum(token, 0), vocab - 1)
            t_lse = _global_lse(t_max_ptr, t_sumexp_ptr, row, NUM_BLOCKS, PADDED_BLOCKS)
            log_p = tl.load(stats_f + tok + row) / t_temp - t_lse
            log_q = 0.0
            if HAS_DRAFT_LOGITS:
                d_lse = _global_lse(
                    d_max_ptr, d_sumexp_ptr, row, NUM_BLOCKS, PADDED_BLOCKS
                )
                log_q = (
                    tl.load(
                        draft_logits_ptr
                        + req * draft_stride_0
                        + i * draft_stride_1
                        + token,
                        mask=token < vocab,
                        other=float("-inf"),
                    ).to(tl.float32)
                    / d_temp
                    - d_lse
                )
            u = tl.maximum(tl.rand(seed, (req * STRIDE + i).to(tl.int32)), _TL_RAND_MIN)
            ok = valid & (log_q > float("-inf")) & (log_p > tl.log(u) + log_q)
            verifying = ok
            accepted += ok.to(tl.int32)
    tl.store(raw_len_ptr + req, accepted)
    capped = accepted
    if HAS_CUTOFF:
        cap = tl.load(verify_lens_ptr + req).to(tl.int32) - 1
        capped = tl.minimum(accepted, cap)
    tl.store(correct_len_ptr + req, capped)
    tl.store(trim_ptr + req, accepted - capped)
    t_lse_rej = 0.0
    d_lse_rej = 0.0
    if accepted < STRIDE - 1:
        rej_row = req * STRIDE + accepted
        t_lse_rej = _global_lse(
            t_max_ptr, t_sumexp_ptr, rej_row, NUM_BLOCKS, PADDED_BLOCKS
        )
        if HAS_DRAFT_LOGITS:
            d_lse_rej = _global_lse(
                d_max_ptr, d_sumexp_ptr, rej_row, NUM_BLOCKS, PADDED_BLOCKS
            )
    tl.store(rej_lse_ptr + req, t_lse_rej)
    tl.store(rej_lse_ptr + bs + req, d_lse_rej)
    if HAS_SIM:
        # _simulated_bonus_row_select_kernel over every rank's splits.
        val = tl.load(stats_f + sim + req * 2 * NSPLIT + tl.arange(0, NSPLIT))
        idx = tl.load(
            local_ptr + sim + req * 2 * NSPLIT + NSPLIT + tl.arange(0, NSPLIT)
        )
        top = tl.max(val, axis=0)
        tl.store(
            sim_bonus_ptr + req,
            tl.min(tl.where(val == top, idx, vocab), axis=0).to(tl.int64),
        )
    tl.debug_barrier()
    for p in range(planes):
        tl.store(
            local_ptr + base + p * plane + row0 * NUM_BLOCKS + cells,
            tl.full(cells.shape, _SENTINEL, tl.int32),
            mask=rows_mask,
        )
    tl.store(
        local_ptr + tok + row0 + steps,
        tl.full(steps.shape, _SENTINEL, tl.int32),
        mask=steps_mask,
    )
    if HAS_SIM:
        tl.store(
            local_ptr + sim + req * 2 * NSPLIT + splits,
            tl.full(splits.shape, _SENTINEL, tl.int32),
        )


@triton.jit
def _vp_resample_kernel(
    mc_ptr,
    parity_ptr,
    raw_len_ptr,
    rej_lse_ptr,
    candidates_ptr,
    candidates_stride,
    shard_logits_ptr,
    shard_stride,
    draft_logits_ptr,
    draft_stride_0,
    draft_stride_1,
    target_temp_ptr,
    draft_temp_ptr,
    softcap,
    col0,
    vocab,
    bs,
    key,
    counter_ptr,
    RANK: tl.constexpr,
    STRIDE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    LOCAL_BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
    SPAN: tl.constexpr,
    RES_OFFSET: tl.constexpr,
    PARITY_WORDS: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
):
    req = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1)
    block_idx = RANK * LOCAL_BLOCKS + j
    seed = _step_seed(key, counter_ptr)
    parity = tl.load(counter_ptr) % 3
    if (req == 0) & (j == 0):
        # The insert reads the parity here: its program 0 advances the counter.
        tl.store(parity_ptr, parity)
    res = parity * PARITY_WORDS + RES_OFFSET
    r = tl.load(raw_len_ptr + req).to(tl.int64)
    row = req * STRIDE + r
    is_bonus = r == STRIDE - 1
    lane = tl.arange(0, BLOCK)
    local = j * SPAN + lane
    offs = col0 + local
    mask = (lane < SPAN) & (offs < vocab)
    t_temp = tl.load(target_temp_ptr + req).to(tl.float32)
    x = tl.load(
        shard_logits_ptr + row * shard_stride + local,
        mask=mask,
        other=float("-inf"),
    )
    t = _gathered_value(x, mask, softcap, HAS_SOFTCAP) / t_temp
    if is_bonus:
        residual = t
    else:
        rejected = tl.load(candidates_ptr + req * candidates_stride + r + 1)
        if HAS_DRAFT_LOGITS:
            d_temp = tl.load(draft_temp_ptr + req).to(tl.float32)
            log_p = t - tl.load(rej_lse_ptr + req)
            log_q = tl.load(
                draft_logits_ptr + req * draft_stride_0 + r * draft_stride_1 + offs,
                mask=offs < vocab,
                other=float("-inf"),
            ).to(tl.float32) / d_temp - tl.load(rej_lse_ptr + bs + req)
            ratio = tl.exp(log_q - log_p)
            residual = tl.where(
                ratio < 1.0, log_p + libdevice.log1p(-ratio), float("-inf")
            ).to(tl.float32)
        else:
            residual = tl.where(offs != rejected, t, float("-inf"))
    u = _murmur3_uniform32(seed, req, offs)
    noised = tl.where(mask, residual - tl.log(-_log1p_neg_stable(u)), float("-inf"))
    value, idx = tl.max(noised, axis=0, return_indices=True)
    _push_f32(mc_ptr + 4 * (res + req * 2 * NUM_BLOCKS + block_idx), value)
    _push_b32(
        mc_ptr + 4 * (res + req * 2 * NUM_BLOCKS + NUM_BLOCKS + block_idx),
        (block_idx * SPAN + idx).to(tl.int32),
    )


@triton.jit
def _vp_insert_kernel(
    bonus_ptr,
    correct_len_ptr,
    raw_len_ptr,
    local_ptr,
    parity_ptr,
    candidates_ptr,
    candidates_stride,
    counter_ptr,
    RES_OFFSET: tl.constexpr,
    PARITY_WORDS: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    PADDED_BLOCKS: tl.constexpr,
):
    req = tl.program_id(0).to(tl.int64)
    res = tl.load(parity_ptr) * PARITY_WORDS + RES_OFFSET + req * 2 * NUM_BLOCKS
    words = tl.arange(0, 2 * PADDED_BLOCKS)
    words_mask = (words % PADDED_BLOCKS) < NUM_BLOCKS
    cells = (words // PADDED_BLOCKS) * NUM_BLOCKS + words % PADDED_BLOCKS
    _wait_words(local_ptr, res + cells, words_mask)
    res_max_ptr = local_ptr.to(tl.pointer_type(tl.float32)) + res
    res_arg_ptr = local_ptr + res + NUM_BLOCKS
    capped = tl.load(correct_len_ptr + req)
    raw = tl.load(raw_len_ptr + req)
    if capped < raw:
        bonus = tl.load(candidates_ptr + req * candidates_stride + capped + 1).to(
            tl.int64
        )
    else:
        blocks = tl.arange(0, PADDED_BLOCKS)
        local_max = tl.load(
            res_max_ptr + blocks, mask=blocks < NUM_BLOCKS, other=float("-inf")
        )
        local_max = tl.where(local_max != local_max, float("-inf"), local_max)
        best = tl.argmax(local_max, axis=0)
        bonus = tl.load(res_arg_ptr + best).to(tl.int64)
    tl.store(bonus_ptr + req, bonus)
    tl.debug_barrier()
    tl.store(
        local_ptr + res + cells,
        tl.full(words.shape, _SENTINEL, tl.int32),
        mask=words_mask,
    )
    if req == 0:
        tl.store(counter_ptr, tl.load(counter_ptr) + 1)


class VocabParallelExchange(msgspec.Struct):
    """The symmetric push buffer of the vocab-parallel rejection and its geometry."""

    rank: int
    world: int
    shard_width: int
    vocab: int
    stride: int
    max_bs: int
    stats_span: int
    res_span: int
    local_blocks: int
    res_local_blocks: int
    sim_splits: int
    rows_max: int
    parity_words: int
    res_offset: int
    buffer: torch.Tensor
    multicast_ptr: int
    parity: torch.Tensor
    handle: object


def make_vocab_parallel_exchange(
    *,
    group: dist.ProcessGroup,
    rank: int,
    world: int,
    vocab: int,
    stride: int,
    max_bs: int,
    device,
) -> VocabParallelExchange:
    """Collective, outside graph capture: every rank calls it with the same args."""
    stats_span, res_span = shard_aligned_spans(vocab, world)
    shard_width = vocab // world
    local_blocks = shard_width // stats_span
    res_local_blocks = shard_width // res_span
    num_blocks = world * local_blocks
    sim_splits = triton.cdiv(shard_width, _SIM_BLOCK)
    nsplit = world * sim_splits
    assert nsplit == triton.next_power_of_2(nsplit)
    rows_max = max_bs * stride
    x1_words = 4 * rows_max * num_blocks + rows_max + max_bs * 2 * nsplit
    res_offset = triton.cdiv(x1_words, 16) * 16
    parity_words = (
        triton.cdiv(res_offset + max_bs * 2 * world * res_local_blocks, 16) * 16
    )
    with torch.inference_mode(False), torch.no_grad():
        buffer = symm_mem.empty(
            (_PARITIES * parity_words,), dtype=torch.int32, device=device
        )
    handle = symm_mem.rendezvous(buffer, group=group)
    assert handle.rank == rank and handle.multicast_ptr != 0
    buffer.fill_(-1)
    torch.cuda.synchronize()
    dist.barrier(group=group)
    return VocabParallelExchange(
        rank=rank,
        world=world,
        shard_width=shard_width,
        vocab=vocab,
        stride=stride,
        max_bs=max_bs,
        stats_span=stats_span,
        res_span=res_span,
        local_blocks=local_blocks,
        res_local_blocks=res_local_blocks,
        sim_splits=sim_splits,
        rows_max=rows_max,
        parity_words=parity_words,
        res_offset=res_offset,
        buffer=buffer,
        multicast_ptr=handle.multicast_ptr,
        parity=torch.zeros((1,), dtype=torch.int64, device=device),
        handle=handle,
    )


class SimulatedBonusArgs(msgspec.Struct, frozen=True):
    """simulated_bonus_sample_graph's inputs."""

    correct_len: torch.Tensor
    temperatures: torch.Tensor
    seed: torch.Tensor


def vocab_parallel_rejection(
    *,
    exchange: VocabParallelExchange,
    candidates: torch.Tensor,
    shard_logits: torch.Tensor,
    softcap: Optional[float],
    draft_logits: Optional[torch.Tensor],
    target_temperatures: torch.Tensor,
    draft_temperatures: torch.Tensor,
    noise: RejectionNoise,
    cutoff_verify_lens: Optional[torch.Tensor] = None,
    sim: Optional[SimulatedBonusArgs] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """``blockwise_rejection_triton(shards=world)`` on the gathered fp32 logits, from
    this rank's LM-head shard ``[bs * stride, vocab / world]`` (pre-softcap when
    ``softcap`` is set). Returns
    (correct_len, bonus, cap_trim_lens, simulated bonus or None); with ``sim`` the
    last is simulated_bonus_sample_graph's draw, bit for bit."""
    ex = exchange
    bs, stride = candidates.shape
    assert stride == ex.stride and bs <= ex.max_bs
    assert shard_logits.shape == (bs * stride, ex.shard_width)
    assert shard_logits.stride(-1) == 1
    has_draft = draft_logits is not None
    if has_draft:
        assert draft_logits.stride(-1) == 1 and draft_logits.shape[1] >= stride - 1
        assert draft_logits.shape[-1] == ex.vocab
    device = candidates.device
    target_temperatures = target_temperatures.reshape(-1).contiguous()
    draft_temperatures = draft_temperatures.reshape(-1).contiguous()
    num_blocks = ex.world * ex.local_blocks
    res_blocks = ex.world * ex.res_local_blocks
    nsplit = ex.world * ex.sim_splits
    col0 = ex.rank * ex.shard_width
    lens = torch.empty((3, bs), dtype=torch.int32, device=device)
    rej_lse = torch.empty((2, bs), dtype=torch.float32, device=device)
    bonus = torch.empty((bs,), dtype=torch.int64, device=device)
    sim_bonus = torch.empty((bs,), dtype=torch.int64, device=device) if sim else None
    correct_len, raw_len, trim = lens[0], lens[1], lens[2]
    d_ptr = draft_logits if has_draft else shard_logits
    d_s0 = draft_logits.stride(0) if has_draft else 0
    d_s1 = draft_logits.stride(1) if has_draft else 0
    has_sim = sim is not None
    has_softcap = softcap is not None
    softcap = softcap if has_softcap else 1.0
    sim_len = sim.correct_len if has_sim else lens[0]
    sim_temp = sim.temperatures if has_sim else target_temperatures
    sim_seed = sim.seed if has_sim else lens[0]
    _vp_stats_kernel[(bs * stride, max(ex.local_blocks, ex.sim_splits))](
        ex.multicast_ptr,
        shard_logits, shard_logits.stride(0),
        d_ptr, d_s0, d_s1,
        candidates, candidates.stride(0),
        target_temperatures, draft_temperatures,
        sim_len, sim_len.stride(0), sim_temp, sim_seed,
        noise.counter,
        softcap, col0, ex.shard_width, ex.vocab,
        RANK=ex.rank, STRIDE=stride, NUM_BLOCKS=num_blocks,
        LOCAL_BLOCKS=ex.local_blocks, BLOCK=_STATS_BLOCK, SPAN=ex.stats_span,
        SIM_SPLITS=ex.sim_splits, SIM_BLOCK=_SIM_BLOCK,
        ROWS_MAX=ex.rows_max, PARITY_WORDS=ex.parity_words,
        HAS_DRAFT_LOGITS=has_draft, HAS_SIM=has_sim, HAS_SOFTCAP=has_softcap,
    )  # fmt: skip
    has_cutoff = cutoff_verify_lens is not None
    verify_lens = (
        cutoff_verify_lens.to(device=device).contiguous() if has_cutoff else correct_len
    )
    _vp_rejection_kernel[(bs,)](
        correct_len, raw_len, trim, rej_lse,
        sim_bonus if has_sim else bonus,
        ex.buffer,
        candidates, candidates.stride(0), verify_lens,
        d_ptr, d_s0, d_s1,
        target_temperatures, draft_temperatures,
        ex.vocab, bs,
        noise.key, noise.counter,
        STRIDE=stride, PADDED_ROWS=triton.next_power_of_2(stride - 1),
        NUM_BLOCKS=num_blocks, PADDED_BLOCKS=triton.next_power_of_2(num_blocks),
        NSPLIT=nsplit, ROWS_MAX=ex.rows_max, PARITY_WORDS=ex.parity_words,
        HAS_DRAFT_LOGITS=has_draft, HAS_CUTOFF=has_cutoff, HAS_SIM=has_sim,
        num_warps=1,
    )  # fmt: skip
    _vp_resample_kernel[(bs, ex.res_local_blocks)](
        ex.multicast_ptr, ex.parity,
        raw_len, rej_lse,
        candidates, candidates.stride(0),
        shard_logits, shard_logits.stride(0),
        d_ptr, d_s0, d_s1,
        target_temperatures, draft_temperatures,
        softcap, col0, ex.vocab, bs,
        noise.key, noise.counter,
        RANK=ex.rank, STRIDE=stride, NUM_BLOCKS=res_blocks,
        LOCAL_BLOCKS=ex.res_local_blocks, BLOCK=_RESAMPLE_BLOCK, SPAN=ex.res_span,
        RES_OFFSET=ex.res_offset, PARITY_WORDS=ex.parity_words,
        HAS_DRAFT_LOGITS=has_draft, HAS_SOFTCAP=has_softcap,
    )  # fmt: skip
    _vp_insert_kernel[(bs,)](
        bonus, correct_len, raw_len,
        ex.buffer, ex.parity,
        candidates, candidates.stride(0), noise.counter,
        RES_OFFSET=ex.res_offset, PARITY_WORDS=ex.parity_words,
        NUM_BLOCKS=res_blocks, PADDED_BLOCKS=triton.next_power_of_2(res_blocks),
        num_warps=1,
    )  # fmt: skip
    return correct_len, bonus, trim, sim_bonus
