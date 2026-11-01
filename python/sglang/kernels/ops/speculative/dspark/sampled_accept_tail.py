"""DSpark static sampled verify: the accept bookkeeping in one launch
(SGLANG_OPT_DSV41_DSPARK_SAMPLED_ACCEPT_TAIL), one program per request row.

From the sampled accept's correct lengths, bonus and cap trims: the commit / new seq
lens (FinalizeAcceptLens), the output tokens (BuildOutTokens), the block accept
lengths, the gated commit lengths the draft-KV commit reads, and optionally the
FutureMap relay scatter. Every output equals the separate ops' bit for bit.
"""

from __future__ import annotations

from typing import Optional

import msgspec
import torch
import triton
import triton.language as tl


class CommitStage(msgspec.Struct, frozen=True):
    """stage_commit's operands: the verify rows' write locations and positions into
    the deferred commit's buffers, with per-request state slots when it keeps them."""

    cache_loc: torch.Tensor
    positions: torch.Tensor
    cache_loc_dst: torch.Tensor
    positions_dst: torch.Tensor
    commit_lens_dst: torch.Tensor
    state_slot_dst: Optional[torch.Tensor]


class CommitTail(msgspec.Struct, frozen=True):
    """The commits run_static issues after the accept, folded into its launch."""

    req_slots: torch.Tensor
    history: Optional[torch.Tensor]
    stage: Optional[CommitStage]


@triton.jit
def _sampled_accept_tail_kernel(
    correct_ptr,
    bonus_ptr,
    cap_trim_ptr,
    seq_lens_ptr,
    candidates_ptr,
    verify_lens_ptr,
    gate_ptr,
    rows_ptr,
    correct_out_ptr,
    bonus_out_ptr,
    cap_trim_out_ptr,
    commit_out_ptr,
    new_seq_out_ptr,
    out_tokens_ptr,
    block_out_ptr,
    gated_out_ptr,
    relay_seq_ptr,
    relay_tok_ptr,
    history_ptr,
    loc_src_ptr,
    loc_dst_ptr,
    pos_src_ptr,
    pos_dst_ptr,
    slot_dst_ptr,
    lens_dst_ptr,
    bs,
    max_bs,
    cand_stride,
    out_tokens_stride,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
    HAS_BLOCK: tl.constexpr,
    HAS_RELAY: tl.constexpr,
    HISTORY_STRIDE: tl.constexpr,
    HISTORY_WIDTH: tl.constexpr,
    HISTORY_BLOCK: tl.constexpr,
    HAS_HISTORY: tl.constexpr,
    HAS_STAGE: tl.constexpr,
    HAS_SLOT: tl.constexpr,
    LENS_BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_T)
    in_t = offs < T
    in_draft = offs < T - 1
    correct = tl.load(correct_ptr + row)
    bonus = tl.load(bonus_ptr + row).to(tl.int64)
    cap_trim = tl.load(cap_trim_ptr + row).to(tl.int32)
    commit = correct.to(tl.int32) + 1
    seq = tl.load(seq_lens_ptr + row)
    new_seq = seq + commit.to(seq.dtype)
    tl.store(correct_out_ptr + row, correct)
    tl.store(bonus_out_ptr + row, bonus)
    tl.store(cap_trim_out_ptr + row, cap_trim)
    tl.store(commit_out_ptr + row, commit)
    tl.store(new_seq_out_ptr + row, new_seq)
    drafts = tl.load(
        candidates_ptr + row * cand_stride + 1 + offs, mask=in_draft, other=0
    )
    token = tl.where(
        offs == correct.to(tl.int32), bonus, tl.where(in_draft, drafts.to(tl.int64), 0)
    )
    tl.store(out_tokens_ptr + row * out_tokens_stride + offs, token, mask=in_t)
    if HAS_BLOCK:
        tl.store(block_out_ptr + row, commit + cap_trim)
    verify_len = tl.load(verify_lens_ptr + row).to(tl.int32)
    gate = tl.load(gate_ptr)
    gated = tl.minimum(commit, verify_len) * gate
    tl.store(gated_out_ptr + row, gated)
    if HAS_RELAY:
        slot = tl.load(rows_ptr + row).to(tl.int64)
        tl.store(relay_seq_ptr + slot, new_seq)
        tl.store(relay_tok_ptr + slot, bonus)
    if HAS_HISTORY:
        # engram_commit_history: padded rows (verify_len 0) and gate 0 skip.
        live = (verify_len != 0) & (gate != 0)
        req = tl.load(rows_ptr + row).to(tl.int64)
        col = tl.arange(0, HISTORY_BLOCK)
        source_col = commit + col
        valid = (col < HISTORY_WIDTH) & live
        old = tl.load(
            history_ptr + req * HISTORY_STRIDE + source_col,
            mask=valid & (source_col < HISTORY_WIDTH),
            other=0,
        )
        new = tl.load(
            candidates_ptr + row * cand_stride + source_col - HISTORY_WIDTH,
            mask=valid & (source_col >= HISTORY_WIDTH),
            other=0,
        )
        values = tl.where(source_col < HISTORY_WIDTH, old, new)
        tl.debug_barrier()
        tl.store(history_ptr + req * HISTORY_STRIDE + col, values, mask=valid)
    if HAS_STAGE:
        # stage_commit: this request's rows, and zeros past the staged requests.
        r = row * T + offs
        tl.store(loc_dst_ptr + r, tl.load(loc_src_ptr + r, mask=in_t), mask=in_t)
        tl.store(pos_dst_ptr + r, tl.load(pos_src_ptr + r, mask=in_t), mask=in_t)
        if HAS_SLOT:
            req_slot = tl.load(rows_ptr + row).to(tl.int64)
            tl.store(
                slot_dst_ptr + r, tl.zeros([BLOCK_T], tl.int64) + req_slot, mask=in_t
            )
        tl.store(lens_dst_ptr + row, gated)
        if row == 0:
            q = tl.arange(0, LENS_BLOCK)
            pad = (q >= bs) & (q < max_bs)
            tl.store(lens_dst_ptr + q, tl.zeros([LENS_BLOCK], tl.int32), mask=pad)


def sampled_accept_tail(
    *,
    correct_len: torch.Tensor,
    bonus: torch.Tensor,
    cap_trim_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    candidates: torch.Tensor,
    verify_lens: torch.Tensor,
    gate: torch.Tensor,
    correct_len_out: torch.Tensor,
    bonus_out: torch.Tensor,
    cap_trim_lens_out: torch.Tensor,
    commit_lens_out: torch.Tensor,
    new_seq_lens_out: torch.Tensor,
    out_tokens_out: torch.Tensor,
    gated_commit_lens_out: torch.Tensor,
    block_accept_lens_out: Optional[torch.Tensor] = None,
    relay: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    commit: Optional[CommitTail] = None,
) -> None:
    """``candidates`` [bs, T] (anchor + drafts); ``relay`` (req_pool_indices,
    new_seq_lens table, output_tokens table) scatters the row results by slot;
    ``commit`` also runs the engram history commit and the deferred-commit staging."""
    bs, T = candidates.shape
    assert candidates.stride(1) == 1 and out_tokens_out.stride(1) == 1
    dummy = seq_lens
    req_slots = relay[0] if relay is not None else None
    history = commit.history if commit is not None else None
    stage = commit.stage if commit is not None else None
    if commit is not None:
        assert req_slots is None or req_slots.data_ptr() == commit.req_slots.data_ptr()
        req_slots = commit.req_slots
    if history is not None:
        assert history.stride(1) == 1 and history.shape[1] > 0
    if stage is not None:
        assert stage.cache_loc.is_contiguous() and stage.positions.is_contiguous()
    max_bs = stage.commit_lens_dst.shape[0] if stage is not None else bs
    _sampled_accept_tail_kernel[(bs,)](
        correct_len,
        bonus.view(-1),
        cap_trim_lens,
        seq_lens,
        candidates,
        verify_lens,
        gate,
        req_slots if req_slots is not None else dummy,
        correct_len_out,
        bonus_out,
        cap_trim_lens_out,
        commit_lens_out,
        new_seq_lens_out,
        out_tokens_out,
        block_accept_lens_out if block_accept_lens_out is not None else dummy,
        gated_commit_lens_out,
        relay[1] if relay is not None else dummy,
        relay[2] if relay is not None else dummy,
        history if history is not None else dummy,
        stage.cache_loc if stage is not None else dummy,
        stage.cache_loc_dst if stage is not None else dummy,
        stage.positions if stage is not None else dummy,
        stage.positions_dst if stage is not None else dummy,
        (
            stage.state_slot_dst
            if stage is not None and stage.state_slot_dst is not None
            else dummy
        ),
        stage.commit_lens_dst if stage is not None else dummy,
        bs,
        max_bs,
        candidates.stride(0),
        out_tokens_out.stride(0),
        T=T,
        BLOCK_T=triton.next_power_of_2(T),
        HAS_BLOCK=block_accept_lens_out is not None,
        HAS_RELAY=relay is not None,
        HISTORY_STRIDE=history.stride(0) if history is not None else 0,
        HISTORY_WIDTH=history.shape[1] if history is not None else 0,
        HISTORY_BLOCK=triton.next_power_of_2(history.shape[1])
        if history is not None
        else 1,
        HAS_HISTORY=history is not None,
        HAS_STAGE=stage is not None,
        HAS_SLOT=stage is not None and stage.state_slot_dst is not None,
        LENS_BLOCK=triton.next_power_of_2(max_bs),
        num_warps=1,
    )
