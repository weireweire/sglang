"""DSpark static sampled verify: the accept bookkeeping in one launch
(SGLANG_OPT_DSV41_DSPARK_SAMPLED_ACCEPT_TAIL), one program per request row.

From the sampled accept's correct lengths, bonus and cap trims: the commit / new seq
lens (FinalizeAcceptLens), the output tokens (BuildOutTokens), the block accept
lengths, the gated commit lengths the draft-KV commit reads, and optionally the
FutureMap relay scatter. Every output equals the separate ops' bit for bit.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


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
    cand_stride,
    out_tokens_stride,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
    HAS_BLOCK: tl.constexpr,
    HAS_RELAY: tl.constexpr,
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
    drafts = tl.load(candidates_ptr + row * cand_stride + 1 + offs, mask=in_draft, other=0)
    token = tl.where(
        offs == correct.to(tl.int32), bonus, tl.where(in_draft, drafts.to(tl.int64), 0)
    )
    tl.store(out_tokens_ptr + row * out_tokens_stride + offs, token, mask=in_t)
    if HAS_BLOCK:
        tl.store(block_out_ptr + row, commit + cap_trim)
    verify_len = tl.load(verify_lens_ptr + row).to(tl.int32)
    gate = tl.load(gate_ptr)
    tl.store(gated_out_ptr + row, tl.minimum(commit, verify_len) * gate)
    if HAS_RELAY:
        slot = tl.load(rows_ptr + row).to(tl.int64)
        tl.store(relay_seq_ptr + slot, new_seq)
        tl.store(relay_tok_ptr + slot, bonus)


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
) -> None:
    """``candidates`` [bs, T] (anchor + drafts); ``relay`` (req_pool_indices,
    new_seq_lens table, output_tokens table) scatters the row results by slot."""
    bs, T = candidates.shape
    assert candidates.stride(1) == 1 and out_tokens_out.stride(1) == 1
    dummy = seq_lens
    _sampled_accept_tail_kernel[(bs,)](
        correct_len,
        bonus.view(-1),
        cap_trim_lens,
        seq_lens,
        candidates,
        verify_lens,
        gate,
        relay[0] if relay is not None else dummy,
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
        candidates.stride(0),
        out_tokens_out.stride(0),
        T=T,
        BLOCK_T=triton.next_power_of_2(T),
        HAS_BLOCK=block_accept_lens_out is not None,
        HAS_RELAY=relay is not None,
        num_warps=1,
    )
