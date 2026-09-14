import random
import threading
from array import array
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import InsertResult, MatchResult
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
)
from sglang.srt.mem_cache.unified_cache.cache_action import (
    BackupKV,
    FreeDeviceKV,
    ReplaceWriteThroughOnNodeSplit,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.full_component import FullComponent
from sglang.srt.mem_cache.unified_cache.components.swa_component import SWAComponent
from sglang.srt.mem_cache.unified_cache.components.tree_component import (
    ExternalLinkerLoadPhase,
    LinkerTransferPhase,
)
from sglang.srt.mem_cache.unified_cache.swa_retention import retained_swa_ranges
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    UnifiedCacheLinker,
    UnifiedCacheLinkerWrapper,
    _PendingLookup,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _FakeLinker(UnifiedCacheLinker):
    def __init__(self):
        self.layer_done_counter = object()
        self.restorable = []
        self.lookup_calls = 0
        self.completed_lookups = []
        self.queued_loads = {}
        self.queued_offloads = []
        self.completed_loads = []
        self.completed_offloads = []
        self.reset_count = 0
        self.closed = False

    def lookup(self, rid, transfers):
        self.lookup_calls += 1
        return list(self.restorable)

    def load(self, rid, transfers):
        self.queued_loads[rid] = list(transfers)
        return True

    def num_completed_lookups(self):
        return len(self.completed_lookups)

    def pop_completed_lookup(self):
        return self.completed_lookups.pop(0)

    def start_layer_wise_loading(self):
        return 3

    def cancel_queued_load(self, rid):
        if rid not in self.queued_loads:
            return False
        del self.queued_loads[rid]
        return True

    def num_completed_loads(self):
        return len(self.completed_loads)

    def pop_completed_load(self):
        return self.completed_loads.pop(0)

    def offload(self, transfers):
        self.queued_offloads.append(list(transfers))
        return True

    def num_completed_offloads(self):
        return len(self.completed_offloads)

    def pop_completed_offload(self):
        return self.completed_offloads.pop(0)

    def reset(self):
        self.reset_count += 1

    def close(self):
        self.closed = True


class _FakeMooncakeSessionStore:
    def __init__(self, failed_key=None):
        self.failed_key = failed_key
        self.start_calls = []
        self.end_calls = []
        self.store = self

    def _get_hybrid_page_component_keys(self, keys, transfer):
        return [f"{key}:{transfer.name}" for key in keys], 1

    def _tag_keys(self, keys):
        return list(keys)

    def batch_get_session_start(self, keys):
        self.start_calls.append(list(keys))
        return [-704 if key == self.failed_key else 0 for key in keys]

    def batch_get_session_end(self, keys):
        self.end_calls.append(list(keys))
        return 0


def _mooncake_linker_for_session_test(failed_key=None):
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.pool_group = SimpleNamespace(
        resolve_transfers=lambda transfers, **kwargs: list(transfers)
    )
    linker.storage = _FakeMooncakeSessionStore(failed_key)
    linker.load_sessions = {}
    linker.load_session_refcounts = {}
    linker.load_session_lock = threading.Lock()
    linker.pending_loads = {}
    linker.stats = {
        "reserve_lock_seconds": 0.0,
        "reserve_lock_max_seconds": 0.0,
        "reserve_rpc_count": 0,
        "reserve_rpc_seconds": 0.0,
        "reserve_rpc_max_seconds": 0.0,
    }
    return linker


class _MappingRecorder:
    def __init__(self):
        self.mapping = []

    def set_full_to_swa_mapping(self, full, swa):
        self.mapping.append((full.clone(), swa.clone()))


def _cache_for_wrapper(**kwargs):
    defaults = {
        "page_size": 1,
        "tree_core": SimpleNamespace(enable_external_cache_linker=False),
        "write_through_threshold": 256,
        "pp_size": 1,
        "pp_group": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_cache_linker_attachment_is_backend_independent():
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.tree_core = SimpleNamespace(
        page_size=1,
        enable_external_cache_linker=False,
        write_through_threshold=256,
    )
    cache.linker = None
    linker = _FakeLinker()

    cache.init_cache_linker(linker)

    assert cache.linker.cache_linker is linker
    assert cache.tree_core.enable_external_cache_linker
    assert cache.write_through_threshold == 1
    assert cache.linker.layer_done_counter is linker.layer_done_counter


def test_external_linker_supports_selective_admission():
    cache = _cache_for_wrapper()

    with envs.SGLANG_EXTERNAL_LINKER_WRITE_THROUGH_THRESHOLD.override(2):
        UnifiedCacheLinkerWrapper(cache, _FakeLinker())

    assert cache.write_through_threshold == 2


def test_mooncake_load_sessions_reference_count_shared_keys():
    linker = _mooncake_linker_for_session_test()
    transfer = PoolTransfer(name=PoolName.DEEPSEEK_V4_C4, keys=["a", "b"])

    assert linker.reserve_load("r1", [transfer])
    assert linker.reserve_load("r2", [transfer])
    assert linker.storage.start_calls == [["a:deepseek_v4_c4", "b:deepseek_v4_c4"]]
    assert linker.stats["reserve_rpc_count"] == 1
    assert linker.stats["reserve_rpc_seconds"] >= 0

    linker._release_load_session("r1")
    assert linker.storage.end_calls == []
    linker._release_load_session("r2")
    assert len(linker.storage.end_calls) == 1
    assert set(linker.storage.end_calls[0]) == {
        "a:deepseek_v4_c4",
        "b:deepseek_v4_c4",
    }


def test_mooncake_load_reservation_miss_releases_partial_session():
    linker = _mooncake_linker_for_session_test(failed_key="b:deepseek_v4_c4")
    transfer = PoolTransfer(name=PoolName.DEEPSEEK_V4_C4, keys=["a", "b"])

    assert not linker.reserve_load("r1", [transfer])
    assert linker.load_sessions == {}
    assert linker.load_session_refcounts == {}
    assert linker.storage.end_calls == [["a:deepseek_v4_c4"]]
    assert linker.stats["reserve_rpc_count"] == 1


def test_mooncake_load_keeps_only_committed_session_keys():
    linker = _mooncake_linker_for_session_test()
    lookup = PoolTransfer(name=PoolName.DEEPSEEK_V4_C4, keys=["a", "b"])
    load = PoolTransfer(name=PoolName.DEEPSEEK_V4_C4, keys=["b"])

    assert linker.reserve_load("r1", [lookup])
    assert linker.load("r1", [load])
    assert linker.load_sessions == {"r1": {"b:deepseek_v4_c4"}}
    assert linker.storage.end_calls == [["a:deepseek_v4_c4"]]


@pytest.mark.parametrize("boundary", [None, 32768])
def test_external_linker_coalesces_insert_walk_backups(boundary):
    core = UnifiedTreeCore.__new__(UnifiedTreeCore)
    core.enable_external_cache_linker = True
    free = FreeDeviceKV([])
    actions = [BackupKV([7], replay_boundary=boundary), free]

    core._append_backup_action(actions, BackupKV([7, 8], replay_boundary=boundary))

    assert actions == [BackupKV([7, 8], replay_boundary=boundary), free]
    assert core._is_deferrable_action(actions[0])


def test_restorable_prefix_intersects_sparse_rank_results():
    remote_mask = torch.tensor([0, 0, 1, 0, 0], dtype=torch.int)

    def intersect_remote_mask(mask, op):
        assert op == torch.distributed.ReduceOp.MIN
        mask.copy_(torch.minimum(mask, remote_mask))

    cache = _cache_for_wrapper(_all_reduce_attn_groups=intersect_remote_mask)
    wrapper = UnifiedCacheLinkerWrapper(cache, _FakeLinker())

    hit_pages = wrapper._sync_restorable_prefix([2, 4], num_pages=4, device_hit_pages=0)

    assert hit_pages == 2


@pytest.mark.parametrize(("restorable", "expected_hit_tokens"), [([], 0), ([2], 4)])
def test_waiting_request_refreshes_consumed_lookup(restorable, expected_hit_tokens):
    class _Component:
        def build_external_linker_transfer(self, phase, node, keys):
            assert phase == LinkerTransferPhase.LOOKUP
            return PoolTransfer(name=PoolName.KV, keys=list(keys))

    linker = _FakeLinker()
    linker.restorable = restorable
    cache = _cache_for_wrapper(
        page_size=2,
        _components_tuple=(_Component(),),
        _all_reduce_attn_groups=lambda mask, op: None,
        get_last_hash_value=lambda node: None,
    )
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)
    key = RadixKey(array("q", range(8)))
    req = SimpleNamespace(rid="request")
    result = MatchResult(
        device_indices=torch.empty(0, dtype=torch.int64),
        last_device_node=None,
        last_host_node=None,
        best_match_node=0,
    )

    first = wrapper.match(key, req, result)
    second = wrapper.match(key, req, result)

    assert first.host_hit_length == expected_hit_tokens
    assert second.host_hit_length == expected_hit_tokens
    assert linker.lookup_calls == 2

    wrapper.release_request(req.rid)
    wrapper.match(key, req, result)
    assert linker.lookup_calls == 3


def test_async_lookup_is_published_after_common_rank_completion():
    class _AsyncLinker(_FakeLinker):
        def lookup(self, rid, transfers):
            self.lookup_calls += 1

    class _Component:
        def build_external_linker_transfer(self, phase, node, keys):
            assert phase == LinkerTransferPhase.LOOKUP
            return PoolTransfer(name=PoolName.KV, keys=list(keys))

    linker = _AsyncLinker()
    cache = _cache_for_wrapper(
        page_size=2,
        _components_tuple=(_Component(),),
        _all_reduce_attn_groups=lambda mask, op: None,
        get_last_hash_value=lambda node: None,
    )
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)
    key = RadixKey(array("q", range(8)))
    req = SimpleNamespace(rid="request")
    result = MatchResult(
        device_indices=torch.empty(0, dtype=torch.int64),
        last_device_node=None,
        last_host_node=None,
        best_match_node=0,
    )

    assert wrapper.match(key, req, result).host_hit_length == 0
    assert wrapper.has_pending_lookup(req.rid)
    assert wrapper.match(key, req, result).host_hit_length == 0
    assert linker.lookup_calls == 1

    linker.completed_lookups.append((req.rid, [2]))
    wrapper.drain_lookups(1)

    assert not wrapper.has_pending_lookup(req.rid)
    assert wrapper.match(key, req, result).host_hit_length == 4
    assert linker.lookup_calls == 1

    # The completed result was consumed by the preceding admission attempt.
    # A still-waiting request refreshes it instead of trusting a stale hit.
    assert wrapper.match(key, req, result).host_hit_length == 0
    assert wrapper.has_pending_lookup(req.rid)
    assert linker.lookup_calls == 2


def test_completed_lookups_share_one_rank_collective():
    linker = _FakeLinker()
    reduce_calls = []

    def reduce_masks(mask, op):
        reduce_calls.append(mask.shape)
        assert op == torch.distributed.ReduceOp.MIN

    cache = _cache_for_wrapper(
        page_size=2,
        _all_reduce_attn_groups=reduce_masks,
    )
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)
    tail_lengths = [3, 1, 5]
    for index, tail_length in enumerate(tail_lengths):
        rid = f"request-{index}"
        wrapper.pending_lookups[rid] = _PendingLookup(
            prefix_key=RadixKey(array("q", range(tail_length * 2))),
            tail_hashes=[f"page-{page}" for page in range(tail_length)],
            device_hit_len=0,
            transfers=[PoolTransfer(name=PoolName.KV, keys=["page-0"])],
        )
        linker.completed_lookups.append((rid, list(range(1, tail_length + 1))))

    wrapper.drain_lookups(3)

    assert reduce_calls == [torch.Size([12])]
    assert not wrapper.pending_lookups
    assert all(
        wrapper.lookup_results[rid] is not None for rid in wrapper.lookup_results
    )


def test_async_offload_pins_node_until_completion():
    class _Component:
        def build_external_linker_transfer(self, phase, node, keys):
            assert phase == LinkerTransferPhase.OFFLOAD
            return PoolTransfer(
                name=PoolName.KV,
                keys=["page"],
                device_indices=torch.tensor([node.id]),
            )

    linker = _FakeLinker()
    lock_params = object()
    locks = []
    unlocks = []

    def inc_lock_ref(node):
        locks.append(node)
        return SimpleNamespace(to_dec_params=lambda: lock_params)

    node_id = 7
    node = SimpleNamespace(
        id=node_id,
        external_cache_stored=False,
        write_through_pending_id=None,
    )
    cache = _cache_for_wrapper(
        tree_core=SimpleNamespace(
            enable_external_cache_linker=False,
            mark_write_through_pending=lambda node_ids, ack_id: (
                setattr(node, "write_through_pending_id", ack_id) or list(node_ids)
            ),
        ),
        _components_tuple=(_Component(),),
        inc_lock_ref=inc_lock_ref,
        dec_lock_ref=lambda node, params: unlocks.append((node, params)),
        resolve_node_handle=lambda value: node if value == node_id else None,
    )
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)

    wrapper.offload_nodes([node_id])

    assert locks == [node_id]
    assert node.external_cache_stored
    assert not unlocks

    linker.completed_offloads.append(False)
    completed = wrapper.take_completed_offloads(finish_count=1)
    wrapper.commit_completed_offloads(completed)

    assert not node.external_cache_stored
    assert unlocks == [(node_id, lock_params)]


@pytest.mark.parametrize("success", [True, False])
def test_offload_pins_every_swa_source_until_completion(success):
    class Component:
        def build_external_linker_transfer(self, phase, node, keys):
            return PoolTransfer(
                name=PoolName.SWA,
                keys=[str(node.id)],
                device_indices=torch.tensor([node.id]),
            )

    nodes = {
        i: SimpleNamespace(
            id=i, external_cache_stored=False, write_through_pending_id=None
        )
        for i in (1, 2)
    }
    locks, unlocks = [], []

    def mark(node_ids, ack_id):
        for i in node_ids:
            nodes[i].write_through_pending_id = ack_id
        return list(node_ids)

    cache = _cache_for_wrapper(
        tree_core=SimpleNamespace(
            enable_external_cache_linker=False, mark_write_through_pending=mark
        ),
        _components_tuple=(Component(),),
        resolve_node_handle=nodes.__getitem__,
        inc_lock_ref=lambda i: (
            locks.append(i) or SimpleNamespace(to_dec_params=lambda: i)
        ),
        dec_lock_ref=lambda i, params: unlocks.append((i, params)),
    )
    linker = _FakeLinker()
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)
    wrapper.offload_nodes([1, 2])
    assert locks == [2, 1]
    assert unlocks == []
    linker.completed_offloads.append(success)
    wrapper.commit_completed_offloads(wrapper.take_completed_offloads(1))
    assert unlocks == [(1, 1), (2, 2)]
    assert all(node.external_cache_stored == success for node in nodes.values())
    assert all(node.write_through_pending_id is None for node in nodes.values())


@pytest.mark.parametrize("stored,pending_ack", [(False, None), (True, None), (True, 99)])
@pytest.mark.parametrize("include_prompt_boundary", [False, True])
def test_sparse_offload_keeps_checkpoint_and_replay_windows(
    stored, pending_ack, include_prompt_boundary
):
    class Component:
        def build_external_linker_transfer(self, phase, node, keys):
            return PoolTransfer(
                name=PoolName.SWA,
                keys=[str(i) for i in range(20)],
                device_indices=torch.arange(20),
            )

    root = object()
    node = SimpleNamespace(
        id=1,
        key=list(range(20)),
        parent=root,
        external_cache_stored=stored,
        write_through_pending_id=pending_ack,
    )
    published = []
    unlocks = []

    def mark(ids, ack_id):
        published.extend(ids)
        for _ in ids:
            node.write_through_pending_id = ack_id
        return list(ids)

    cache = _cache_for_wrapper(
        sliding_window_size=2,
        tree_core=SimpleNamespace(
            root_node=root,
            enable_external_cache_linker=False,
            mark_write_through_pending=mark,
        ),
        _components_tuple=(Component(),),
        resolve_node_handle=lambda _: node,
        inc_lock_ref=lambda _: SimpleNamespace(to_dec_params=object),
        dec_lock_ref=lambda *args: unlocks.append(args),
    )
    linker = _FakeLinker()
    with envs.SGLANG_EXTERNAL_LINKER_SWA_RETENTION_INTERVAL.override(8):
        wrapper = UnifiedCacheLinkerWrapper(cache, linker)
    wrapper.offload_nodes(
        [1], replay_boundary=20, include_prompt_boundary=include_prompt_boundary
    )
    transfer = linker.queued_offloads[0][0]
    expected = [6, 7, 14, 15] + ([18, 19] if include_prompt_boundary else [])
    assert transfer.keys == [str(index) for index in expected]
    assert transfer.device_indices.tolist() == expected
    assert published == ([] if stored else [1])
    assert not unlocks
    linker.completed_offloads.append(False)
    wrapper.commit_completed_offloads(wrapper.take_completed_offloads(1))
    assert node.external_cache_stored == stored
    assert node.write_through_pending_id == pending_ack
    assert len(unlocks) == 1


@pytest.mark.parametrize("interval,expected_end", [(0, 33280), (32768, 32512)])
def test_sparse_checkpoint_survives_preinsert_release(interval, expected_end):
    freed = []
    component = object.__new__(SWAComponent)
    component.sliding_window_size = 256
    component.tree_core = SimpleNamespace(external_swa_retention_interval=interval)
    component.cache = SimpleNamespace(
        page_size=256,
        swa_retain_floor=lambda req: None,
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.arange(33792).reshape(1, -1)
        ),
        token_to_kv_pool_allocator=SimpleNamespace(
            free_swa_segment=lambda indices, start_pos: freed.append(
                (start_pos, start_pos + len(indices))
            )
        ),
    )
    req = SimpleNamespace(
        kv=SimpleNamespace(
            holds_kv=True,
            cache_protected_len=31744,
            swa_evicted_seqlen=31744,
            swa_dead_lo=lambda page_size: 31744,
            req_pool_idx=0,
        )
    )
    params = SimpleNamespace()
    component.free_out_of_window_slots(req, 33791, params)
    assert freed == [(31744, expected_end)]
    assert params.swa_evicted_seqlen == expected_end


def test_sparse_retention_matches_checkpoint_page_union():
    rng = random.Random(42)
    page = 256
    for case in range(2000):
        prompt = rng.randrange(1, 200) * page
        start = rng.randrange(prompt // page + 1) * page
        end = rng.randrange(start // page, prompt // page + 1) * page
        interval = rng.randrange(1, 40) * page
        window = rng.randrange(1, 2000)
        include_prompt = bool(case % 2)
        ranges = retained_swa_ranges(
            start,
            end,
            prompt_boundary=prompt,
            window=window,
            interval=interval,
            page_size=page,
            include_prompt_boundary=include_prompt,
        )
        boundaries = list(range(interval, prompt + 1, interval))
        if include_prompt:
            boundaries.append(prompt)
        rounded_window = (window + page - 1) // page * page
        expected = {
            pos
            for pos in range(start, end, page)
            if any(boundary - rounded_window <= pos < boundary for boundary in boundaries)
        }
        actual = {pos for left, right in ranges for pos in range(left, right, page)}
        assert actual == expected, (case, start, end, prompt, interval, window)
        assert all(start <= left < right <= end for left, right in ranges)
        assert all(left % page == right % page == 0 for left, right in ranges)
        assert all(a[1] < b[0] for a, b in zip(ranges, ranges[1:]))


@pytest.mark.parametrize(
    "chunked,previous,end,expected",
    [(True, 31744, 33792, True), (True, 33792, 35840, False),
     (False, 33792, 35840, True)],
)
def test_sparse_chunk_backup_only_when_crossing_checkpoint(
    chunked, previous, end, expected
):
    tree = SimpleNamespace(
        external_swa_sparse_retention=True,
        external_swa_retention_interval=32768,
        root_node=object(),
    )
    state = SimpleNamespace(
        params=SimpleNamespace(chunked=chunked, prev_prefix_len=previous),
        replay_boundary=end,
        target_node=SimpleNamespace(evicted=False),
    )
    assert UnifiedTreeCore._should_backup_after_insert(tree, state) is expected


def test_offload_batches_one_write_through_chain():
    class _Component:
        def __init__(self, name):
            self.name = name

        def build_external_linker_transfer(self, phase, node, keys):
            assert phase == LinkerTransferPhase.OFFLOAD
            return PoolTransfer(
                name=self.name,
                keys=[f"{self.name.value}-page-{node.id}"],
                device_indices=torch.tensor([node.id]),
            )

    linker = _FakeLinker()
    lock_params = object()
    nodes = {
        node_id: SimpleNamespace(
            id=node_id,
            external_cache_stored=False,
            write_through_pending_id=None,
        )
        for node_id in (7, 8)
    }
    locks = []
    unlocks = []

    def mark_pending(node_ids, ack_id):
        for node_id in node_ids:
            nodes[node_id].write_through_pending_id = ack_id
        return list(node_ids)

    cache = _cache_for_wrapper(
        tree_core=SimpleNamespace(
            enable_external_cache_linker=False,
            mark_write_through_pending=mark_pending,
        ),
        _components_tuple=(_Component(PoolName.KV), _Component(PoolName.SWA)),
        inc_lock_ref=lambda node_id: (
            locks.append(node_id) or SimpleNamespace(to_dec_params=lambda: lock_params)
        ),
        dec_lock_ref=lambda node_id, params: unlocks.append((node_id, params)),
        resolve_node_handle=nodes.__getitem__,
    )
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)

    wrapper.offload_nodes([7, 8])

    assert locks == [8, 7]
    assert len(linker.queued_offloads) == 1
    assert [transfer.name for transfer in linker.queued_offloads[0]] == [
        PoolName.KV,
        PoolName.SWA,
    ]
    assert linker.queued_offloads[0][0].keys == ["kv-page-7", "kv-page-8"]
    assert linker.queued_offloads[0][1].keys == ["swa-page-7", "swa-page-8"]
    assert all(
        transfer.device_indices.tolist() == [7, 8]
        for transfer in linker.queued_offloads[0]
    )
    assert all(node.external_cache_stored for node in nodes.values())

    linker.completed_offloads.append(True)
    wrapper.commit_completed_offloads(wrapper.take_completed_offloads(1))

    assert unlocks == [(7, lock_params), (8, lock_params)]
    assert all(node.write_through_pending_id is None for node in nodes.values())


def test_async_load_pins_node_until_completion():
    linker = _FakeLinker()
    lock_params = object()
    locks = []
    unlocks = []

    def inc_lock_ref(node):
        locks.append(node)
        return SimpleNamespace(to_dec_params=lambda: lock_params)

    node_id = 7
    cache = _cache_for_wrapper(
        inc_lock_ref=inc_lock_ref,
        dec_lock_ref=lambda node, params: unlocks.append((node, params)),
    )
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)

    wrapper._queue_load("rid", node_id, [object()])

    assert locks == [node_id]
    assert not unlocks

    linker.completed_loads.append(["rid"])
    wrapper.drain_loads(finish_count=1)

    assert unlocks == [(node_id, lock_params)]


def test_release_request_cancels_queued_load():
    linker = _FakeLinker()
    lock_params = object()
    unlocks = []
    cache = _cache_for_wrapper(
        dec_lock_ref=lambda node, params: unlocks.append((node, params))
    )
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)
    wrapper.hit_markers["rid"] = object()
    wrapper.pending_loads["rid"] = (7, lock_params)
    linker.queued_loads["rid"] = [object()]

    wrapper.release_request("rid")

    assert wrapper.hit_markers == {}
    assert wrapper.pending_loads == {}
    assert "rid" not in linker.queued_loads
    assert unlocks == [(7, lock_params)]


def test_failed_offload_rolls_back_split_fragments():
    class _Component:
        def build_external_linker_transfer(self, phase, node, keys):
            return PoolTransfer(
                name=PoolName.KV,
                keys=["page"],
                device_indices=torch.tensor([node.id]),
            )

    linker = _FakeLinker()
    lock_params = object()
    unlocks = []
    child = SimpleNamespace(
        id=7,
        external_cache_stored=False,
        write_through_pending_id=None,
    )
    parent = SimpleNamespace(
        id=8,
        external_cache_stored=False,
        write_through_pending_id=None,
    )
    nodes = {child.id: child, parent.id: parent}

    def mark_pending(node_ids, ack_id):
        for node_id in node_ids:
            nodes[node_id].write_through_pending_id = ack_id
        return list(node_ids)

    cache = _cache_for_wrapper(
        tree_core=SimpleNamespace(
            enable_external_cache_linker=False,
            mark_write_through_pending=mark_pending,
        ),
        _components_tuple=(_Component(),),
        inc_lock_ref=lambda node_id: SimpleNamespace(to_dec_params=lambda: lock_params),
        dec_lock_ref=lambda node_id, params: unlocks.append((node_id, params)),
        resolve_node_handle=nodes.__getitem__,
    )
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)
    wrapper.offload_nodes([child.id])

    parent.external_cache_stored = child.external_cache_stored
    parent.write_through_pending_id = child.write_through_pending_id
    wrapper.replace_pending_offload_node(child.id, child.id, [parent.id, child.id])
    linker.completed_offloads.append(False)
    wrapper.commit_completed_offloads(wrapper.take_completed_offloads(finish_count=1))

    assert not parent.external_cache_stored
    assert not child.external_cache_stored
    assert parent.write_through_pending_id is None
    assert child.write_through_pending_id is None
    assert unlocks == [(child.id, lock_params)]


def test_split_action_retargets_pending_external_offload():
    calls = []
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.linker = SimpleNamespace(
        replace_pending_offload_node=lambda *args: calls.append(("linker", *args))
    )
    cache._replace_pending_write_through_node = lambda *args: calls.append(
        ("hicache", *args)
    )
    action = ReplaceWriteThroughOnNodeSplit(
        ack_id=7,
        old_node_id=7,
        new_node_id=8,
        new_child_node_id=7,
    )

    cache._apply_cache_action(action)

    assert calls == [
        ("hicache", 7, 7, [8, 7]),
        ("linker", 7, 7, [8, 7]),
    ]


def test_reset_quiesces_backend_before_releasing_pending_locks():
    class _Component:
        def build_external_linker_transfer(self, phase, node, keys):
            return PoolTransfer(
                name=PoolName.KV,
                keys=["page"],
                device_indices=torch.tensor([node.id]),
            )

    events = []

    class _QuiescentFakeLinker(_FakeLinker):
        def reset(self):
            events.append("backend")
            super().reset()

    linker = _QuiescentFakeLinker()
    node = SimpleNamespace(
        id=7,
        external_cache_stored=False,
        write_through_pending_id=None,
    )
    cache = _cache_for_wrapper(
        tree_core=SimpleNamespace(
            enable_external_cache_linker=False,
            mark_write_through_pending=lambda node_ids, ack_id: (
                setattr(node, "write_through_pending_id", ack_id) or list(node_ids)
            ),
        ),
        _components_tuple=(_Component(),),
        inc_lock_ref=lambda node_id: SimpleNamespace(to_dec_params=object),
        dec_lock_ref=lambda node_id, params: events.append(("unlock", node_id)),
        resolve_node_handle=lambda node_id: node,
    )
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)
    wrapper._queue_load("rid", node.id, [object()])
    wrapper.offload_nodes([node.id])

    wrapper.reset()

    assert events == ["backend", ("unlock", node.id), ("unlock", node.id)]
    assert wrapper.pending_loads == {}
    assert wrapper.pending_offloads == []
    assert not node.external_cache_stored
    assert node.write_through_pending_id is None


def test_close_quiesces_backend_before_releasing_pending_loads():
    events = []

    class _ClosingFakeLinker(_FakeLinker):
        def close(self):
            events.append("backend")
            super().close()

    linker = _ClosingFakeLinker()
    cache = _cache_for_wrapper(
        dec_lock_ref=lambda node_id, params: events.append(("unlock", node_id))
    )
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)
    wrapper.pending_loads["rid"] = (7, object())

    wrapper.close()

    assert events == ["backend", ("unlock", 7)]
    assert linker.closed
    assert wrapper.pending_loads == {}


def test_check_hicache_events_commits_common_rank_results():
    committed = []
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.linker = SimpleNamespace(
        maybe_log_debug_stats=lambda: None,
        cache_linker=SimpleNamespace(num_completed_lookups=lambda: 2),
        drain_lookups=lambda count: committed.append(("lookup", count)),
        num_completed_loads=lambda: 1,
        drain_loads=lambda count: committed.append(("load", count)),
        num_completed_offloads=lambda: 3,
        take_completed_offloads=lambda count: [True] * count,
        commit_completed_offloads=committed.append,
    )

    reduce_calls = 0

    def reduce_to_common_state(value, op):
        nonlocal reduce_calls
        assert op == torch.distributed.ReduceOp.MIN
        reduce_calls += 1
        if reduce_calls == 1:
            value.copy_(torch.tensor([2, 1, 1]))
        else:
            value.fill_(0)

    cache._all_reduce_attn_groups = reduce_to_common_state

    cache.check_hicache_events()

    assert committed == [("lookup", 2), ("load", 1), [False]]


def test_component_commit_keeps_only_adopted_pages():
    mapping = _MappingRecorder()
    cache = _cache_for_wrapper(
        page_size=2,
        token_to_kv_pool_allocator=SimpleNamespace(
            set_full_to_swa_mapping=mapping.set_full_to_swa_mapping
        ),
    )
    wrapper = UnifiedCacheLinkerWrapper(cache, _FakeLinker())
    full_component = FullComponent.__new__(FullComponent)
    full_component.cache = cache
    full_component.component_type = ComponentType.FULL
    swa_component = SWAComponent.__new__(SWAComponent)
    swa_component.cache = cache
    swa_component.component_type = ComponentType.SWA
    full = PoolTransfer(
        name=PoolName.KV,
        keys=["a", "b", "c", "d"],
        device_indices=torch.tensor([100, 101, 102, 103, 104, 105, 106, 107]),
    )
    canonical_tail = torch.tensor([10, 11, 102, 103, 14, 15, 106, 107])
    swa = PoolTransfer(
        name=PoolName.SWA,
        keys=["a", "b", "c", "d"],
        device_indices=torch.tensor([200, 201, 202, 203, 204, 205, 206, 207]),
    )
    insert_result = InsertResult(
        prefix_len=0,
        adopted_ranges={
            ComponentType.FULL: [(2, 4), (6, 8)],
            ComponentType.SWA: [(2, 4), (6, 8)],
        },
    )

    filtered = wrapper._update_load(
        ExternalLinkerLoadPhase.COMMIT,
        SimpleNamespace(),
        [(full_component, full), (swa_component, swa)],
        prefix_len=8,
        insert_result=insert_result,
        canonical_full=canonical_tail,
    )

    assert filtered == [full, swa]
    assert full.keys == ["b", "d"]
    assert full.device_indices.tolist() == [102, 103, 106, 107]
    assert swa.keys == ["b", "d"]
    assert swa.device_indices.tolist() == [202, 203, 206, 207]
    mapped_full, mapped_swa = mapping.mapping[0]
    assert mapped_full.tolist() == [102, 103, 106, 107]
    assert mapped_swa.tolist() == [202, 203, 206, 207]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
