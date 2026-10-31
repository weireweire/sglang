"""Captured CUDA-graph regions a replay can switch off (cudaGraphNodeSetEnabled).

A region records the kernel / memcpy / memset nodes its ops added to the graph being
captured on the current stream, under a group name. A graph backend that keeps its raw
graphs calls ``apply`` before each replay: each group's regions are enabled iff
``SKIPPABLE.enabled[group]`` (default on), and a disabled node runs as an empty node,
keeping its dependencies.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Dict, List, Tuple

import torch

from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.cuda_utils import (
    checkCudaErrors,
    rt,
)

logger = logging.getLogger(__name__)


def _graph_nodes(graph) -> List:
    _, n = checkCudaErrors(rt.cudaGraphGetNodes(graph, 0))
    nodes, _ = checkCudaErrors(rt.cudaGraphGetNodes(graph, n))
    return list(nodes or [])


class SkippableGraphNodes:
    DEFAULT = "epilogue"

    def __init__(self) -> None:
        self.enabled: Dict[str, bool] = {}
        # (raw graph handle, group) -> the region nodes captured into it
        self._nodes: Dict[Tuple[int, str], List] = {}
        # (raw graph exec handle, group) -> whether its region nodes are enabled now
        self._state: Dict[Tuple[int, str], bool] = {}
        self.num_toggles = 0

    @contextlib.contextmanager
    def region(self, group: str = DEFAULT):
        """Record the nodes the ops inside add to the current stream's capture."""
        if rt is None or not torch.cuda.is_current_stream_capturing():
            yield
            return
        stream = torch.cuda.current_stream().cuda_stream
        try:
            graph = checkCudaErrors(rt.cudaStreamGetCaptureInfo(stream))[2]
            before = {int(n) for n in _graph_nodes(graph)}
        except RuntimeError:
            logger.warning("Cannot read the capture graph; the region stays on.")
            yield
            return
        yield
        kinds = (
            rt.cudaGraphNodeType.cudaGraphNodeTypeKernel,
            rt.cudaGraphNodeType.cudaGraphNodeTypeMemcpy,
            rt.cudaGraphNodeType.cudaGraphNodeTypeMemset,
        )
        added = [
            n
            for n in _graph_nodes(graph)
            if int(n) not in before
            and checkCudaErrors(rt.cudaGraphNodeGetType(n)) in kinds
        ]
        self._nodes.setdefault((int(graph), group), []).extend(added)

    def apply(self, graph: torch.cuda.CUDAGraph) -> None:
        if not self._nodes:
            return
        try:
            raw = int(graph.raw_cuda_graph())
            graph_exec = int(graph.raw_cuda_graph_exec())
        except RuntimeError:
            return
        for (graph_raw, group), nodes in self._nodes.items():
            if graph_raw != raw or not nodes:
                continue
            want = self.enabled.get(group, True)
            if self._state.get((graph_exec, group), True) == want:
                continue
            for node in nodes:
                checkCudaErrors(rt.cudaGraphNodeSetEnabled(graph_exec, node, int(want)))
            self._state[(graph_exec, group)] = want
            self.num_toggles += 1
            if self.num_toggles <= 2:
                logger.info(
                    "Captured graph region %s switched %s (%d nodes) for this replay.",
                    group,
                    "on" if want else "off",
                    len(nodes),
                )

    def forget(self, graph: torch.cuda.CUDAGraph) -> None:
        with contextlib.suppress(RuntimeError):
            raw = int(graph.raw_cuda_graph())
            for key in [k for k in self._nodes if k[0] == raw]:
                del self._nodes[key]
        with contextlib.suppress(RuntimeError):
            graph_exec = int(graph.raw_cuda_graph_exec())
            for key in [k for k in self._state if k[0] == graph_exec]:
                del self._state[key]

SKIPPABLE = SkippableGraphNodes()
