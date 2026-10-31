"""Captured CUDA-graph regions a replay can switch off (cudaGraphNodeSetEnabled).

A region records the kernel / memcpy / memset nodes its ops added to the graph being
captured on the current stream. A graph backend that keeps its raw graphs calls
``apply`` before each replay: the regions are enabled iff ``SKIPPABLE.enabled``, and
a disabled node runs as an empty node, keeping its dependencies.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Dict, List

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
    def __init__(self) -> None:
        self.enabled = True
        # raw graph handle -> the region nodes captured into it
        self._nodes: Dict[int, List] = {}
        # raw graph exec handle -> whether its region nodes are enabled now
        self._state: Dict[int, bool] = {}
        self.num_toggles = 0

    @contextlib.contextmanager
    def region(self):
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
        self._nodes.setdefault(int(graph), []).extend(added)

    def apply(self, graph: torch.cuda.CUDAGraph) -> None:
        if not self._nodes:
            return
        try:
            nodes = self._nodes.get(int(graph.raw_cuda_graph()))
        except RuntimeError:
            return
        if not nodes:
            return
        graph_exec = int(graph.raw_cuda_graph_exec())
        want = self.enabled
        if self._state.get(graph_exec, True) == want:
            return
        for node in nodes:
            checkCudaErrors(rt.cudaGraphNodeSetEnabled(graph_exec, node, int(want)))
        self._state[graph_exec] = want
        self.num_toggles += 1
        if self.num_toggles == 1:
            logger.info(
                "Captured graph regions switched %s (%d nodes) for this replay.",
                "on" if want else "off",
                len(nodes),
            )

    def forget(self, graph: torch.cuda.CUDAGraph) -> None:
        with contextlib.suppress(RuntimeError):
            self._nodes.pop(int(graph.raw_cuda_graph()), None)
        with contextlib.suppress(RuntimeError):
            self._state.pop(int(graph.raw_cuda_graph_exec()), None)


SKIPPABLE = SkippableGraphNodes()
