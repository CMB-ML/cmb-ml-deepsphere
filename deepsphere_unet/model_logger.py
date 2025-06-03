"""
Handles the model logging logic.
Tracks the following:
- Average batch time
- Average data loading time
- Epoch time
- Training loss
- Valid loss

Inspired by AverageMeter implementation from: https://github.com/pytorch/examples/blob/main/imagenet/main.py#L400
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class ValueCounter:
    value: float = 0
    count: int = 0
    running_avg: float = 0
    running_sum: float = 0


class ModelTracker:
    def __init__(self):
        self.to_track = [
            "batch_process_time",
            "data_load_time",
            "epoch_process_time",
            "train_loss",
            "val_loss",
        ]

        self.trackers = {tracker: ValueCounter() for tracker in self.to_track}

    def reset_tracker(self, tracker: str):
        self.trackers[tracker] = ValueCounter()

    def reset_all(self):
        self.trackers = {tracker: ValueCounter() for tracker in self.to_track}

    def update_tracker(self, tracker, value, count=1):
        self.trackers[tracker].value = value
        self.trackers[tracker].count += count
        self.trackers[tracker].running_sum += value
        self.trackers[tracker].running_avg = (
            self.trackers[tracker].running_sum / self.trackers[tracker].count
        )

    def allreduce_tracker(self, tracker):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        total = torch.tensor(
            [self.trackers[tracker].running_sum, self.trackers[tracker].count],
            dtype=torch.float32,
            device=device,
        )
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.trackers[tracker].running_sum, self.trackers[tracker].count = (
            total.tolist()
        )
        self.trackers[tracker].running_avg = (
            self.trackers[tracker].running_sum / self.trackers[tracker].count
        )

    def allreduce_all_trackers(self, include_val=True):
        for tracker in self.to_track:
            if not include_val:
                if tracker == "val_loss":
                    continue
            self.allreduce_tracker(tracker)

        if not include_val:
            self.combined_loss = self.trackers["train_loss"].running_avg
        else:
            self.combined_loss = (
                0.8 * self.trackers["val_loss"].running_avg
                + 0.2 * self.trackers["train_loss"].running_avg
            )

    def get_combined_loss(self):
        return self.combined_loss

    def get_log_values(self):
        log_vals = [tracker.running_avg for tracker in self.trackers.values()] + [
            self.get_combined_loss()
        ]
        return log_vals


@contextmanager
def timer(model_tracker: ModelTracker, tracker: str):
    start_time = time.perf_counter()
    try:
        yield start_time
    finally:
        elapsed_time = time.perf_counter() - start_time
        model_tracker.update_tracker(tracker, elapsed_time)


# TODO: utilize this memory stat function somewhere
def get_mem_stats(device=None):
    mem = torch.cuda.memory_stats(device)
    props = torch.cuda.get_device_properties(device)
    return {
        "cuda_device": device,
        "gpu_name": props.name,
        "total_gb": 1e-9 * props.total_memory,
        "curr_alloc_gb": 1e-9 * mem["allocated_bytes.all.current"],
        "peak_alloc_gb": 1e-9 * mem["allocated_bytes.all.peak"],
        "curr_resv_gb": 1e-9 * mem["reserved_bytes.all.current"],
        "peak_resv_gb": 1e-9 * mem["reserved_bytes.all.peak"],
    }
