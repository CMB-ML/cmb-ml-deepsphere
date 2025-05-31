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
        _tracker = self.trackers[tracker]

        _tracker.value = value
        _tracker.count += count
        _tracker.running_sum += value * count
        _tracker.running_avg = _tracker.running_sum / _tracker.count

        self.trackers[tracker] = _tracker

    def all_reduce_tracker(self, tracker):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        cur_tracker = self.trackers[tracker]
        total = torch.tensor(
            [cur_tracker.running_sum, cur_tracker.count],
            dtype=torch.float32,
            device=device,
        )
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        cur_tracker.running_sum, cur_tracker.count = total.tolist()
        cur_tracker.running_avg = cur_tracker.running_sum / cur_tracker.count
        self.trackers[tracker] = cur_tracker


@contextmanager
def timer(model_tracker: ModelTracker, tracker: str):
    start_time = time.perf_counter()
    try:
        yield start_time
    finally:
        elapsed_time = time.perf_counter() - start_time
        model_tracker.update_tracker(tracker, elapsed_time)
