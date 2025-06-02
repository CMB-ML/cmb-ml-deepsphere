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

from enum import Enum
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

    def all_reduce_tracker(self, tracker):
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


@contextmanager
def timer(model_tracker: ModelTracker, tracker: str):
    start_time = time.perf_counter()
    try:
        yield start_time
    finally:
        elapsed_time = time.perf_counter() - start_time
        model_tracker.update_tracker(tracker, elapsed_time)


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
