# Derived from Psi0 (https://github.com/physical-superintelligence-lab/Psi0, Apache License 2.0) and modified; see NOTICE.
import torch
from torch.utils.data import Sampler
from accelerate import Accelerator
import math
import os
import random
from dataclasses import dataclass
from psi.utils import initialize_overwatch
overwatch = initialize_overwatch(__name__)

class BatchMixtureSampler(Sampler):
    """
    Multi-node safe dataset mixture sampler.
    Example: datasets = [ds1, ds2], ratio = [4, 1]
    """
    def __init__(self, dataset_lens, mixture_ratios, num_samples_per_epoch, batch_size, seed=42):
        self.dataset_lens = dataset_lens
        self.weights = torch.tensor(mixture_ratios, dtype=torch.double)
        self.weights /= self.weights.sum()
        self.num_samples = num_samples_per_epoch
        self.batch_size = batch_size
        self.seed = seed

        # Distributed settings
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        # each rank gets this many samples
        self.num_samples_rank = math.ceil(self.num_samples / self.world_size)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Sample dataset IDs for all ranks
        all_dataset_ids = torch.multinomial(
            self.weights,
            self.num_samples_rank * self.world_size,
            replacement=True,
            generator=g,
        )

        # Shard by rank
        dataset_ids = all_dataset_ids[self.rank::self.world_size]

        # Vectorized sampling of indices within each dataset for efficiency
        dataset_ids_tensor = dataset_ids if isinstance(dataset_ids, torch.Tensor) else torch.tensor(dataset_ids)
        dataset_lens_tensor = torch.tensor(self.dataset_lens)
        lens_for_ids = dataset_lens_tensor[dataset_ids_tensor]
        rand_indices = torch.floor(torch.rand(len(dataset_ids_tensor), generator=g) * lens_for_ids).long()
        indices = list(zip(dataset_ids_tensor.tolist(), rand_indices.tolist()))
        
        # Group indices into batches
        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        return iter(batches)

    def __len__(self):
        return math.ceil(self.num_samples_rank / self.batch_size)

@dataclass
class DatasetSpec:
    # dataset: torch.utils.data.Dataset
    dataset_length: int
    prob: float                     # mixture ratio
    image_size: tuple[int, int]     # e.g. 224, 448
    tokens_per_image: int           # (image_size / patch)^2

class TokenMixtureSampler(Sampler):
    """
    Multi-node safe dataset mixture sampler that samples from datasets with different image resolutoion while
    keeping the total number of tokens per batch (approximately) fixed.
    """
    def __init__(
        self,
        specs: list[DatasetSpec],
        tokens_per_batch: int, # per rank
        num_batches_per_rank: int,
        seed: int = 0,
    ):  
        self.specs = specs
        self.tokens_per_batch = tokens_per_batch
        self.num_batches_per_rank = num_batches_per_rank
        self.seed = seed

        self.probs = [s.prob for s in specs]
        self.epoch = 0

        # Distributed settings
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

    def set_epoch(self, epoch):
        self.epoch = epoch
        # overwatch.info(f"TokenMixtureSampler set to epoch {epoch}")

    def __iter__(self):
        # assert self.epoch is not None, "TokenMixtureSampler epoch not set. Please call set_epoch(epoch) before iterating."

        # For each global batch index, synchronize dataset_id selection across all ranks
        for batch_idx in range(self.num_batches_per_rank * self.world_size):
            if batch_idx % self.world_size != self.rank:
                continue
            # Use a deterministic RNG for this global! batch to pick dataset_id and indices
            batch_seed = self.seed + self.epoch + batch_idx #  // self.world_size
            local_rng = random.Random(batch_seed)
            dataset_id = local_rng.choices(range(len(self.specs)), weights=self.probs, k=1)[0]
            # print(f"[TokenMixtureSampler] rank={self.rank} seed={batch_seed} dataset_id={dataset_id}")
            spec = self.specs[dataset_id]
            batch_size = max(1, self.tokens_per_batch // spec.tokens_per_image)
            local_batch_rng = random.Random(self.seed + self.epoch + batch_idx)
            indices = [
                (dataset_id, local_batch_rng.randrange(spec.dataset_length))
                for _ in range(batch_size)
            ]
            # overwatch.info(f"rank {overwatch.rank()}:  {self.epoch} batch {batch_idx} dataset_id {dataset_id} batch_size {batch_size}")
            yield indices

    def __len__(self):
        return self.num_batches_per_rank