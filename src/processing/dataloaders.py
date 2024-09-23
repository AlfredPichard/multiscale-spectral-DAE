import lmdb
import numpy as np
import torch
from tqdm import tqdm

from src.processing.audio_example import AudioExample


def collate_fn(batch):
    waveform = np.stack([b["waveform"] for b in batch])
    z = waveform = np.stack([b["z"] for b in batch])

    waveform, z = torch.from_numpy(waveform), torch.from_numpy(z)
    return waveform, z


class CachedSimpleDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        path,
        keys=["waveform", "metadata"],
        max_samples=None,
        num_sequential=20,
        recache_every=None,
    ) -> None:
        super().__init__()

        self.num_sequential = num_sequential
        self.buffer_keys = keys
        self.max_samples = max_samples
        self.recache_every = recache_every
        self.recache_counter = 0

        self.env = lmdb.open(
            path, lock=False, readonly=True, readahead=True, map_async=False
        )

        with self.env.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

        self.indexes = list(range(len(self.keys)))

        if self.max_samples is not None and self.max_samples < len(self.indexes):
            self.indexes = np.random.choice(
                self.indexes, self.max_samples, replace=False
            )

        self.cached = False

        if self.recache_every is not None:
            self.build_cache()

    def __len__(self):
        return len(self.indexes)

    def build_cache(self):
        self.cached = False
        self.cache = []

        self.indexes = list(range(len(self.keys)))

        if self.max_samples is not None:
            self.indexes_start = np.random.choice(
                self.indexes[: -self.num_sequential],
                self.max_samples // self.num_sequential,
                replace=False,
            )

            self.indexes = [
                start + i
                for start in self.indexes_start
                for i in range(self.num_sequential)
            ]

        for i in tqdm(range(len(self.indexes))):
            self.cache.append(self.__getitem__(i))

        self.cached = True

    def __getitem__(self, index):
        if self.cached == True:
            self.recache_counter += 1
            if (
                self.recache_every is not None
                and self.recache_counter == self.recache_every
            ):
                self.build_cache()
                self.recache_counter = 0

            return self.cache[index]

        index = self.indexes[index]

        with self.env.begin() as txn:
            ae = AudioExample(txn.get(self.keys[index]))

        out = {}
        for key in self.buffer_keys:
            if key == "metadata":
                out[key] = ae.get_metadata()
            elif key == "midi":
                out[key] = ae.get_midi()
            else:
                try:
                    out[key] = ae.get(key)
                except:
                    print("key: ", key, " not found")

        return out
