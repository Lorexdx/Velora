import json
from pathlib import Path

import numpy as np
import torch
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import Dataset

from utils.console import (
    format_dataset_cache_loaded,
    format_dataset_cache_saved,
    format_dataset_tokenized
)


def _file_metadata(path):
    path = Path(path)
    stat = path.stat()

    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns
    }


class VeloraDataset(Dataset):
    def __init__(
        self,
        corpus_path,
        vocab_path,
        merges_path,
        seq_length=128,
        max_chars=None,
        token_cache_path=None,
        start_idx=0,
        end_idx=None,
        stride=1
    ):
        self.seq_length = seq_length
        self.start_idx = start_idx
        self.stride = stride
        self.cache_metadata = {
            "corpus": _file_metadata(corpus_path),
            "vocab": _file_metadata(vocab_path),
            "merges": _file_metadata(merges_path),
            "max_chars": max_chars
        }

        if token_cache_path:
            cached_tokens = self._load_token_cache(token_cache_path)

            if cached_tokens is not None:
                self.tokens = cached_tokens
                self.end_idx = end_idx if end_idx is not None else len(self.tokens)
                print(format_dataset_cache_loaded(token_cache_path, len(self.tokens)))
                return

        tokenizer = ByteLevelBPETokenizer(vocab_path, merges_path)

        with open(corpus_path, "r", encoding="utf-8") as file:
            text = file.read(max_chars) if max_chars else file.read()

        token_ids = tokenizer.encode(text).ids
        print(format_dataset_tokenized(len(token_ids)))

        if token_cache_path:
            self._save_token_cache(token_cache_path, token_ids)
            self.tokens = self._load_token_cache(token_cache_path)
        else:
            self.tokens = np.asarray(token_ids, dtype=np.int32)

        self.end_idx = end_idx if end_idx is not None else len(self.tokens)

    def __len__(self):
        available_tokens = self.end_idx - self.start_idx

        if available_tokens < self.seq_length + 1:
            return 0

        return 1 + (available_tokens - self.seq_length - 1) // self.stride

    def __getitem__(self, idx):
        idx = self.start_idx + idx * self.stride
        chunk = self.tokens[idx:idx + self.seq_length + 1].astype(np.int64)
        chunk = torch.from_numpy(np.asarray(chunk))

        input_ids = chunk[:-1]
        target_ids = chunk[1:]

        return input_ids, target_ids

    def _metadata_path(self, token_cache_path):
        return Path(f"{token_cache_path}.metadata.json")

    def _load_token_cache(self, token_cache_path):
        token_cache_path = Path(token_cache_path)
        metadata_path = self._metadata_path(token_cache_path)

        if not token_cache_path.exists() or not metadata_path.exists():
            return None

        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        if metadata != self.cache_metadata:
            return None

        return np.load(token_cache_path, mmap_mode="r")

    def _save_token_cache(self, token_cache_path, token_ids):
        token_cache_path = Path(token_cache_path)
        metadata_path = self._metadata_path(token_cache_path)
        token_cache_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(token_cache_path, np.asarray(token_ids, dtype=np.int32))

        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(self.cache_metadata, file, indent=2)

        print(format_dataset_cache_saved(token_cache_path, len(token_ids)))
