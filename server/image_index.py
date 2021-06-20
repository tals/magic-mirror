import json
import os
import time
from pathlib import Path
from threading import Thread
from typing import *

import numpy as np
import torch
import torchvision.transforms.functional as TF
from loguru import logger
from PIL import Image
from pydantic import BaseModel
from tqdm.auto import tqdm


class QueryResultEntry(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    path: Path
    score: float
    idx: int
    emb: torch.Tensor

    def get_score_from(self, emb: torch.Tensor):
        x = self.emb @ emb.float().squeeze()
        x = x.item()

        return x
        
class PendingImage(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    emb: torch.Tensor
    owner_id: int
    img: np.ndarray

def torchify(x):
    if isinstance(x, torch.Tensor):
        return x

    return torch.from_numpy(x)

class OpenImageIndex:
    def __init__(self):
        self.path = Path("/home/tal/datasets/open-images-validation")
        flavor = self.path.name.split("-")[-1]
        print("Loading data")
        self.filenames = json.loads((self.path / "index/filenames.json").read_text())
        self.keys = [f"{flavor}/{os.path.splitext(x)[0]}" for x in self.filenames]
        self.pool = torch.load(self.path / "index/clip_codes_normed.pt").float().cuda()

    @torch.no_grad()
    def query(self, emb, topk=5) -> List[QueryResultEntry]:
        emb = torchify(emb)
        if emb.ndim == 1:
            emb = emb[None]

        x = self.pool @ emb.float().T
        x = x.squeeze().topk(topk, dim=0)
        I, D = x.indices.cpu().numpy(), x.values.cpu().numpy()
        # D = (D * 100).astype(np.uint8)

        keys = [self.keys[x.item()] for x in I.squeeze()]
        results = []
        for i, score in zip(I, D):
            results.append(QueryResultEntry(
                path=self.path / self.filenames[i],
                score=score,
                emb=self.pool[i],
                idx=i,
            ))

        return results

    def add(self, *args, **kwargs):
        logger.warning("Unimplemented")


SHARD_SIZE = 100_000
DATASET_PATH = Path("~/datasets/magic_mirror_the_archive").expanduser()
# DATASET_PATH = Path("/tmp/stupid")
DATASET_PATH.mkdir(exist_ok=True, parents=True)


class UserImageIndex:
    """Shitty index lol"""
    pending: List[PendingImage]
    path: Path
    idxs: torch.IntTensor
    
    def __init__(self, path=DATASET_PATH, autoflush=True, device=torch.device("cuda")):
        self.path = path
        self.pending = []
        self.running = True
        self.pool_size = 0
        self.device = device
        self.pool = torch.zeros(1_000_000, 512).to(device)
        disk_tiles = list(self.path.glob("tile_*.pt"))
        disk_tiles.sort()

        for tile_idx, dt in tqdm(enumerate(disk_tiles), desc="Loading"):
            try:
                data = torch.load(dt)
                start = tile_idx * SHARD_SIZE
                self.pool_size += len(data)
                self.pool[start:start+len(data)] = data.to(self.device)
            except:
                logger.exception("Got an exception yo")
                # TODO: reindex tile
                break
        # keep track of where its safe to query from
        self.safe_idx = self.pool_size
        self.shuffle()

        self.worker_thread = Thread(target=self._write_worker, daemon=True)
        if autoflush:
            self.worker_thread.start()

    def _write_worker(self):
        while self.running:
            time.sleep(5)
            self._flush()

    def shuffle(self):
        k = 1000
        query_edge = max(100, self.safe_idx - 10 * 60)
        query_edge = min(query_edge, self.safe_idx)
        # print("query_edge", query_edge)
        perm = torch.randperm(query_edge)
        self.idxs = perm[:k]
        # print(self.idxs)

    def _flush(self):
        if not self.pending:
            return

        dirty = set()
        pending: List[PendingImage] = self.pending
        self.pending = []

        # print("Flushing", len(pending))
        for pi in pending:
            idx = self.pool_size
            self.pool_size += 1

            # write to pool
            self.pool[idx] = torchify(pi.emb).to(self.device).float()
            dirty.add(idx // SHARD_SIZE)

            key = f"{idx:08}.jpg"

            img = Image.fromarray(pi.img, "RGB")
            img.save(self.path / key)

        # for d in tqdm(dirty, desc="Saving tiles"):
        for d in dirty:
            self._write_dirty(d)

        self.safe_idx = self.pool_size

    def _write_dirty(self, tile_idx):
        start = tile_idx * SHARD_SIZE
        end = min(self.pool_size, (tile_idx + 1) * SHARD_SIZE)

        # write to a temp to avoid corrupting data
        target_file = self.path / f"tile_{tile_idx:08}.pt"
        target_file_tmp: Path = target_file.with_suffix(".tmp")
        torch.save(self.pool[start:end].cpu(), target_file_tmp)
        target_file_tmp.rename(target_file)

        
    @torch.no_grad()
    def query(self, emb, topk=5) -> List[QueryResultEntry]:
        emb = torchify(emb)
        if emb.ndim == 1:
            emb = emb[None]

        query_edge = max(100, self.safe_idx - 10 * 60)
        query_edge = min(query_edge, self.safe_idx)
        x = self.pool[:query_edge] @ emb.float().T
        x = x[self.idxs]
        x = x.topk(min(topk, len(x)), dim=0)
        I, D = x.indices.cpu().numpy(), x.values.cpu().numpy()
        # D = (D * 100).astype(np.uint8)
        if len(I) == 0:
            return []

        results = []
        for idx, score in zip(I, D):
            idx = self.idxs[int(idx)]
            key = f"{idx:08}.jpg"
            results.append(QueryResultEntry(
                path=self.path / key,
                score=score,
                emb=self.pool[idx],
                idx=idx,
            ))

        # print(results[0].idx, results[0].score)
        return results

    def add(self, img, emb):
        assert emb.ndim == 1 or emb.shape[0] == 1
        emb = emb.squeeze()

        self.pending.append(PendingImage(
            emb=emb,
            img=img,
            owner_id=1,
        ))

