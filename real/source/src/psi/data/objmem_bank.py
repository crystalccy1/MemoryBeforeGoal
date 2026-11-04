"""Reader for the SAM3-v2 memory bank of one real SONIC task (built by tools/build_objmem_bank.py).

Layout (per recorder source episode e):
  native/episode_{e:06d}.safetensors   features [F,77,2048] bf16 (Qwen3-VL vision-merger 7x11 grid), raw_frame_indices [F]
  objects/episode_{e:06d}.safetensors  sam3_raw_frame_indices / sam3_frame_mask [F,7] (causal SAM3 event selection,
                                       current frame included), object_features [F,S,2048] bf16 (mask-pooled native
                                       tokens), object_geometry [F,S,8], object_valid [F,S]; slot k = SAM3 prompt k.
memory_for_frame(e, t) returns the 539 visual tokens + 7*S object tokens for decision frame t, frame-major and
right padded, in the same format as EmbodiedDyna's Sam3ObjectNativeTrainingBank.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

SCHEMA = "psi0-objmem-bank/v1"
MAX_FRAMES = 7
TOKENS_PER_FRAME = 77
FEATURE_DIM = 2048
GEOMETRY_DIM = 8
GEOMETRY_FIELDS = ("x0", "y0", "x1", "y1", "cx", "cy", "area_fraction", "visible")


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ObjMemBank:
    def __init__(self, root, *, expected_manifest_sha256: str, verify_hashes: bool = False) -> None:
        self.root = Path(root).resolve(strict=True)
        manifest_path = self.root / "manifest.json"
        digest = sha256_file(manifest_path)
        ready = json.loads((self.root / "READY").read_text())
        if digest != expected_manifest_sha256 or ready.get("manifest_sha256") != digest:
            raise ValueError("objmem bank manifest is not the pinned, completed bank")
        self.manifest = json.loads(manifest_path.read_text())
        m = self.manifest
        if (m.get("schema_version") != SCHEMA or m.get("status") != "complete"
                or m.get("geometry_fields") != list(GEOMETRY_FIELDS) or m.get("max_frames") != MAX_FRAMES):
            raise ValueError("unsupported objmem bank contract")
        self.slots = int(m["object_slots"])
        self.prompts = list(m["prompts"])
        if len(self.prompts) != self.slots:
            raise ValueError("one SAM3 prompt per object slot is required")
        self.records = {int(row["source_episode"]): row for row in m["episodes"]}
        self.data_root = self.root
        for row in self.records.values():
            for key in ("native", "objects"):
                path = self._path(row[key]["path"])
                if path.stat().st_size != row[key]["bytes"]:
                    raise ValueError(f"objmem bank file changed size: {path}")
                if verify_hashes and sha256_file(path) != row[key]["sha256"]:
                    raise ValueError(f"objmem bank file hash mismatch: {path}")
        self._pid = None
        self._handles: dict = {}
        self._small: dict = {}
        self._preloaded: dict = {}

    def preload(self) -> int:
        """Read every shard once (sequential, sha-checked) into memory before dataloader workers fork.

        Workers then share these tensors copy-on-write; per-sample random reads of cold NFS pages
        stalled the dataloader ~10x in smoke 282812. Returns the number of bytes loaded."""
        from safetensors.torch import load as load_bytes
        loaded = {}
        total = 0
        for episode, row in self.records.items():
            tensors = {}
            for key in ("native", "objects"):
                data = self._path(row[key]["path"]).read_bytes()
                if len(data) != row[key]["bytes"] or hashlib.sha256(data).hexdigest() != row[key]["sha256"]:
                    raise ValueError(f"objmem bank shard mismatch: episode {episode} {key}")
                tensors[key] = load_bytes(data)
                total += len(data)
                del data
            loaded[episode] = tensors
        self._preloaded = loaded
        return total

    def _path(self, relative: str) -> Path:
        path = (self.data_root / relative).resolve(strict=True)
        path.relative_to(self.data_root)
        return path

    def _handle(self, episode: int, key: str):
        if self._pid != os.getpid():  # dataloader workers must not share file handles across fork
            self._pid, self._handles, self._small = os.getpid(), {}, {}
        handle = self._handles.get((episode, key))
        if handle is None:
            handle = safe_open(str(self._path(self.records[episode][key]["path"])), framework="pt", device="cpu")
            self._handles[episode, key] = handle
        return handle

    def _selection(self, episode: int):
        self._handle(episode, "objects")  # resets per-process caches after fork
        cached = self._small.get(episode)
        if cached is None:
            handle = self._handles[episode, "objects"]
            cached = {name: handle.get_tensor(name) for name in (
                "sam3_raw_frame_indices", "sam3_frame_mask", "object_geometry", "object_valid")}
            self._small[episode] = cached
        return cached

    def memory_for_frame(self, episode: int, frame: int) -> dict[str, torch.Tensor]:
        row = self.records[episode]
        if not 0 <= frame < int(row["raw_frames"]):
            raise ValueError("decision frame lies outside the episode")
        pre = self._preloaded.get(episode)
        if pre is not None:
            small, native, objects = pre["objects"], pre["native"]["features"], pre["objects"]["object_features"]
        else:
            small = self._selection(episode)
            native = self._handle(episode, "native").get_slice("features")
            objects = self._handle(episode, "objects").get_slice("object_features")
        selected = small["sam3_raw_frame_indices"][frame][small["sam3_frame_mask"][frame]]
        count = len(selected)
        if (count != min(MAX_FRAMES, frame + 1) or int(selected[-1]) != frame or int(selected[0]) < 0
                or bool(torch.any(selected[1:] <= selected[:-1]))):
            raise ValueError(f"invalid causal selection at episode {episode} frame {frame}: {selected.tolist()}")
        budget, obudget = MAX_FRAMES * TOKENS_PER_FRAME, MAX_FRAMES * self.slots
        features = torch.zeros((budget, FEATURE_DIM), dtype=torch.bfloat16)
        mask = torch.zeros(budget, dtype=torch.bool)
        raw = torch.full((budget,), -1, dtype=torch.int64)
        spatial = torch.full((budget,), -1, dtype=torch.int64)
        ofeatures = torch.zeros((obudget, FEATURE_DIM), dtype=torch.bfloat16)
        ogeometry = torch.zeros((obudget, GEOMETRY_DIM), dtype=torch.float32)
        omask = torch.zeros(obudget, dtype=torch.bool)
        oraw = torch.full((obudget,), -1, dtype=torch.int64)
        oslots = torch.full((obudget,), -1, dtype=torch.int64)
        for i, t in enumerate(selected.tolist()):
            a = i * TOKENS_PER_FRAME
            features[a:a + TOKENS_PER_FRAME] = native[t:t + 1][0]
            mask[a:a + TOKENS_PER_FRAME] = True
            raw[a:a + TOKENS_PER_FRAME] = t
            spatial[a:a + TOKENS_PER_FRAME] = torch.arange(TOKENS_PER_FRAME)
            b = i * self.slots
            valid = small["object_valid"][t]
            ofeatures[b:b + self.slots] = objects[t:t + 1][0]
            ogeometry[b:b + self.slots] = small["object_geometry"][t]
            omask[b:b + self.slots] = valid
            oraw[b:b + self.slots] = t
            oslots[b:b + self.slots] = torch.arange(self.slots)
        ofeatures[~omask] = 0
        ogeometry[~omask] = 0
        oraw[~omask] = -1
        oslots[~omask] = -1
        return {"memory_features": features, "memory_mask": mask, "memory_raw_frame_indices": raw,
                "memory_spatial_indices": spatial, "memory_object_features": ofeatures,
                "memory_object_geometry": ogeometry, "memory_object_mask": omask,
                "memory_object_raw_frame_indices": oraw, "memory_object_slots": oslots}
