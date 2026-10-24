"""Subliminal-learning replication library. Importing it sets cluster env defaults."""
import os

_PROJ = "/scratch/users/jkli/subliminal-learning"
for _key, _value in {
    "HF_HOME": "/scratch/users/jkli/hf",
    "HF_HUB_OFFLINE": "1",
    "VLLM_CACHE_ROOT": f"{_PROJ}/.cache/vllm",
    "TRITON_CACHE_DIR": f"{_PROJ}/.cache/triton",
    "TORCHINDUCTOR_CACHE_DIR": f"{_PROJ}/.cache/inductor",
    "XDG_CACHE_HOME": f"{_PROJ}/.cache",
}.items():
    os.environ.setdefault(_key, _value)
