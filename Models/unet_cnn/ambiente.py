"""Ambiente comum: caches no D: e GPU obrigatória.

O disco do WSL (ext4.vhdx) mora no C:, que está cheio — qualquer coisa gravada
em ~/.cache ou /tmp do WSL ocupa o C:. Por isso todo cache/temporário vai para
Models/.cache (no D:). Importar ESTE módulo antes de torch/xgboost/matplotlib.
"""

import os
import sys

MODELS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(MODELS, ".cache")

_DESTINOS = {
    "XDG_CACHE_HOME": "",
    "TMPDIR": "tmp",
    "CUDA_CACHE_PATH": "nv",
    "MPLCONFIGDIR": "matplotlib",
    "TORCH_HOME": "torch",
    "HF_HOME": "huggingface",
    "TRITON_CACHE_DIR": "triton",
    "TORCHINDUCTOR_CACHE_DIR": "inductor",
    "PIP_CACHE_DIR": "pip",
}
for _var, _sub in _DESTINOS.items():
    _p = os.path.join(CACHE, _sub)
    os.makedirs(_p, exist_ok=True)
    os.environ[_var] = _p


def exigir_gpu():
    """Aborta se não houver CUDA (treino e features sempre na GPU)."""
    import torch
    if not torch.cuda.is_available():
        print("CUDA indisponivel: este projeto roda so na GPU. Use o venv com "
              "torch CUDA (Data/.venv) dentro do WSL.", file=sys.stderr)
        sys.exit(1)
    return torch.device("cuda")