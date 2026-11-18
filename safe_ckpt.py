"""Restricted loader for torch.save checkpoints whose pickle carries omegaconf config objects
(e.g. Mammo-FM). Only tensors and plain containers are rebuilt; every omegaconf / typing class
is replaced by an inert stub, and any other global raises. No code from the file is run.

Usage: from safe_ckpt import load; ck = load(path)   # dict; ck["model"] -> state dict
"""
import collections
import pickle
import zipfile

import torch

_DTYPES = {"FloatStorage": torch.float32, "LongStorage": torch.int64, "HalfStorage": torch.float16,
           "IntStorage": torch.int32, "BoolStorage": torch.bool, "DoubleStorage": torch.float64,
           "BFloat16Storage": torch.bfloat16}


class _Stub:
    """Inert stand-in for omegaconf / typing objects: stores whatever state it is given."""
    def __init__(self, *a, **k):
        self.args = a

    def __setstate__(self, state):
        self.state = state

    def __setitem__(self, k, v):
        pass


def _rebuild(storage, offset, size, stride, *_):
    t = torch.empty(0, dtype=storage.dtype)
    t.set_(storage, offset, torch.Size(size), tuple(stride))
    return t


class _Unpickler(pickle.Unpickler):
    def __init__(self, f, zf, prefix):
        super().__init__(f)
        self.zf, self.prefix = zf, prefix

    def find_class(self, module, name):
        if (module, name) in (("collections", "OrderedDict"), ("collections", "defaultdict")):
            return getattr(collections, name)
        if module in ("__builtin__", "builtins") and name in ("dict", "list", "long", "int"):
            return {"dict": dict, "list": list, "long": int, "int": int}[name]
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild
        if module == "torch" and name in _DTYPES:
            return name
        if module.startswith("omegaconf") or module == "typing":
            return _Stub
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")

    def persistent_load(self, pid):
        _, kind, key, _loc, numel = pid
        dtype = _DTYPES[kind]
        raw = self.zf.read(f"{self.prefix}/data/{key}")
        st = torch.UntypedStorage.from_buffer(raw, dtype=torch.uint8) if raw else torch.UntypedStorage(0)
        return torch.storage.TypedStorage(wrap_storage=st, dtype=dtype, _internal=True)


def load(path):
    zf = zipfile.ZipFile(path)
    pkl = next(n for n in zf.namelist() if n.endswith("/data.pkl"))
    prefix = pkl[: -len("/data.pkl")]
    import io
    return _Unpickler(io.BytesIO(zf.read(pkl)), zf, prefix).load()
