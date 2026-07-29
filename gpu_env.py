"""Make TensorFlow find the pip CUDA-12 wheels so it runs on the GPU.

In this env TF (built for CUDA 12) can't auto-discover the nvidia-*-cu12 libs, so
it silently falls back to CPU. We put those wheels' lib dirs on LD_LIBRARY_PATH
and re-exec once -- the dynamic loader only reads LD_LIBRARY_PATH at startup, so a
plain os.environ tweak in-process is too late; re-exec is the reliable fix.

PyTorch uses its own CUDA-13 libs and must NOT call this (it would shove cu12 libs
ahead of cu13 and mismatch). Only TF scripts import it. Call enable_tf_gpu() at
the very top of a TF entrypoint, before `import tensorflow`.
"""

import glob
import os
import site
import sys

_FLAG = "_TF_CUDA_READY"


def _nvidia_lib_dirs():
    roots = []
    try:
        roots += site.getsitepackages()
    except Exception:  # noqa: BLE001  (some envs lack getsitepackages)
        pass
    roots.append(site.getusersitepackages())
    dirs = []
    for r in roots:
        dirs += glob.glob(os.path.join(r, "nvidia", "*", "lib"))
    return sorted(set(dirs))


def enable_tf_gpu():
    """Ensure the cu12 libs are on LD_LIBRARY_PATH, re-exec once so it takes effect."""
    if os.environ.get(_FLAG):
        return                                   # already re-exec'd -> loader is set
    dirs = _nvidia_lib_dirs()
    if dirs:
        prev = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(dirs + ([prev] if prev else []))
    os.environ[_FLAG] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)  # fds (stdin pipe) survive execv


if __name__ == "__main__":                       # quick check: does TF see the GPU now?
    enable_tf_gpu()
    import tensorflow as tf
    print("GPUs seen by TF:", tf.config.list_physical_devices("GPU"))
