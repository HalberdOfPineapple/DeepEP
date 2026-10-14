# Installing DeepEP in the `deepep` conda env

Installed version: `deep-ep-2.5.0+93eb6eb`, built with torch `2.14.1+cu132`, nvcc 13.0, Python 3.12, on 2026-10-08.

## Steps

```bash
conda activate deepep

# 1. NCCL >= 2.32.3 (re-run this after any torch install or upgrade, see below)
python -m pip install "nvidia-nccl-cu13>=2.32.3" --no-deps

# 2. elfutils headers, installed into the env instead of system-wide
conda install -y -c conda-forge elfutils

# 3. Build and install, with the env's headers on the include path
export CPATH=$CONDA_PREFIX/include${CPATH:+:$CPATH}
bash install.sh

# 4. Check the install (run from outside the repo so the installed package is imported)
cd /tmp && python -c "import deep_ep; print(deep_ep.__file__)"
```

## Problems and fixes

### `fatal error: elfutils/libdwfl.h: No such file or directory`

`third-party/deep_jit/include/deep_jit/utils/exception.hpp` includes `<elfutils/libdwfl.h>`. The system has the runtime library (`libdw1t64`) but not the dev headers (`libdw-dev`). The headers are only needed at compile time: at runtime the code `dlopen`s `libdw.so.1`, which the system already provides. Because the machine is shared, the headers were installed into the conda env (`conda install -c conda-forge elfutils`) instead of with apt. `setup.py` doesn't add the env's include directory, so `CPATH=$CONDA_PREFIX/include` must be set when building.

### `'struct ncclCommProperties' has no member named 'ginMinStride'` (and similar GIN errors)

The env had `nvidia-nccl-cu13 2.30.7` even though `>=2.32.3` had been installed earlier. Installing torch afterwards replaced it, because torch 2.14.1 pins `nvidia-nccl-cu13==2.30.7`. NCCL 2.30.7 lacks the APIs DeepEP uses (`ginMinStride`, `ginSupport`, `ginCustomStride`, `NCCL_GIN_CONNECTION_CUSTOM_STRIDE`, `NCCL_WIN_GIN_ONLY`). Fix: reinstall with `python -m pip install "nvidia-nccl-cu13>=2.32.3" --no-deps`.

## Caveats

- `pip check` will report `torch 2.14.1+cu132 has requirement nvidia-nccl-cu13==2.30.7, but you have nvidia-nccl-cu13 2.32.3`. This is expected with the `--no-deps` step.
- Reinstalling or upgrading torch will put NCCL back to 2.30.7. Re-run step 1 afterwards, then rebuild if needed.
- `install.sh` doesn't stop on a failed build. A failure shows up at the end as `ERROR: Invalid wheel filename (wrong number of parts): '*'`, and the real compile error is further up in the log.
