#!/usr/bin/env python3
"""Clean-environment verification of PUBLISHED artifacts, on real hardware.

This is the developer-box half of the clean-environment test. CI proves that
a package IMPORTS from a fresh solve of only its declared dependencies (the
`rattler-build test` step in build.yml), but a GitHub runner has no GPU, so
the package's own verify op -- the kernel launch that turns a mis-arch'd
build into cudaErrorNoKernelImageForDevice -- can only run here.

Every "works on my machine" defect of the first audit round shipped because
nothing ever installed the package into an environment that held ONLY what
the package declares: cumm needed /usr/local/cuda, sageattention needed
cuda.h, mmcv imported torchvision, torch-sparse imported torch_scatter,
detectron2 died on Pillow 12, flex-gemm imported triton, pytorch3d imported
PIL, pccm imported setuptools. All of those pass on a box with a system CUDA
and a fat site-packages. So this tool builds the environment a USER gets and
nothing more:

  conda half   the published .conda is downloaded from the release and
               handed to `rattler-build test --package-file`, against
               [cuda-foundry, conda-torch, conda-forge] with strict priority:
               a fresh solve of the artifact plus only what it declares, in
               which EVERY entry of the artifact's own info/tests/tests.yaml
               runs -- the import test, the package_contents test and, for
               artifacts built since the verify op moved into the recipe,
               the GPU op itself. CUDA_HOME is unset and every /usr/local/
               cuda* entry is scrubbed from PATH and LD_LIBRARY_PATH so a
               system toolkit cannot stand in for a missing dependency.
               An artifact that predates in-recipe ops carries no script
               test; for those the op is taken from package.yml and run in
               a pixi environment solved the same way, and the table says
               so;
  wheel half   a fresh venv with PyPI torch from download.pytorch.org/whl/
               cu<NNN> (none at all for a links_torch: false package -- there
               is nothing to preload libcudart for it), then the published
               wheel from the live /deps/ index with its sidecar-declared
               dependencies resolved from PyPI, under the same scrubbed
               environment.

In each, `verify.op` from package.yml runs (after `import <verify.import>`).
win-64 artifacts cannot run here, so they are SOLVED only -- `pixi lock` for
the .conda, a uv cross-platform resolve for the wheel -- and reported as such
rather than as passes.

Usage:
  clean_verify.py [--package NAME ...]   (default: every package with a
                                          published fragment under meta/)
                  [--subdir linux-64|win-64|all] [--formats conda,wheel]
                  [--cuda 12.8] [--torch 2.8] [--python 3.12]
                  [--jobs N] [--work DIR] [--timeout SEC] [--keep]

Exit: 0 when every cell passed (a win-64 SOLVED counts as a pass; it is the
strongest statement this box can make), 1 otherwise. A summary table is
printed at the end and written as JSON beside the work tree.

Nothing here compiles anything. Every artifact is downloaded from the
channel or the index exactly as a user would get it.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

CHANNEL = "https://comfy-forge.github.io/cuda-foundry"
TORCH_CHANNEL = "https://comfy-forge.github.io/conda-torch"
DEPS_INDEX = "https://comfy-forge.github.io/pypi-cuda-wheels/deps"
PYPI = "https://pypi.org/simple"
TORCH_INDEX = "https://download.pytorch.org/whl"

BUILD_RE = re.compile(
    r"^cuda(?P<cu>\d+)_(?:torch(?P<torch>\d+)_)?py(?P<py>\d+)_h[0-9a-f]+_(?P<n>\d+)$")

PIXI = os.environ.get("PIXI") or shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
UV = os.environ.get("UV") or shutil.which("uv")


# ---------------------------------------------------------------------------
# the environment the op runs in: no system CUDA anywhere in reach
# ---------------------------------------------------------------------------

_CUDA_VARS = ("CUDA_HOME", "CUDA_PATH", "CUDA_ROOT", "CUDA_TOOLKIT_ROOT_DIR",
              "CUDACXX", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS",
              "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH",
              "PYTHONPATH", "CONDA_PREFIX", "VIRTUAL_ENV", "PIXI_ENVIRONMENT_NAME",
              "PIXI_PROJECT_ROOT", "PIXI_PROJECT_MANIFEST")


def _is_cuda_path(entry: str) -> bool:
    e = entry.replace("\\", "/").lower()
    return "/cuda" in e or "nvidia" in e


def scrubbed_env() -> dict:
    """os.environ minus every way a system CUDA could leak into the op.

    /usr/local/cuda-13.0/bin on PATH is exactly what let cumm pass its op on
    the developer box while the published artifact had no toolkit dependency
    at all: cumm's NVRTC path found `nvcc` on PATH and read the headers beside
    it. The scrub is by substring on purpose -- a toolkit lives under a path
    with "cuda" in it on every box this tool has been run on, and a false
    positive here costs one PATH entry the op should not need anyway.
    """
    env = {k: v for k, v in os.environ.items() if k not in _CUDA_VARS}
    for var in ("PATH", "LD_LIBRARY_PATH"):
        parts = [p for p in env.get(var, "").split(os.pathsep) if p and not _is_cuda_path(p)]
        if parts:
            env[var] = os.pathsep.join(parts)
        else:
            env.pop(var, None)
    # ~/.local/site-packages is a fat site-packages by another name.
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return env


# ---------------------------------------------------------------------------
# published cells, from the committed fragments
# ---------------------------------------------------------------------------

@dataclass
class Cell:
    name: str
    version: str
    build: str
    subdir: str
    cu: str            # "128"
    torch: str | None  # "2.8" or None for a torch-free package
    py: str            # "3.12"
    build_number: int

    @property
    def cuda(self) -> str:
        return f"{self.cu[:2]}.{self.cu[2:]}"

    @property
    def label(self) -> str:
        t = f"torch{self.torch}" if self.torch else "torch-free"
        return f"{self.name} {self.version} cu{self.cu} {t} py{self.py} {self.subdir} #{self.build_number}"

    @property
    def wheel_local(self) -> str:
        # cuda-wheels' convention: +cu128torch2.8. A torch-free package still
        # carries a torch tag in its wheel name (cumm-0.7.11+cu128torch2.8),
        # because the index is one namespace and the tag says which torch
        # LINE the wave was built for even where nothing links it.
        return f"+cu{self.cu}torch{self.torch or _default_torch_tag()}"


def _default_torch_tag() -> str:
    return os.environ.get("CUW_TORCHFREE_WHEEL_TAG", "2.8")


def published_cells(meta_dir: Path, subdirs: list[str]) -> list[Cell]:
    """Newest build number per (name, version, cu, torch, py, subdir)."""
    best: dict[tuple, Cell] = {}
    for sub in subdirs:
        d = meta_dir / sub
        if not d.is_dir():
            continue
        for frag in sorted(d.glob("*.conda.json")):
            e = json.loads(frag.read_text())
            m = BUILD_RE.match(str(e.get("build", "")))
            if not m:
                continue
            t = m.group("torch")
            torch = f"{t[0]}.{t[1:]}" if t else None
            py = f"{m.group('py')[0]}.{m.group('py')[1:]}"
            c = Cell(e["name"], str(e["version"]), e["build"], sub, m.group("cu"),
                     torch, py, int(m.group("n")))
            key = (c.name, c.version, c.cu, c.torch, c.py, c.subdir)
            if key not in best or c.build_number > best[key].build_number:
                best[key] = c
    return sorted(best.values(), key=lambda c: (c.name, c.subdir, c.version))


# ---------------------------------------------------------------------------
# one result per (cell, format)
# ---------------------------------------------------------------------------

@dataclass
class Result:
    cell: Cell
    fmt: str                 # "conda" | "wheel"
    status: str              # PASS | SOLVED | FAIL | SKIP
    step: str = ""           # which step failed / what was done
    detail: str = ""         # first useful line of output
    seconds: float = 0.0
    log: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("PASS", "SOLVED", "SKIP")


def _tail(text: str, n: int = 40) -> str:
    return "\n".join(text.strip().splitlines()[-n:])


def _first_error(text: str) -> str:
    """The line a human would point at, from a failing command's output."""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    # A Python exception line names the cause; rattler-build's trailing
    # "Error: x failed to run test" only says that there was one, so the
    # specific patterns come first.
    for pat in (r"^(ModuleNotFoundError|ImportError|AttributeError|RuntimeError|AssertionError|"
                r"OSError|TypeError|ValueError|KeyError|NameError|SyntaxError)\b",
                r"^\w*Error\b", r"ERROR:", r"No matching", r"cannot", r"Cannot",
                r"Error:", r"error:", r"failed", r"Failed"):
        for ln in reversed(lines):
            if re.search(pat, ln):
                return ln.strip()[:200]
    return (lines[-1] if lines else "")[:200]


def run(cmd: list[str], cwd: Path, env: dict, timeout: int) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, cwd=str(cwd), env=env, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              timeout=timeout)
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") if isinstance(e.stdout, str) else (e.stdout or b"").decode(errors="replace")
        return subprocess.CompletedProcess(cmd, 124, out + f"\n[timeout after {timeout}s]", "")


def op_script(cfg: dict) -> str:
    """`import <verify.import>` plus `verify.op`, as one script.

    The import is stated first and separately so a failure there reads as
    "the package does not import from its declared dependencies" and not as
    a failure inside the op.
    """
    v = cfg.get("verify") or {}
    imp = v.get("import") or cfg["import_name"]
    op = v.get("op") or ""
    return textwrap.dedent(f"""\
        import importlib, os, sys
        print("python", sys.version.split()[0], "prefix", sys.prefix)
        print("CUDA_HOME", os.environ.get("CUDA_HOME"))
        try:
            import torch
            print("torch", torch.__version__, "cuda.is_available", torch.cuda.is_available())
        except ImportError as e:
            print("torch not importable in this env:", e)
        importlib.import_module({imp!r})
        print("import ok:", {imp!r})
        # ---- verify.op ------------------------------------------------------
        """) + op + "\nprint('OP_RESULT=PASS')\n"


# ---------------------------------------------------------------------------
# conda half
# ---------------------------------------------------------------------------

RELEASE = "https://github.com/Comfy-Forge/cuda-foundry/releases/download"
RATTLER_BUILD = (os.environ.get("RATTLER_BUILD") or shutil.which("rattler-build")
                 or os.path.expanduser("~/.pixi/bin/rattler-build"))


def download_conda(cell: Cell, work: Path, env: dict, timeout: int) -> Path:
    """The published bytes, exactly as a user's solver would fetch them."""
    cache = work / "artifacts" / cell.subdir
    cache.mkdir(parents=True, exist_ok=True)
    fn = f"{cell.name}-{cell.version}-{cell.build}.conda"
    dest = cache / fn
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    url = f"{RELEASE}/{cell.subdir}/{fn}"
    p = run(["curl", "-fsSL", "--retry", "3", "-o", str(dest) + ".part", url], cache, env, timeout)
    if p.returncode != 0:
        raise RuntimeError(f"download failed: {url}: {_first_error(p.stdout)}")
    os.replace(str(dest) + ".part", dest)
    return dest


def artifact_tests(conda: Path, work: Path) -> list[dict]:
    """info/tests/tests.yaml from the artifact ([] for an artifact without one)."""
    sys.path.insert(0, str(REPO / "tools"))
    from verify_conda import extract_conda
    root = extract_conda(conda, work / "extract" / conda.stem)
    t = root / "info" / "tests" / "tests.yaml"
    if not t.is_file():
        return []
    try:
        import yaml
        return list(yaml.safe_load(t.read_text()) or [])
    except Exception:
        # A tests.yaml we cannot parse is still a tests.yaml rattler-build
        # will run; report it as "present, unreadable" rather than "absent".
        return [{"unparsed": True}]


def pixi_env(cell: Cell, proj: Path, env: dict, timeout: int, lock_only: bool):
    """A pixi project pinning the published build, strict priority, ours first."""
    shutil.rmtree(proj, ignore_errors=True)
    proj.mkdir(parents=True)
    # Strict priority, our channel first: a name we carry hides conda-forge's
    # builds of it entirely, which is the coverage decision package.yml's
    # `carry` records. The CUDA virtual package is declared for the solve
    # (no GPU is consulted by the solver) and the op then proves the GPU.
    (proj / "pixi.toml").write_text(textwrap.dedent(f"""\
        [workspace]
        name = "clean-verify"
        channels = ["{CHANNEL}", "{TORCH_CHANNEL}", "conda-forge"]
        channel-priority = "strict"
        platforms = ["{cell.subdir}"]

        [system-requirements]
        cuda = "{cell.cuda}"

        [dependencies]
        python = "{cell.py}.*"
        "{cell.name}" = {{ version = "=={cell.version}", build = "{cell.build}" }}
        """))
    p = run([PIXI, "lock" if lock_only else "install"], proj, env, timeout)
    if p.returncode != 0:
        return p, None
    lock = (proj / "pixi.lock").read_text()
    want = f"/{cell.subdir}/{cell.name}-{cell.version}-{cell.build}.conda"
    return p, (want in lock)


def conda_half(cfg: dict, cell: Cell, work: Path, env: dict, timeout: int,
               run_op: bool) -> Result:
    t0 = time.time()
    if cell.subdir.startswith("win"):
        # No Windows here: a solve is the strongest statement this box can
        # make about a win-64 artifact.
        proj = work / "conda" / f"{cell.name}-{cell.build}-{cell.subdir}"
        p, pinned = pixi_env(cell, proj, env, timeout, lock_only=True)
        if p.returncode != 0:
            return Result(cell, "conda", "FAIL", "solve (win-64)", _first_error(p.stdout),
                          time.time() - t0, p.stdout)
        if not pinned:
            return Result(cell, "conda", "FAIL", "solve (win-64)",
                          "lock does not pin the published build", time.time() - t0, p.stdout)
        return Result(cell, "conda", "SOLVED", "pixi lock --platform win-64 (cannot run here)",
                      "", time.time() - t0, p.stdout)

    # ---- the artifact's OWN tests, in rattler-build's fresh solve ----------
    try:
        conda = download_conda(cell, work, env, timeout)
    except RuntimeError as e:
        return Result(cell, "conda", "FAIL", "download", str(e), time.time() - t0)
    tests = artifact_tests(conda, work)
    has_script = any(isinstance(t, dict) and ("script" in t or t.get("unparsed")) for t in tests)
    test_env = dict(env, CONDA_OVERRIDE_CUDA=cell.cuda)
    log, pip_check_failed, pip_note = "", False, ""
    if tests:
        cmd = [RATTLER_BUILD, "test", "--package-file", str(conda),
               "-c", CHANNEL, "-c", TORCH_CHANNEL, "-c", "conda-forge",
               "--channel-priority", "strict", "--log-style", "plain"]
        p = run(cmd, work, test_env, timeout)
        log += p.stdout
        # conda-torch's repacked pytorch keeps upstream's Requires-Dist
        # (nvidia-cublas-cu12 and thirteen more), so the python test's default
        # `pip check` fails in EVERY clean env that holds it -- measured on
        # cc-torch, imports passed and pip check did not. That is a real
        # finding about the artifact's own test entry (it must say
        # pip_check: false), but it must not mask what the op would say, so
        # it is recorded and the run continues.
        pip_check_failed = (p.returncode != 0 and "imports test passed" in p.stdout
                            and "pip check" in p.stdout)
        if p.returncode != 0 and not pip_check_failed:
            step = "rattler-build test" + (" (in-artifact op)" if has_script else " (imports)")
            return Result(cell, "conda", "FAIL", step, _first_error(p.stdout),
                          time.time() - t0, log)
        if pip_check_failed:
            pip_note = "pip check FAILED in the clean env (torch's PyPI Requires-Dist); the artifact's python test needs pip_check: false"
        if has_script or not run_op:
            if pip_check_failed:
                return Result(cell, "conda", "FAIL", "pip check", pip_note, time.time() - t0, log)
            what = "rattler-build test: every in-artifact test entry, GPU op included"
            if not has_script:
                what = "rattler-build test (imports); op not run (--no-op)"
            return Result(cell, "conda", "PASS" if has_script else "SOLVED", what, "",
                          time.time() - t0, log)
    else:
        log += "(artifact carries no info/tests/tests.yaml)\n"

    # ---- an artifact from before the op lived in the recipe ---------------
    # Its tests.yaml has the import test only (or nothing), so the GPU op is
    # taken from package.yml and run in a pixi environment solved the same
    # way rattler-build solved the test env above.
    if not run_op:
        return Result(cell, "conda", "SOLVED", "imports only; op not run (--no-op)", "",
                      time.time() - t0, log)
    proj = work / "conda" / f"{cell.name}-{cell.build}-{cell.subdir}"
    p, pinned = pixi_env(cell, proj, env, timeout, lock_only=False)
    log += p.stdout
    if p.returncode != 0:
        return Result(cell, "conda", "FAIL", "solve/install (pixi)", _first_error(p.stdout),
                      time.time() - t0, log)
    if not pinned:
        return Result(cell, "conda", "FAIL", "solve (pixi)",
                      "lock does not pin the published build", time.time() - t0, log)
    (proj / "op.py").write_text(op_script(cfg))
    # `pixi run` activates the environment exactly as a user's would be --
    # including any activation script a dependency ships (cuda-nvcc's sets
    # CUDA_HOME to the prefix, which is legitimate: it came from a declared
    # dependency, not from the box).
    p = run([PIXI, "run", "--manifest-path", str(proj / "pixi.toml"),
             "python", "op.py"], proj, env, timeout)
    log += p.stdout
    if p.returncode != 0 or "OP_RESULT=PASS" not in p.stdout:
        step = "import" if "import ok:" not in p.stdout else "op"
        return Result(cell, "conda", "FAIL", step + " (package.yml op, pixi env)",
                      _first_error(p.stdout), time.time() - t0, log)
    if tests and pip_check_failed:
        return Result(cell, "conda", "FAIL", "pip check (imports + op passed)", pip_note,
                      time.time() - t0, log)
    return Result(cell, "conda", "PASS",
                  "rattler-build test (imports) + package.yml op in a pixi env "
                  "(artifact predates in-recipe ops)", "", time.time() - t0, log)


# ---------------------------------------------------------------------------
# wheel half
# ---------------------------------------------------------------------------



def wheel_half(cfg: dict, cell: Cell, work: Path, env: dict, timeout: int,
               run_op: bool) -> Result:
    t0 = time.time()
    if not UV:
        return Result(cell, "wheel", "FAIL", "setup",
                      "uv not found -- needed to create a venv for the cell's python "
                      "and to cross-resolve win-64 wheels", 0)
    pypi_name = cfg.get("pypi_name") or cell.name
    spec = f"{pypi_name}=={cell.version}{cell.wheel_local}"
    links_torch = bool(cfg.get("links_torch", True))
    torch_spec = f"torch=={cell.torch}.*" if cell.torch else None

    if cell.subdir.startswith("win"):
        # No Windows here: resolve for it. uv can solve for a foreign
        # platform without installing anything, which is exactly the
        # "does the sidecar's dependency list resolve on win-64 against
        # PyPI" question. torch comes from its own index so the resolve
        # sees the cu-flavoured wheel a user would have installed first.
        d = work / "wheel" / f"{cell.name}-{cell.build}-{cell.subdir}"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        reqs = [spec] + ([torch_spec] if links_torch and torch_spec else [])
        (d / "requirements.in").write_text("\n".join(reqs) + "\n")
        cmd = [UV, "pip", "compile", "requirements.in",
               "--python-platform", "windows", "--python-version", cell.py,
               "--index-url", DEPS_INDEX, "--extra-index-url", PYPI,
               "--extra-index-url", f"{TORCH_INDEX}/cu{cell.cu}",
               "--index-strategy", "unsafe-best-match", "--no-header", "--quiet",
               "-o", "resolved.txt"]
        p = run(cmd, d, env, timeout)
        if p.returncode != 0:
            return Result(cell, "wheel", "FAIL", "resolve (win-64)", _first_error(p.stdout),
                          time.time() - t0, p.stdout)
        resolved = (d / "resolved.txt").read_text()
        if f"{cell.version}{cell.wheel_local}" not in resolved:
            return Result(cell, "wheel", "FAIL", "resolve (win-64)",
                          f"resolution does not pin {spec}", time.time() - t0, resolved)
        return Result(cell, "wheel", "SOLVED", "uv cross-resolve for win-64 (cannot run here)",
                      "", time.time() - t0, resolved)

    # The base environment comes from conda-torch, not from PyTorch's index.
    # conda-torch's `pytorch cuda<NNN>_repack_*` IS the PyPI wheel repackaged
    # (byte-identical libtorch, docs/ARCHITECTURE.md), so downloading the
    # same 889 MB torch from download.pytorch.org -- plus ~3 GB of nvidia-*
    # libraries, unpacked per venv -- tested nothing the channel does not
    # already hold and was the slow, contended part of every sweep. pixi
    # hardlinks the env from its cache in seconds. What this gives up is the
    # pip-only environment where CUDA libraries arrive as nvidia-*-cu12
    # wheels; our wheels are auditwheel-repaired and load CUDA through
    # torch's own lib dir either way, so the bytes under test are the same.
    venv = work / "wheel" / f"{cell.name}-{cell.build}-{cell.subdir}"
    shutil.rmtree(venv, ignore_errors=True)
    venv.mkdir(parents=True)
    torch_line = ""
    if links_torch and torch_spec:
        torch_line = (f'pytorch = {{ version = "{cell.torch}.*", build = "cuda{cell.cu}_repack_*" }}\n'
                      f'        libtorch = {{ version = "{cell.torch}.*", build = "cuda{cell.cu}_repack_*" }}\n')
    (venv / "pixi.toml").write_text(textwrap.dedent(f"""\
        [workspace]
        name = "clean-verify-wheel"
        channels = ["{TORCH_CHANNEL}", "conda-forge"]
        channel-priority = "strict"
        platforms = ["{cell.subdir}"]

        [system-requirements]
        cuda = "{cell.cuda}"

        [dependencies]
        python = "{cell.py}.*"
        pip = "*"
        {torch_line}"""))
    p = run([PIXI, "install"], venv, env, timeout)
    log = p.stdout
    if p.returncode != 0:
        return Result(cell, "wheel", "FAIL", "conda-torch base env (pixi install)", _first_error(p.stdout),
                      time.time() - t0, log)
    py = venv / ".pixi" / "envs" / "default" / "bin" / "python"
    pip = [str(py), "-m", "pip", "install", "--quiet", "--no-input"]
    constraints = venv / "constraints.txt"
    if links_torch and torch_spec:
        got = subprocess.run([str(py), "-c", "import torch;print(torch.__version__)"],
                             capture_output=True, text=True, env=env).stdout.strip()
        # Pinned for the sidecar install: a dependency on PyPI that asks for
        # a newer torch must be refused rather than pip swapping the torch.
        constraints.write_text(f"torch=={got}\n")
        # torchvision / torchaudio the package declares: the sidecar omits
        # the whole torch family on purpose (make_wheel TORCH_NAMES), so a
        # consumer installs them beside torch. They are small (3-9 MB) and
        # come from PyTorch's index under the torch pin; pip sees the
        # repack's torch dist-info and resolves against it without touching
        # torch.
        family = sorted({str(d).split()[0] for d in (cfg.get("run_deps") or []) if isinstance(d, str)}
                        & {"torchvision", "torchaudio"})
        if family:
            p = run(pip + family + ["--index-url", f"{TORCH_INDEX}/cu{cell.cu}", "-c", str(constraints)],
                    venv, env, timeout)
            log += p.stdout
            if p.returncode != 0:
                return Result(cell, "wheel", "FAIL", f"install {' '.join(family)}", _first_error(p.stdout),
                              time.time() - t0, log)
    else:
        constraints.write_text("")

    # The exact published pair for THIS build number, by release URL: the
    # deps twin from <subdir>-deps and its sidecar beside it. Going through
    # the /deps/ index instead tests the index's freshness, not the wheel --
    # with the index six-hourly, a version pin matched the previous build's
    # twin and its older sidecar, and the cell labelled #2 installed #1
    # (cumesh, 2026-09-12). The index is checked separately by verify_wheel
    # and generate_index's own tests.
    index_note = ""
    tag = f"{cell.subdir}-deps"
    lst = subprocess.run(["gh", "release", "view", tag, "--repo", "Comfy-Forge/cuda-foundry",
                          "--json", "assets", "-q", ".assets[].name"],
                         capture_output=True, text=True)
    want_prefix = f"{pypi_name.replace('-', '_')}-"
    want_mid = f"-{cell.build_number}-cp{cell.py.replace('.', '')}-"
    plat_tag = "win_amd64" if cell.subdir == "win-64" else "manylinux"
    names = [a for a in lst.stdout.split()
             if a.startswith(want_prefix) and want_mid in a and a.endswith(".whl") and plat_tag in a]
    if len(names) != 1:
        return Result(cell, "wheel", "FAIL", "locate deps twin",
                      f"expected exactly one {want_prefix}*{want_mid}*{plat_tag}*.whl on release {tag}, "
                      f"found {names}", time.time() - t0, lst.stdout + lst.stderr)
    whl_url = f"{RELEASE}/{tag}/{names[0]}"
    side = run(["curl", "-fsSL", whl_url + ".metadata"], venv, env, timeout)
    log += side.stdout
    if side.returncode != 0:
        return Result(cell, "wheel", "FAIL", "fetch sidecar", _first_error(side.stdout),
                      time.time() - t0, log)
    reqs = [ln.split(":", 1)[1].strip() for ln in side.stdout.splitlines()
            if ln.startswith("Requires-Dist:")]
    p = run(pip + [whl_url, "--no-deps", "-c", str(constraints)], venv, env, timeout)
    log += p.stdout
    if p.returncode == 0 and reqs:
        # The sidecar's own list, resolved from PyPI under the torch pin --
        # what a consumer of /deps/ gets once the index lists this build.
        # Our index first, PyPI as fallback -- the consumer's configuration.
        # A sidecar may name a sibling from this channel that is not on
        # PyPI at all (nvdiffrec-render -> nvdiffrast, 2026-09-12).
        p = run(pip + reqs + ["-c", str(constraints), "--index-url", DEPS_INDEX, "--extra-index-url", PYPI],
                venv, env, timeout)
        log += p.stdout
    if p.returncode != 0:
        return Result(cell, "wheel", "FAIL", "install wheel (+deps from PyPI)",
                      _first_error(p.stdout), time.time() - t0, log)
    if not run_op:
        return Result(cell, "wheel", "FAIL" if index_note else "SOLVED",
                      "installed, op not run (--no-op)" + (" -- " + index_note if index_note else ""),
                      "", time.time() - t0, log)
    (venv / "op.py").write_text(op_script(cfg))
    p = run([str(py), "op.py"], venv, env, timeout)
    log += p.stdout
    if p.returncode != 0 or "OP_RESULT=PASS" not in p.stdout:
        step = "import" if "import ok:" not in p.stdout else "op"
        return Result(cell, "wheel", "FAIL", step, _first_error(p.stdout),
                      time.time() - t0, log)
    if index_note:
        return Result(cell, "wheel", "FAIL", "/deps/ index (import + op passed)", index_note,
                      time.time() - t0, log)
    return Result(cell, "wheel", "PASS", "import + op on GPU", "", time.time() - t0, log)


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", action="append", default=[],
                    help="package folder or conda name; repeatable; default all published")
    ap.add_argument("--subdir", default="linux-64", help="linux-64, win-64 or all")
    ap.add_argument("--formats", default="conda,wheel")
    ap.add_argument("--cuda", default="", help="only cells of this CUDA line, e.g. 12.8")
    ap.add_argument("--torch", default="", help="only cells of this torch minor, e.g. 2.8")
    ap.add_argument("--python", default="", help="only cells of this python, e.g. 3.12")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=1800, help="per step, seconds")
    ap.add_argument("--work", type=Path, default=Path(os.environ.get("CUW_CLEAN_VERIFY_WORK", "/tmp/cuw-clean-verify")))
    ap.add_argument("--meta-dir", type=Path, default=REPO / "meta")
    ap.add_argument("--keep", action="store_true", help="keep the environments afterwards")
    ap.add_argument("--no-op", action="store_true", help="solve and install only; skip the GPU op")
    ap.add_argument("--json", type=Path, default=None, help="where to write results (default <work>/results.json)")
    args = ap.parse_args()

    import package_loader as pl
    configs: dict[str, dict] = {}
    for folder, cfg in pl.iter_packages():
        configs[cfg["name"]] = cfg
        configs.setdefault(folder, cfg)

    subdirs = ["linux-64", "win-64"] if args.subdir == "all" else [args.subdir]
    cells = published_cells(args.meta_dir, subdirs)
    if args.package:
        wanted = set()
        for p in args.package:
            if p not in configs:
                sys.exit(f"clean_verify: no packages/*/package.yml for {p!r}")
            wanted.add(configs[p]["name"])
        cells = [c for c in cells if c.name in wanted]
    if args.cuda:
        cells = [c for c in cells if c.cuda == args.cuda]
    if args.torch:
        cells = [c for c in cells if (c.torch or "") == args.torch]
    if args.python:
        cells = [c for c in cells if c.py == args.python]
    cells = [c for c in cells if c.name in configs]
    if not cells:
        print("clean_verify: no published cells match")
        return 1
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]

    env = scrubbed_env()
    args.work.mkdir(parents=True, exist_ok=True)
    print(f"clean_verify: {len(cells)} cell(s) x {formats}, {args.jobs} at a time, work={args.work}")
    print(f"  PATH after scrub: {env.get('PATH', '')}")
    print(f"  CUDA_HOME: {env.get('CUDA_HOME', '<unset>')}")

    tasks = []
    for c in cells:
        cfg = configs[c.name]
        if "conda" in formats:
            tasks.append((c, "conda", cfg))
        if "wheel" in formats:
            tasks.append((c, "wheel", cfg))

    def one(task):
        c, fmt, cfg = task
        try:
            if fmt == "conda":
                r = conda_half(cfg, c, args.work, env, args.timeout, not args.no_op)
            else:
                r = wheel_half(cfg, c, args.work, env, args.timeout, not args.no_op)
        except Exception as e:  # a tool crash is a FAIL with a reason, never a hang
            r = Result(c, fmt, "FAIL", "tool", f"{type(e).__name__}: {e}", 0)
        finally:
            # Each cell's environment goes the moment the cell is done, not at
            # the end of the run: with torch in every one they are 5-8 GB
            # each, and letting ~180 of them accumulate filled a 916 GB disk
            # mid-sweep (2026-09-12, ENOSPC on simple-knn). --keep keeps them.
            if not args.keep:
                shutil.rmtree(args.work / fmt / f"{c.name}-{c.build}-{c.subdir}", ignore_errors=True)
        mark = "ok  " if r.ok else "FAIL"
        print(f"{mark} {r.status:6s} {fmt:5s} {c.label:60s} {r.step}"
              + (f" -- {r.detail}" if r.detail else "") + f"  ({r.seconds:.0f}s)", flush=True)
        (args.work / "logs").mkdir(exist_ok=True)
        (args.work / "logs" / f"{c.name}-{c.build}-{c.subdir}.{fmt}.log").write_text(r.log or "")
        return r

    results: list[Result] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        for r in ex.map(one, tasks):
            results.append(r)

    # ---- the table -----------------------------------------------------
    by_cell: dict[str, dict[str, Result]] = {}
    for r in results:
        by_cell.setdefault(r.cell.label, {})[r.fmt] = r
    w = max(len(k) for k in by_cell) + 2
    print("\n" + "=" * (w + 60))
    print(f"{'cell':{w}s} {'conda':22s} {'wheel':22s} detail")
    print("-" * (w + 60))
    for label, d in sorted(by_cell.items()):
        cols, details = [], []
        for fmt in ("conda", "wheel"):
            r = d.get(fmt)
            if r is None:
                cols.append(f"{'-':22s}")
                continue
            cols.append(f"{r.status + ('' if r.ok else ' @' + r.step):22s}")
            if not r.ok and r.detail:
                details.append(f"{fmt}: {r.detail}")
        print(f"{label:{w}s} {cols[0]} {cols[1]} {' | '.join(details)}")
    print("=" * (w + 60))
    n_fail = sum(1 for r in results if not r.ok)
    print(f"{len(results) - n_fail}/{len(results)} ok; logs under {args.work}/logs")

    out = args.json or (args.work / "results.json")
    out.write_text(json.dumps([{
        "package": r.cell.name, "cell": r.cell.label, "subdir": r.cell.subdir,
        "format": r.fmt, "status": r.status, "step": r.step, "detail": r.detail,
        "seconds": round(r.seconds, 1)} for r in results], indent=1))
    if not args.keep:
        for sub in ("conda", "wheel"):
            shutil.rmtree(args.work / sub, ignore_errors=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
