#!/usr/bin/env python3
"""Build matrix, derived from the conda-torch channel's own repodata.

cuda-wheels derives its cells from a scraped download.pytorch.org matrix.
Here the source of truth is the channel we build against: if a torch build is
not in conda-torch, an extension for it is unbuildable by construction and
must never be emitted. Two consequences worth stating:

  * a conda-torch republish wave (build numbers _2 -> _3) is visible here
    automatically, and every emitted cell names the EXACT torch build string
    it will compile against;
  * a new torch version appearing in the channel costs zero edits in this
    repo.

Emits matrix.json: a list of job objects (one per shard for sharded
packages), consumed by .github/workflows/build.yml.

Usage:
  generate_matrix.py --package flash-attn [--cuda all] [--pytorch all]
                     [--python all] [--platform all] [-o matrix.json]
"""

import argparse
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import package_loader as pl  # noqa: E402

TORCH_CHANNEL = "https://comfy-forge.github.io/conda-torch"
# pytorch build strings look like cuda128_repack_py312_h<hash>_2 or, for the
# mirrored conda-forge builds, cuda129_mkl_py312_h<hash>_302.
_BUILD_RE = re.compile(r"^cuda(\d+)_[a-z]+_py(\d+)_h[0-9a-f]+_(\d+)$")


def fetch_torch_repodata(subdir: str, cache: Path) -> dict:
    """conda-torch's repodata for a subdir, cached on disk."""
    local = cache / f"conda-torch-{subdir}-repodata.json"
    if not local.is_file():
        cache.mkdir(parents=True, exist_ok=True)
        url = f"{TORCH_CHANNEL}/{subdir}/repodata.json"
        print(f"fetching {url}", file=sys.stderr)
        req = urllib.request.Request(url, headers={"User-Agent": "cuda-foundry"})
        local.write_bytes(urllib.request.urlopen(req, timeout=300).read())
    return json.loads(local.read_text())


def torch_cells(subdir: str, cache: Path) -> dict:
    """{(cuda, torch_version, python): best pytorch build string}.

    "Best" = highest build number, which is what an unpinned solve would take
    and what a fresh build should compile against. Both our repacks and the
    mirrored conda-forge builds are eligible: an extension linking either is
    equally valid, since the ABI is torch's, not the packaging's.
    """
    d = fetch_torch_repodata(subdir, cache)
    out = {}
    for filename, e in d.get("packages.conda", {}).items():
        if e.get("name") != "pytorch":
            continue
        m = _BUILD_RE.match(e.get("build", ""))
        if not m:
            continue
        cuda_digits, py_digits, build_number = m.groups()
        cuda = f"{cuda_digits[:2]}.{cuda_digits[2:]}"
        python = f"{py_digits[0]}.{py_digits[1:]}"
        key = (cuda, e["version"], python)
        prev = out.get(key)
        if prev is None or int(build_number) > prev[1]:
            out[key] = (e["build"], int(build_number))
    return {k: v[0] for k, v in out.items()}


PYTORCH_INDEX = "https://download.pytorch.org/whl"
_PLAT_TAG = {
    "manylinux_2_28_x86_64": "linux-64", "linux_x86_64": "linux-64",
    "manylinux_2_28_aarch64": "linux-aarch64", "linux_aarch64": "linux-aarch64",
    "win_amd64": "win-64",
}


def upstream_cells(pypi_name: str, flavours, cache: Path) -> set:
    """{(version, cuda, python, subdir)} that upstream actually publishes.

    Only for FAMILY packages: torchvision/torchaudio ship official per-flavour
    wheels, and building a combo upstream never shipped would invent a
    pairing nobody has ever tested. Everything else in this repo is built for
    whatever torch exists, because upstream ships no CUDA-flavoured wheel of
    it at all.

    The `+cuNNN` local tag is REQUIRED to count. Every flavour directory also
    lists tag-less wheels (torchvision 0.16.x, torchaudio 0.4.0 ... 2.2.0),
    which are the same default-flavour files repeated in all six directories
    -- counting them would claim, for example, a cu132 torchaudio 2.2.0 that
    does not exist as a cu132 build.
    """
    out = set()
    for fl in flavours:
        cu = "cu" + fl.replace(".", "")
        local = cache / f"upstream-{pypi_name}-{cu}.html"
        if not local.is_file():
            cache.mkdir(parents=True, exist_ok=True)
            url = f"{PYTORCH_INDEX}/{cu}/{pypi_name}/"
            print(f"fetching {url}", file=sys.stderr)
            req = urllib.request.Request(url, headers={"User-Agent": "cuda-foundry"})
            try:
                local.write_bytes(urllib.request.urlopen(req, timeout=300).read())
            except Exception as exc:                      # noqa: BLE001
                print(f"  !! {url}: {exc}", file=sys.stderr)
                continue
        html = local.read_text(errors="replace")
        pat = rf"{re.escape(pypi_name)}-([0-9][0-9.]*)\+{cu}-cp(\d+)-cp\d+(t?)-([a-z0-9_]+)\.whl"
        for ver, py, freethreaded, plat in re.findall(pat, html):
            if freethreaded or plat not in _PLAT_TAG:
                continue      # freethreaded ABI is a separate axis, not built
            out.add((ver, fl, f"{py[0]}.{py[1:]}", _PLAT_TAG[plat]))
    return out


_PRERELEASE = re.compile(r"[abc]|rc|dev", re.I)


def stable_pythons(subdir: str, cache: Path) -> set:
    """Python minors conda-forge ships a STABLE build of, for this subdir.

    conda-torch carries py3.15 artifacts, but conda-forge's python 3.15 is
    alpha-only (3.15.0a2/a3 today), and a solve will not take a prerelease --
    which is why conda-torch's README says its py3.15 records cannot resolve
    yet. Emitting those cells would queue ~27 jobs per package that cannot
    build a host env. Checked against the live channel rather than pinned to
    a hardcoded ceiling, so this self-heals the day 3.15.0 final lands.
    """
    local = cache / f"cf-pythons-{subdir}.json"
    if not local.is_file():
        cache.mkdir(parents=True, exist_ok=True)
        url = "https://api.anaconda.org/package/conda-forge/python"
        req = urllib.request.Request(url, headers={"User-Agent": "cuda-foundry"})
        local.write_bytes(urllib.request.urlopen(req, timeout=300).read())
    files = json.loads(local.read_text()).get("files", [])
    out = set()
    for f in files:
        if (f.get("attrs") or {}).get("subdir") != subdir:
            continue
        v = f.get("version", "")
        if _PRERELEASE.search(v.split(".", 2)[-1] if v.count(".") >= 2 else v):
            continue
        parts = v.split(".")
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            out.add(f"{parts[0]}.{parts[1]}")
    return out


def arch_list_for(cfg: dict, arch_policy: dict, cuda: str, torch_version: str,
                  subdir: str) -> str:
    """Arch list for a cell: per-package override, then exception, then policy."""
    by_cuda = cfg.get("arch_list_by_cuda") or {}
    if cuda in by_cuda:
        return str(by_cuda[cuda])
    if subdir == "linux-aarch64":
        if cfg.get("arch_list_aarch64"):
            return str(cfg["arch_list_aarch64"])
        return str(arch_policy.get("arch_policy_aarch64", {}).get(cuda, "")).replace(";", " ")
    if cfg.get("arch_list"):
        return str(cfg["arch_list"])
    minor = ".".join(torch_version.split(".")[:2])
    exc = arch_policy.get("arch_exceptions", {}).get(f"{cuda}/{minor}")
    if exc:
        return exc.replace(";", " ")
    return str(arch_policy.get("arch_policy", {}).get(cuda, "")).replace(";", " ")


def build_string(cfg: dict, cuda: str, torch_version: str, python: str,
                 build_number: int = 0) -> str:
    """cuda128_torch211_py312_<n>  (the hash is added by the recipe).

    The flavour axis lives here, in the build string, where conda can order
    and match it -- not in a local version tag like the wheel index's
    `+cu128torch2.8`, which conda sorts BELOW the plain version.
    """
    cu = cuda.replace(".", "")
    py = python.replace(".", "")
    if not cfg["links_torch"]:
        return f"cuda{cu}_py{py}_{build_number}"
    tv = ".".join(torch_version.split(".")[:2]).replace(".", "")
    return f"cuda{cu}_torch{tv}_py{py}_{build_number}"


def _has_fragment(name: str, version: str, build_string: str, subdir: str) -> bool:
    """Is this cell already published, i.e. does a repodata fragment exist?

    The fragment filename carries rattler-build's variant hash, which is not
    known until the recipe is rendered, so the cell can only match by glob:
    <name>-<version>-cuda128_torch211_py312_*_<n>.conda.json. That is precise
    enough -- the hash is the only free field, and everything either side of
    it identifies the cell exactly.
    """
    head, _, num = build_string.rpartition("_")
    meta = Path(__file__).resolve().parent.parent / "meta" / subdir
    return any(meta.glob(f"{name}-{version}-{head}_*_{num}.conda.json"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--cuda", default="all")
    ap.add_argument("--pytorch", default="all", help="torch MINOR, e.g. 2.11")
    ap.add_argument("--python", default="all")
    ap.add_argument("--platform", default="all")
    ap.add_argument("--build-number", type=int, default=0)
    ap.add_argument("--cache-dir", type=Path, default=Path("/tmp/conda-cuda-matrix"))
    ap.add_argument("--skip-published", action="store_true",
                    help="drop cells that already have a committed repodata "
                         "fragment. Fragment existence -- not a build log, not "
                         "a run conclusion -- is what 'done' means for this "
                         "channel, so a re-dispatch after a partial wave "
                         "rebuilds only what is actually missing.")
    ap.add_argument("-o", "--output", type=Path)
    args = ap.parse_args()

    policy = pl.load_policy()
    arch_policy = pl.load_arch_policy()
    pkg_dir = pl.PACKAGES_DIR / args.package
    if not pkg_dir.is_dir():
        sys.exit(f"unknown package {args.package!r} (packages/{args.package} not found)")
    cfg = pl.load_package(pkg_dir)

    platforms = policy["platforms"] if args.platform == "all" else [args.platform]
    py_min = tuple(int(x) for x in str(policy["python_min"]).split("."))
    jobs = []

    # ---- family packages: version is a function of the torch built against --
    family = cfg.get("family_versions") or {}
    upstream = None
    holes = {"no_pairing": set(), "not_published": set(), "published": set()}
    if family:
        upstream = upstream_cells(cfg["pypi_name"], policy["supported_cudas"],
                                  args.cache_dir)
        print(f"  upstream publishes {len(upstream)} tagged {cfg['pypi_name']} "
              f"cells across {len(policy['supported_cudas'])} flavours",
              file=sys.stderr)

    for subdir in platforms:
        cells = torch_cells(policy["torch_subdir"][subdir], args.cache_dir)
        buildable_py = stable_pythons(subdir, args.cache_dir)
        skipped_py = set()
        # links_torch: false -> one build per (cuda, python); collapse the
        # torch axis by keeping one representative torch per (cuda, python).
        seen_no_torch = set()
        for (cuda, torch_version, python), torch_build in sorted(cells.items()):
            if cuda not in policy["supported_cudas"]:
                continue
            if args.cuda != "all" and cuda != args.cuda:
                continue
            minor = ".".join(torch_version.split(".")[:2])
            if args.pytorch != "all" and minor != args.pytorch:
                continue
            if args.python != "all" and python != args.python:
                continue
            if tuple(int(x) for x in python.split(".")) < py_min:
                continue
            if python not in buildable_py:
                skipped_py.add(python)   # conda-forge has no stable build yet
                continue
            if cfg.get("min_pytorch") and _ver(minor) < _ver(str(cfg["min_pytorch"])):
                continue
            if not cfg["links_torch"]:
                if (cuda, python) in seen_no_torch:
                    continue
                seen_no_torch.add((cuda, python))

            # A family package's own version and source come from the torch
            # version of the cell. Two distinct holes, both named rather than
            # silently dropped:
            #   no_pairing    -- no release of ours goes with that torch
            #   not_published -- the pair exists but upstream never shipped
            #                    that (version, flavour, python) wheel, so no
            #                    one has ever run this combination
            cell_version, cell_rev = cfg.get("version"), cfg.get("source_rev")
            if family:
                entry = family.get(torch_version)
                if entry is None:
                    holes["no_pairing"].add((torch_version, cuda, python))
                    continue
                cell_version, cell_rev = entry["version"], entry["source_rev"]
                if (cell_version, cuda, python, subdir) not in upstream:
                    holes["not_published"].add((cell_version, cuda, python))
                    continue

            arch = arch_list_for(cfg, arch_policy, cuda, torch_version, subdir)
            if not arch:
                continue  # a CUDA line absent from the arch table is not built
            gcc = (policy.get("host_gcc", {}).get("by_cuda", {}).get(cuda)
                   or policy.get("host_gcc", {}).get("default", "13"))
            shards = int(cfg.get("sharding") or 1)
            bstr = build_string(cfg, cuda, torch_version, python,
                                args.build_number)
            if args.skip_published and _has_fragment(cfg["name"], cell_version,
                                                     bstr, subdir):
                holes["published"].add((cell_version, cuda, python))
                continue
            for shard_index in range(1, shards + 1):
                jobs.append({
                    "package": cfg["name"],
                    "folder": args.package,
                    "version": cell_version,
                    "source_repo": cfg["source_repo"],
                    "source_rev": cell_rev,
                    "cuda": cuda,
                    "cuda_short": cuda.replace(".", ""),
                    "pytorch": minor,
                    "pytorch_full": torch_version,
                    "torch_build": torch_build,
                    "python": python,
                    "platform": subdir,
                    "arch_list": arch,
                    "jobs": cfg["jobs"],
                    "nvcc_threads": cfg["nvcc_threads"],
                    "sharding": shards,
                    "shard_index": shard_index,
                    "shard_count": shards,
                    "patch_script": cfg.get("patch_script", ""),
                    "force_source_build": cfg.get("force_source_build") or {},
                    "links_torch": cfg["links_torch"],
                    "clone_recursive": bool(cfg.get("clone_recursive", False)),
                    "free_disk_space": bool(cfg.get("free_disk_space", True)),
                    "nvcc_flags": cfg.get("nvcc_flags", ""),
                    "build_subdir": cfg.get("build_subdir", ""),
                    "runner": policy["runners"][subdir],
                    # The host compiler nvcc will accept, per CUDA line. Sent
                    # as a variant rather than pinned in variants.yaml
                    # because it is a property of the CELL, not of the repo.
                    "gcc_version": gcc,
                    "build_string": bstr,
                    "family": bool(family),
                    # A family package has one tarball per version, so the
                    # cell names the one it needs; everything else has one.
                    "src_tarball": (f"{cfg['name']}-{cell_version}-source.tar.gz"
                                    if family else f"{cfg['name']}-source.tar.gz"),
                })
        if holes["no_pairing"]:
            pairs = sorted({t for t, _, _ in holes["no_pairing"]})
            print(f"  {subdir}: HOLE, no {cfg['name']} release pairs with torch "
                  f"{pairs} -- upstream never shipped one", file=sys.stderr)
        if holes["published"]:
            done = sorted(holes["published"])
            print(f"  {subdir}: SKIP {len(done)} cell(s) already published "
                  f"(fragment exists); re-dispatch builds only the rest",
                  file=sys.stderr)
        if holes["not_published"]:
            byfl = {}
            for ver, cu, _py in sorted(holes["not_published"]):
                byfl.setdefault(cu, set()).add(ver)
            for cu, vers in sorted(byfl.items()):
                print(f"  {subdir}: HOLE, {cfg['name']} {sorted(vers)} has no "
                      f"upstream {cu} wheel -- combination never published",
                      file=sys.stderr)
        if skipped_py:
            print(f"  {subdir}: skipped python {sorted(skipped_py)} -- conda-torch "
                  f"has builds but conda-forge ships no stable python there yet",
                  file=sys.stderr)

    payload = json.dumps(jobs, indent=1)
    if args.output:
        args.output.write_text(payload + "\n")
    else:
        print(payload)
    cells_n = len({(j["cuda"], j["pytorch_full"], j["python"], j["platform"]) for j in jobs})
    print(f"{cfg['name']}: {cells_n} cells, {len(jobs)} jobs "
          f"(sharding {cfg.get('sharding') or 1})", file=sys.stderr)
    return 0


def _ver(v: str):
    return tuple(int(x) for x in re.findall(r"\d+", v))


if __name__ == "__main__":
    sys.exit(main())
