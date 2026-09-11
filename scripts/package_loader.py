"""Single loader for this repo's package layout.

Ported from cuda-wheels' package_loader.py. Its hard errors are a written
record of real failures and every one still applies here, so they are kept
verbatim in spirit: parallelism must be declared, source refs must not float,
`links_torch` must be explicit, and any override needs a README saying why.

What changed for conda:

  * `requires_dist` is no longer a hard error but its INVERSE is. The wheel
    farm stripped all dependency metadata and made declaring it an error;
    a conda package that declares nothing is a lie the solver believes, so
    here `run_deps` is required (write `run_deps: []` to state "none", which
    is a claim someone reviewed, not an omission).
  * `force_source_build` guards the no-prebuilt-wheel rule: an upstream whose
    build can fetch a binary must say how it is turned off.
  * `host_deps`, `pypi_name`, `import_name`, `carry` are new and owned here.

Layout:
    defaults/policy.yml              owned axes (cudas, python floor, platforms)
    defaults/arch_policy.yml         owned arch policy
    packages/<name>/package.yml      source, build knobs, dependencies
    packages/<name>/arch_override.yml   optional, needs README
    packages/<name>/patches/*.py     optional source patches
"""
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
POLICY_FILE = ROOT / "defaults" / "policy.yml"
ARCH_POLICY_FILE = ROOT / "defaults" / "arch_policy.yml"
PACKAGES_DIR = ROOT / "packages"

# Upstreams whose build downloads a prebuilt binary unless explicitly told
# not to. Keyed by source_repo. The value is the env var that forces a source
# build and the ONLY value upstream accepts — flash-attention compares
# `os.getenv("FLASH_ATTENTION_FORCE_BUILD", "FALSE") == "TRUE"`, so "1" is
# silently permissive, which is exactly how the earlier experiment shipped
# Dao-AILab's binary while believing it had compiled it.
PREBUILT_FETCHING_UPSTREAMS = {
    "Dao-AILab/flash-attention": {"FLASH_ATTENTION_FORCE_BUILD": "TRUE"},
}

CARRY_VALUES = {"complete", "distinct-name"}

# Platform selectors a conditional run dep may use. rattler-build's own
# vocabulary, so the template renders them as `- if: <sel>` unchanged.
PLATFORM_SELECTORS = {"linux", "win", "unix", "osx"}


def resolve_run_deps(run_deps, platform: str) -> list:
    """Flatten conditional run deps for one target platform (conda subdir).

    `{if: linux, then: triton}` contributes "triton" on linux-64/aarch64 and
    nothing on win-64; plain strings pass through. The wheel side needs this
    because its sidecar is written from package.yml, not from the rendered
    recipe, and must say the same thing the .conda says for that platform.
    """
    fam = "win" if platform.startswith("win") else (
        "osx" if platform.startswith("osx") else "linux")
    out = []
    for d in run_deps or []:
        if isinstance(d, dict):
            sel = d["if"]
            if sel == fam or (sel == "unix" and fam != "win"):
                out.append(d["then"])
        else:
            out.append(d)
    return out


def load_policy() -> dict:
    """Owned axes: supported_cudas, python_min, platforms, runners, defaults."""
    return yaml.safe_load(POLICY_FILE.read_text())


def load_arch_policy() -> dict:
    """arch_policy[_aarch64] and arch_exceptions."""
    return yaml.safe_load(ARCH_POLICY_FILE.read_text())


_CREDENTIAL_WORDS = ("Authorization", "Bearer", "GH_TOKEN", "GITHUB_TOKEN",
                     "authorization", "api_key", "apikey", "password")


def _check_pre_build_not_redactable(cfg: dict, pkg_dir: Path) -> None:
    """GitHub redacts a job output that looks credential-bearing.

    generate_matrix embeds pre_build_script verbatim into the matrix output,
    so a package whose inline pre-build mentions a token vanishes from its own
    build: every job sees a null matrix, skips, and the run reports success
    having produced nothing (spconv, cuda-wheels, 2026-08-24).
    """
    script = cfg.get("pre_build_script") or ""
    if "\n" not in script.strip():
        return
    hits = sorted({w for w in _CREDENTIAL_WORDS if w in script})
    if hits:
        raise SystemExit(
            f"ERROR: {pkg_dir.name}/package.yml: inline pre_build_script "
            f"mentions {hits} -- GitHub will redact the matrix output and the "
            f"package will silently build NOTHING. Move it to "
            f"packages/{pkg_dir.name}/pre_build.sh and reference that file.")


def _check_parallelism_declared(cfg: dict, pkg_dir: Path) -> None:
    """`jobs` and `nvcc_threads` are mandatory. No defaults, deliberately.

    In cuda-wheels these fell back to a shared default, and the result was
    that 30 of 42 packages never stated a job count and NOT ONE stated a
    thread count -- while those two numbers multiply into peak compile memory
    (jobs x nvcc_threads x one cicc), which on a 16GB runner with CUTLASS
    sources decides whether the compile swaps or dies. "Unset" is also not
    neutral: several upstreams pick their own MAX_JOBS when the env is empty,
    and some pick 10.
    """
    missing = [k for k in ("jobs", "nvcc_threads") if cfg.get(k) is None]
    if missing:
        raise SystemExit(
            f"ERROR: {pkg_dir.name}/package.yml does not declare "
            f"{' and '.join(missing)}. Both are required in every package, "
            f"with no default -- they decide peak compile memory together. "
            f"Start from jobs: 3, nvcc_threads: 1 and lower `jobs` if the "
            f"build swaps.")
    for k in ("jobs", "nvcc_threads"):
        v = cfg[k]
        if not isinstance(v, int) or v < 1:
            raise SystemExit(
                f"ERROR: {pkg_dir.name}/package.yml has {k}: {v!r} -- must be "
                f"an integer >= 1; the value is used verbatim.")


def _check_shard_sources(cfg: dict, pkg_dir: Path) -> None:
    """A package that shards must say WHICH files may be divided, and only then.

    Linux needs no such declaration: its nvcc wrapper sits in the compiler seat
    and sees each translation unit as it is invoked. Windows has no seat --
    ninja hands its command lines to CreateProcess, which appends only `.exe`,
    so PATHEXT never applies and nothing but a real executable can occupy it --
    so the win-64 partition works on the SOURCE files and has to be told which
    ones are translation units.

    Required rather than defaulted, for the reason the parallelism check gives:
    a wrong-but-plausible default here is a shard that compiles a subset of its
    own slice and still exits 0. build_win.py checks the declaration against
    .ninja_log after the build, so a wrong list fails loudly -- but only if
    there IS one.
    """
    shards = int(cfg.get("sharding") or 1)
    declared = cfg.get("shard_sources") or []
    partition = cfg.get("shard_partition")
    # `shard_partition: source` is the other way a sharded package can be
    # partitioned: its OWN build reads CUW_SHARD_INDEX / CUW_SHARD_COUNT and
    # compiles only its slice, on both platforms. It exists for the package
    # whose translation units do not exist until the build generates them --
    # natten stamps out ~150 CUTLASS kernels from templates inside setup.py
    # and a source-file glob evaluated before `pip wheel` matches nothing. The
    # Linux nvcc-seat partition is then switched OFF (running both would
    # compile the intersection, ~1/N^2 per shard), and win-64 skips the file
    # stubbing and instead asserts that every nvcc TU ninja built was stored
    # in the cache. It is a declaration the build honours, not a default.
    if partition not in (None, "source"):
        raise SystemExit(
            f"ERROR: {pkg_dir.name}/package.yml: shard_partition: {partition!r} "
            f"is not understood; the only value is 'source' (the package's own "
            f"build partitions its translation units), or omit it.")
    if partition == "source" and declared:
        raise SystemExit(
            f"ERROR: {pkg_dir.name}/package.yml declares both shard_sources and "
            f"shard_partition: source -- one partition mechanism, not two.")
    if partition == "source" and shards <= 1:
        raise SystemExit(
            f"ERROR: {pkg_dir.name}/package.yml declares shard_partition: source "
            f"but sharding is {shards} -- nothing reads it.")
    if shards > 1 and not declared and partition != "source":
        raise SystemExit(
            f"ERROR: {pkg_dir.name}/package.yml sets sharding: {shards} but "
            f"declares no shard_sources. win-64 partitions on source files and "
            f"cannot infer the translation unit list; list the globs (relative "
            f"to the source root, one entry per pattern, EVERY translation "
            f"unit including the C++ ones), or declare `shard_partition: "
            f"source` if the package's own build partitions itself.")
    if declared and shards <= 1:
        raise SystemExit(
            f"ERROR: {pkg_dir.name}/package.yml declares shard_sources but "
            f"sharding is {shards} -- nothing reads it. Either shard, or drop "
            f"the list rather than leaving a declaration that does nothing.")
    if declared and not isinstance(declared, list):
        raise SystemExit(
            f"ERROR: {pkg_dir.name}/package.yml shard_sources must be a list "
            f"of glob patterns, got {type(declared).__name__}.")


def _check_dependencies_declared(cfg: dict, pkg_dir: Path) -> None:
    """`run_deps` is mandatory -- the inverse of the wheel farm's rule.

    cuda-wheels strips every Requires-Dist from every wheel and installs with
    --no-deps, so declaring dependencies there was a hard error. A conda
    channel is the opposite: an artifact with an empty `run:` is a lie the
    solver believes. So the list is required, and `run_deps: []` is a
    reviewed claim that this package genuinely needs nothing at runtime
    beyond python and its torch -- not an omission.

    Build-only tools do NOT belong here. The wheel farm shipped `ninja` as a
    runtime dep of gsplat, a setup_requires; put those in `build_deps`. Read
    the imports before deciding, though: mmcv 1.7.2's `yapf` looks like the
    same mistake and is not -- mmcv/utils/config.py imports it at module
    scope and `import mmcv` fails without it.
    """
    if "run_deps" not in cfg:
        raise SystemExit(
            f"ERROR: {pkg_dir.name}/package.yml does not declare run_deps. "
            f"Every package must state its runtime dependencies as conda "
            f"names; write `run_deps: []` if it genuinely has none. Do not "
            f"list build tools (ninja, packaging, psutil, setuptools) -- "
            f"those go in build_deps.")
    # host_deps_linux / host_deps_win exist for the one kind of dependency
    # that genuinely differs by platform rather than by package: where a
    # toolkit component lives. The libcuda link stub is `cuda-driver-dev` on
    # linux-64 (lib/stubs/libcuda.so) and does not exist as a package on
    # win-64 at all -- cuda.lib ships inside cuda-cudart-dev_win-64 -- so a
    # package linking -lcuda (sageattention's Hopper TMA path) cannot state
    # that dependency in one unconditional list without making one platform
    # UNSAT. They are additive to host_deps, never a replacement for it.
    for key in ("run_deps", "host_deps", "build_deps",
                "host_deps_linux", "host_deps_win"):
        v = cfg.get(key)
        if v is not None and not isinstance(v, list):
            raise SystemExit(
                f"ERROR: {pkg_dir.name}/package.yml: {key} must be a list, "
                f"got {type(v).__name__}.")
    # A run dep may be platform-conditional: `{if: linux, then: triton}`.
    # Needed because a dependency can exist on one subdir and nowhere on
    # another -- `triton` has no win-64 build on conda-forge or conda-torch
    # (measured), so an unconditional `triton` makes every win-64 cell of a
    # triton-using package UNSAT, while omitting it everywhere makes the
    # linux-64 metadata a lie. The selector vocabulary is rattler-build's own
    # (`if: linux` / `if: win` / `if: unix`), rendered verbatim into `run:`,
    # and tools/make_wheel.py resolves the same entries for the sidecar.
    for d in cfg.get("run_deps") or []:
        if isinstance(d, str):
            continue
        if (not isinstance(d, dict) or set(d) != {"if", "then"}
                or d["if"] not in PLATFORM_SELECTORS
                or not isinstance(d["then"], str) or not d["then"].strip()):
            raise SystemExit(
                f"ERROR: {pkg_dir.name}/package.yml: run_deps entry {d!r} must "
                f"be a conda spec string or a mapping {{if: <selector>, then: "
                f"<spec>}} with selector in {sorted(PLATFORM_SELECTORS)}.")


def _check_constrains(cfg: dict, pkg_dir: Path) -> None:
    """`constrains`: conda specs this package must never be installed beside.

    For two packages that install the SAME files -- a fork published under
    its own name without a Python-level rename (gsplat and gsplat-maskgaussian
    both ship gsplat/) -- a conda solve would otherwise let both in and the
    second silently overwrite the first's files. A `run_constraints` entry
    of the form `<other> <0.0a0` is unsatisfiable by any real version, so
    the solver refuses the pair outright instead. It is deliberately not a
    dependency: a constraint only bites when the other package is ALSO
    requested. Each entry must name a package (a bare spec string); the
    template renders the list verbatim into `requirements.run_constraints`.
    """
    v = cfg.get("constrains")
    if v is None:
        return
    if not isinstance(v, list) or not v or not all(
            isinstance(s, str) and s.strip() and " " in s.strip() for s in v):
        raise SystemExit(
            f"ERROR: {pkg_dir.name}/package.yml: constrains must be a non-empty "
            f"list of conda match specs WITH a version clause (e.g. "
            f"'gsplat-maskgaussian <0.0a0' to forbid co-installation), got {v!r}.")
    for s in v:
        if s.split()[0] == cfg.get("name"):
            raise SystemExit(
                f"ERROR: {pkg_dir.name}/package.yml: constrains names the package "
                f"itself ({s!r}).")


def _check_force_source_build(cfg: dict, pkg_dir: Path) -> None:
    """An upstream that can download a binary must say how that is disabled.

    This is layer L2 of the from-source guarantee (ARCHITECTURE.md). L1 (no
    network in the build script) is the real mechanism; this exists because
    one mechanism will not stay true across 42 packages and a year, and
    because the failure is silent: the build succeeds, the package installs,
    the kernels run -- they are just not ours.
    """
    repo = str(cfg.get("source_repo") or "").strip()
    required = PREBUILT_FETCHING_UPSTREAMS.get(repo)
    declared = cfg.get("force_source_build") or {}
    if not isinstance(declared, dict):
        raise SystemExit(
            f"ERROR: {pkg_dir.name}/package.yml: force_source_build must be a "
            f"mapping of env var -> value, got {type(declared).__name__}.")
    if not required:
        return
    for var, value in required.items():
        if var not in declared:
            raise SystemExit(
                f"ERROR: {pkg_dir.name}: {repo} downloads a prebuilt wheel "
                f"unless {var} is set, and package.yml does not declare it. "
                f"Add `force_source_build: {{{var}: \"{value}\"}}`.")
        if str(declared[var]) != value:
            raise SystemExit(
                f"ERROR: {pkg_dir.name}: force_source_build sets "
                f"{var}={declared[var]!r}, but upstream only accepts the "
                f"exact string {value!r} -- any other value silently permits "
                f"the prebuilt wheel. This is the defect that shipped "
                f"upstream's flash-attn binary from the earlier experiment.")


def _check_carry(cfg: dict, pkg_dir: Path) -> None:
    """Names conda-forge also ships need an explicit coverage decision.

    Under strict channel priority, carrying ANY build of a name hides
    conda-forge's builds of that name entirely, so a partial flavour set
    removes their coverage rather than adding to ours.
    """
    carry = cfg.get("carry")
    if carry is None:
        return
    if carry not in CARRY_VALUES:
        raise SystemExit(
            f"ERROR: {pkg_dir.name}/package.yml: carry: {carry!r} is not one "
            f"of {sorted(CARRY_VALUES)}.")


def _all_source_revs(cfg: dict) -> list:
    """Every revision this package can be built from, family map included."""
    revs = []
    if cfg.get("source_rev"):
        revs.append(str(cfg["source_rev"]).strip())
    for entry in (cfg.get("family_versions") or {}).values():
        if isinstance(entry, dict) and entry.get("source_rev"):
            revs.append(str(entry["source_rev"]).strip())
    return revs


def _check_family_versions(cfg: dict, pkg_dir: Path) -> None:
    """`family_versions` maps a TORCH version to this package's own version.

    torchvision and torchaudio version independently of torch (torchvision
    0.26.0 goes with torch 2.11.0), so a family package cannot state one
    `version`. The map is derived data -- torchvision's from each release's
    own `Requires-Dist: torch==X`, torchaudio's from upstream's release
    convention because it declares no torch dependency at all -- and a wrong
    entry silently produces an extension whose ABI does not match the torch it
    will be installed beside. So every entry must be complete and pinned.
    """
    fv = cfg.get("family_versions")
    if fv is None:
        return
    if not isinstance(fv, dict) or not fv:
        raise SystemExit(
            f"ERROR: {pkg_dir.name}/package.yml: family_versions must be a "
            f"non-empty mapping of torch version -> {{version, source_rev}}.")
    for torch_version, entry in fv.items():
        where = f"{pkg_dir.name}/package.yml: family_versions[{torch_version!r}]"
        if not isinstance(entry, dict):
            raise SystemExit(f"ERROR: {where} must be a mapping, got "
                             f"{type(entry).__name__}.")
        for field in ("version", "source_rev"):
            if not str(entry.get(field) or "").strip():
                raise SystemExit(f"ERROR: {where} is missing {field!r}.")
        if not re.fullmatch(r"\d+(\.\d+)+", str(torch_version)):
            raise SystemExit(
                f"ERROR: {where}: the KEY must be a torch version like "
                f"'2.11.0'; the package's own version goes in `version`.")


def _check_build_env(cfg: dict, pkg_dir: Path) -> None:
    """`build_env` is rendered into the build script as `export K="V"`.

    It exists for values that must be computed from $PREFIX, which the
    recipe's own `build.script.env` cannot express (rattler-build sets those
    literally, no shell expansion). Because the value lands inside double
    quotes in a generated shell script, a quote or a backtick in it would end
    the string and run whatever follows -- so the value is constrained here
    rather than trusted.
    """
    for field in ("build_env", "build_env_win"):
        env = cfg.get(field)
        if env is None:
            continue
        if not isinstance(env, dict) or not env:
            raise SystemExit(
                f"ERROR: {pkg_dir.name}/package.yml: {field} must be a non-empty "
                f"mapping of NAME -> value.")
        for k, v in env.items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(k)):
                raise SystemExit(
                    f"ERROR: {pkg_dir.name}/package.yml: {field} key {k!r} is not "
                    f"a shell variable name.")
            if re.search(r'["`$][({]|["`]|\\', str(v)):
                raise SystemExit(
                    f"ERROR: {pkg_dir.name}/package.yml: {field}[{k!r}] value "
                    f"{v!r} contains a quote, backtick, backslash or command "
                    f"substitution -- it is rendered into a shell script inside "
                    f'double quotes. Plain text and $VAR references only.')

    # There was a check here refusing a build_env_win key that build_env does
    # not also set, on the theory that the two platforms need the same variables
    # pointing at different places rather than different variables. That theory
    # was wrong within the hour: torchaudio needs CMAKE_LIBRARY_PATH on win-64
    # and only there, because the win-64 pytorch package keeps its one import
    # library somewhere upstream's find_library does not look, and no Linux
    # value for that variable would be correct or useful. A rule that forbids a
    # legitimate case in order to catch a hypothetical typo is the wrong trade,
    # and the typo it was guarding against fails the build loudly anyway.


def load_package(pkg_dir: Path) -> dict:
    """One package's flat config dict, overrides merged in."""
    cfg = yaml.safe_load((pkg_dir / "package.yml").read_text()) or {}
    _check_pre_build_not_redactable(cfg, pkg_dir)
    _check_parallelism_declared(cfg, pkg_dir)
    _check_dependencies_declared(cfg, pkg_dir)
    _check_shard_sources(cfg, pkg_dir)
    _check_force_source_build(cfg, pkg_dir)
    _check_constrains(cfg, pkg_dir)
    _check_carry(cfg, pkg_dir)

    for extra in ("arch_override.yml",):
        p = pkg_dir / extra
        if p.exists():
            cfg.update(yaml.safe_load(p.read_text()) or {})
    overrides = [e for e in ("arch_override.yml",) if (pkg_dir / e).exists()]
    if overrides:
        readme = pkg_dir / "README.md"
        if not readme.exists() or "verride" not in readme.read_text():
            raise SystemExit(
                f"ERROR: {pkg_dir.name}: has {', '.join(overrides)} but no "
                f"README.md explaining the override -- every deviation from "
                f"defaults/ must say why (add an '## Overrides' section).")

    _check_family_versions(cfg, pkg_dir)
    _check_build_env(cfg, pkg_dir)

    # A family package (torchvision, torchaudio) has no single version: its
    # version is a function of the torch it builds against, so the matrix
    # resolves both from `family_versions` per cell.
    required = ["name", "source_repo", "pypi_name", "import_name"]
    if not cfg.get("family_versions"):
        required += ["version", "source_rev"]
    for req in required:
        if not str(cfg.get(req) or "").strip():
            raise SystemExit(
                f"ERROR: {pkg_dir.name}: '{req}' is required in package.yml.")

    for rev in _all_source_revs(cfg):
        if rev.lower() in ("main", "master", "head"):
            raise SystemExit(
                f"ERROR: {pkg_dir.name}: source_rev is a floating ref ({rev!r}) "
                f"-- pin a tag or commit SHA, or two artifacts of one version "
                f"need not come from the same source.")

    if "links_torch" not in cfg:
        raise SystemExit(
            f"ERROR: {pkg_dir.name}: no links_torch declared -- state it "
            f"explicitly (true: one build per (cuda x torch); false: "
            f"torch-free, one build per cuda, and no torch axis at all).")

    if cfg["name"] != cfg["name"].lower() or "_" in cfg["name"]:
        raise SystemExit(
            f"ERROR: {pkg_dir.name}: conda name {cfg['name']!r} must be the "
            f"lowercase hyphenated PyPI name (flash-attn, not flash_attn) so "
            f"the purl maps and pixi's conda->pypi recognition works. The "
            f"underscore import name goes in import_name.")
    return cfg


def iter_packages():
    """Yield (folder, config) for every package folder, sorted."""
    for d in sorted(PACKAGES_DIR.iterdir()):
        if d.is_dir() and (d / "package.yml").exists():
            yield d.name, load_package(d)
