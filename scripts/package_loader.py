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
  * `license_files` (required), `pypi_project`, `repository`,
    `documentation`, `distribution_restriction`, `keep_run_exports`,
    `run_exports` and `verify.allow_dso` / `verify.op_requires` came out of
    the 2026-09 audits of the published channel; each check below says which
    finding it closes.

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

# Host packages whose run_exports the template IGNORES by default. Every
# torch-linked extension needs their HEADERS (ATen/cuda/CUDAContextLight.h
# includes <cusparse.h>, and the CUDAContext.h chain pulls the rest), so they
# sit in host: -- but their run_exports then declare libcublas, libcufft,
# libcurand, libcusolver and libcusparse on every artifact, and not one binary
# in the channel links any of them (measured on the published artifacts: the
# DT_NEEDED sets carry libcudart and libtorch's own libraries, nothing else;
# the four audits of the 88 published artifacts all cited this). cuda-nvtx is
# header-only since v3, and libnvrtc is linked by exactly one package (cumm).
# A package whose binary GENUINELY links one of these names it in
# `keep_run_exports:` and the template leaves that export in place.
IGNORED_CUDA_RUN_EXPORTS = [
    "libcublas-dev", "libcufft-dev", "libcurand-dev", "libcusolver-dev",
    "libcusparse-dev", "cuda-nvtx-dev", "cuda-nvrtc-dev",
]

# Host packages whose run_export SUBSUMES a bare run dep of the same name.
# numpy's own export is `numpy >=1.25,<3` (its ABI window), so a package that
# lists `numpy` in host_deps AND a bare `numpy` in run_deps would ship both --
# cumm did -- and the bare one adds nothing. The template drops the bare entry
# for these names only; anything else in run_deps is rendered verbatim, because
# most host packages export nothing and dropping a bare run dep for one of
# those would silently lose a real dependency (sympy, for one).
HOST_RUN_EXPORT_SUBSUMES = {"numpy"}

# Platform selectors a conditional run dep may use. rattler-build's own
# vocabulary, so the template renders them as `- if: <sel>` unchanged.
PLATFORM_SELECTORS = {"linux", "win", "unix", "osx"}

# The full selector grammar for a conditional run dep: a platform selector,
# a torch-minor clause, or both joined by `and`. The torch clause is
# rattler-build's own `match(<variant>, "<spec>")` builtin, evaluated on the
# `pytorch` variant ("2.8"), so the recipe renders it verbatim and the wheel
# side (resolve_run_deps) evaluates the same spec in Python. It exists for a
# dependency that conda-forge builds per torch minor and not for every minor
# this family grid has a torch for -- torchvision-extra-decoders exists for
# torch 2.5.1 through 2.13 and not for 2.4.x or 2.14 -- so an unconditional
# entry would make cells UNSAT and no entry at all loses a real dependency.
_SELECTOR_RE = re.compile(
    r"^(?:(?P<plat>linux|win|unix|osx)(?:\s+and\s+(?=match))?)?"
    r"(?:match\(\s*pytorch\s*,\s*['\"](?P<spec>[^'\"]+)['\"]\s*\))?$")
_SPEC_CLAUSE_RE = re.compile(r"^(>=|<=|==|!=|>|<)\s*(\d+(?:\.\d+)*)$")

# Conda subdirs a package may publish its .conda on (package.yml
# `conda_platforms`). The wheel is built on every platform the matrix emits
# regardless; this only gates the .conda. flex-gemm's case: it imports
# triton unconditionally and triton has no win-64 conda build, so its win-64
# .conda would be UNSAT-or-unimportable while the win-64 wheel works beside
# PyPI's triton-windows.
KNOWN_SUBDIRS = {"linux-64", "linux-aarch64", "win-64"}


def parse_selector(sel: str):
    """(platform-or-None, spec-or-None) for a run_deps `if:` string, else None."""
    m = _SELECTOR_RE.match(str(sel).strip())
    if not m or (m.group("plat") is None and m.group("spec") is None):
        return None
    return m.group("plat"), m.group("spec")


def _version_tuple(v: str) -> tuple:
    return tuple(int(x) for x in str(v).split("."))


def version_matches(version: str, spec: str) -> bool:
    """Does a dotted version satisfy a comma-joined spec (">=2.7,<2.14")?

    Numeric dotted versions and the six comparison operators only -- the
    subset a torch-minor clause needs, evaluated the way rattler-build's
    match() evaluates it on the variant value. Missing trailing components
    compare as zero (2.8 == 2.8.0), which is how conda compares them too.
    """
    have = _version_tuple(version)
    for clause in str(spec).split(","):
        m = _SPEC_CLAUSE_RE.match(clause.strip())
        if not m:
            raise ValueError(f"unsupported version clause {clause!r} in {spec!r}")
        op, want = m.group(1), _version_tuple(m.group(2))
        n = max(len(have), len(want))
        a, b = have + (0,) * (n - len(have)), want + (0,) * (n - len(want))
        ok = {">=": a >= b, "<=": a <= b, "==": a == b, "!=": a != b,
              ">": a > b, "<": a < b}[op]
        if not ok:
            return False
    return True


def resolve_run_deps(run_deps, platform: str, pytorch: str | None = None) -> list:
    """Flatten conditional run deps for one target platform (conda subdir).

    `{if: linux, then: triton}` contributes "triton" on linux-64/aarch64 and
    nothing on win-64; `{if: 'linux and match(pytorch, ">=2.7,<2.14")', then:
    torchvision-extra-decoders}` contributes it on linux for torch minors in
    that window; plain strings pass through. The wheel side needs this
    because its sidecar is written from package.yml, not from the rendered
    recipe, and must say the same thing the .conda says for that cell -- so
    a torch clause needs the cell's torch minor, and refusing to guess is
    the only honest answer when it is not given.
    """
    fam = "win" if platform.startswith("win") else (
        "osx" if platform.startswith("osx") else "linux")
    out = []
    for d in run_deps or []:
        if isinstance(d, dict):
            plat, spec = parse_selector(d["if"])
            if plat is not None and not (plat == fam or (plat == "unix" and fam != "win")):
                continue
            if spec is not None:
                if pytorch is None:
                    raise ValueError(
                        f"run dep {d['then']!r} is conditional on the torch minor "
                        f"({d['if']!r}) and no pytorch version was given to "
                        f"resolve it for; pass the cell's torch minor.")
                if not version_matches(pytorch, spec):
                    continue
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
        parsed = parse_selector(d["if"]) if isinstance(d, dict) and "if" in d else None
        if (not isinstance(d, dict) or set(d) != {"if", "then"} or parsed is None
                or not isinstance(d["then"], str) or not d["then"].strip()):
            raise SystemExit(
                f"ERROR: {pkg_dir.name}/package.yml: run_deps entry {d!r} must "
                f"be a conda spec string or a mapping {{if: <selector>, then: "
                f"<spec>}} where the selector is one of {sorted(PLATFORM_SELECTORS)}, "
                f"a torch-minor clause `match(pytorch, \">=2.7,<2.14\")`, or "
                f"both joined by ` and `.")
        _plat, spec = parsed
        if spec is not None:
            if not cfg.get("links_torch", True):
                raise SystemExit(
                    f"ERROR: {pkg_dir.name}/package.yml: run_deps entry {d!r} "
                    f"is conditional on the torch minor, but this package is "
                    f"links_torch: false and has no torch axis.")
            try:
                version_matches("2.8", spec)
            except ValueError as exc:
                raise SystemExit(
                    f"ERROR: {pkg_dir.name}/package.yml: run_deps entry {d!r}: "
                    f"{exc}") from None


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


def _check_license_files(cfg: dict, pkg_dir: Path) -> None:
    """`license_files` is mandatory: the paths rattler-build ships into
    info/licenses/.

    An artifact that redistributes a compiled upstream without its licence
    text is the same defect conda-torch had when it repacked NVIDIA binaries
    with the EULA stripped, and every one of the six audits of this channel
    flagged that no artifact here carries info/licenses/ at all. Paths are
    relative to the source root (the fetched, patched tree), one entry per
    file, and a vendored third-party licence counts:

        license_files: [LICENSE]
        license_files: [LICENSE, third_party/cutlass/LICENSE.txt]
    """
    v = cfg.get("license_files")
    if v is None or (isinstance(v, list) and not v):
        raise SystemExit(
            f"ERROR: {pkg_dir.name}/package.yml does not declare license_files. "
            f"List the licence file(s) rattler-build must ship into "
            f"info/licenses/, as paths relative to the source root, e.g. "
            f"`license_files: [LICENSE]` or "
            f"`license_files: [LICENSE, third_party/cutlass/LICENSE.txt]`. "
            f"Read the pinned revision to find them; an artifact that "
            f"redistributes upstream without its licence text may not be "
            f"published.")
    if not isinstance(v, list) or not all(isinstance(x, str) and x.strip() for x in v):
        raise SystemExit(
            f"ERROR: {pkg_dir.name}/package.yml: license_files must be a list of "
            f"non-empty relative paths, got {v!r}.")
    for x in v:
        if x.startswith("/") or x.startswith("..") or "\\" in x:
            raise SystemExit(
                f"ERROR: {pkg_dir.name}/package.yml: license_files entry {x!r} "
                f"must be a forward-slash path relative to the source root.")


def _check_about_links(cfg: dict, pkg_dir: Path) -> None:
    """`repository` / `documentation`: optional URLs for about: in the recipe.

    `repository` defaults to the GitHub URL of source_repo; `documentation`
    has no default because guessing one (docs.<name>.io) invents a link.
    """
    for key in ("repository", "documentation"):
        v = cfg.get(key)
        if v is None:
            continue
        if not isinstance(v, str) or not re.match(r"https?://\S+$", v.strip()):
            raise SystemExit(
                f"ERROR: {pkg_dir.name}/package.yml: {key} must be an http(s) "
                f"URL, got {v!r}.")


def _check_cuda_host_deps(cfg: dict, pkg_dir: Path) -> None:
    """The CUDA runtime -dev package may not be a host dep, and here is why.

    The toolkit is pinned to the cell in build:, where the solve has no torch
    in it. host: is a different solve: it holds pytorch, and pytorch's triton
    pin drags host's `cuda-version` to whatever triton was built for (12.9
    for the cu128 line, measured). A `cuda-cudart-dev` listed in host_deps
    therefore resolves to 12.9 and its run_export stamps
    `cuda-cudart >=12.9.79` onto an artifact labelled cuda128 -- which is
    what every artifact in the channel carried before this check, and what
    every audit cited first. Nothing needs it there: the headers come from
    the build env (build.sh bridges $BUILD_PREFIX/targets/<arch>/include into
    $CUDA_HOME/include and puts it FIRST on the include path; on win-64
    nvcc.profile and build_win.py's INCLUDE/LIB prepend do the same), the
    link resolves libcudart.so through the build env's dev symlink, and the
    runtime dep is declared explicitly by the template with the cell's own
    floor (`cuda-cudart >=<cuda_compiler_version>,<<major+1>.0a0`).

    `keep_run_exports:` is the escape for the OTHER ignored -dev packages
    (IGNORED_CUDA_RUN_EXPORTS): a package whose binary genuinely links one of
    them keeps that export. cumm links libnvrtc, and is the only one today.
    """
    for key in ("host_deps", "host_deps_linux", "host_deps_win"):
        for dep in cfg.get(key) or []:
            if isinstance(dep, str) and dep.split()[0] == "cuda-cudart-dev":
                raise SystemExit(
                    f"ERROR: {pkg_dir.name}/package.yml lists cuda-cudart-dev in "
                    f"{key}. Remove it: host's cuda-version floats to whatever "
                    f"pytorch's triton needs (12.9 for the cu128 line), so a host "
                    f"cuda-cudart-dev exports `cuda-cudart >=12.9.x` onto a "
                    f"cuda128 artifact. The CUDA headers and the libcudart link "
                    f"stub come from build: (cuda-cudart-dev is already there, "
                    f"pinned to the cell), and the template declares the "
                    f"`cuda-cudart` run dep itself with the cell's floor.")
    keep = cfg.get("keep_run_exports")
    if keep is None:
        return
    if not isinstance(keep, list) or not keep or not all(
            isinstance(x, str) and x.strip() for x in keep):
        raise SystemExit(
            f"ERROR: {pkg_dir.name}/package.yml: keep_run_exports must be a "
            f"non-empty list of package names, got {keep!r}.")
    unknown = [x for x in keep if x not in IGNORED_CUDA_RUN_EXPORTS]
    if unknown:
        raise SystemExit(
            f"ERROR: {pkg_dir.name}/package.yml: keep_run_exports names "
            f"{unknown}, which the template does not ignore in the first place "
            f"(it ignores exactly {IGNORED_CUDA_RUN_EXPORTS}). A host package "
            f"outside that list keeps its run_exports by default.")


def normalize_pypi(name: str) -> str:
    """PEP 503 normalisation: case-folded, runs of [-_.] collapsed to '-'."""
    return re.sub(r"[-_.]+", "-", str(name)).lower()


def _check_pypi_project(cfg: dict, pkg_dir: Path) -> None:
    """`pypi_project`: the PyPI project this artifact GENUINELY provides.

    tools/fragment.py and tools/make_repodata.py emit a `pkg:pypi/<project>`
    purl only from this field. It used to be derived from pypi_name for every
    package, and about 20 of those purls were false: 404 on PyPI, or a
    DIFFERENT project that happens to share the name (`nunchaku` on PyPI is a
    data-segmentation library, `drtk` is a squatted junk package, `cumesh` is
    someone else's, `mmcv` on PyPI is the ops-less distribution). A purl is a
    claim that pixi's conda->pypi map may act on -- it stops a second copy
    being installed from PyPI -- so a wrong one is worse than none.

    Unset (null) means "no purl": the safe default until the owner has checked
    PyPI. When set it must match pypi_name after PEP 503 normalisation, or the
    package must say why in `pypi_project_differs_because:`.
    """
    proj = cfg.get("pypi_project")
    why = cfg.get("pypi_project_differs_because")
    if proj is None:
        if why:
            raise SystemExit(
                f"ERROR: {pkg_dir.name}/package.yml sets "
                f"pypi_project_differs_because but no pypi_project.")
        return
    if not isinstance(proj, str) or not proj.strip():
        raise SystemExit(
            f"ERROR: {pkg_dir.name}/package.yml: pypi_project must be a PyPI "
            f"project name or null, got {proj!r}.")
    if normalize_pypi(proj) != normalize_pypi(cfg.get("pypi_name") or ""):
        if not isinstance(why, str) or not why.strip():
            raise SystemExit(
                f"ERROR: {pkg_dir.name}/package.yml: pypi_project {proj!r} does "
                f"not match pypi_name {cfg.get('pypi_name')!r} (PEP 503 "
                f"normalised). If that is deliberate, say why in "
                f"`pypi_project_differs_because:`; otherwise fix one of them.")


def _check_distribution_restriction(cfg: dict, pkg_dir: Path) -> None:
    """`distribution_restriction`: free text, rendered into about.extra and
    surfaced in the README package table.

    For a licence that forbids distribution somewhere (custom-rasterizer-hy3d2's
    forbids the EU, the UK and South Korea). Whether such a package may be on
    the channel at all is the owner's legal decision; this field makes the
    restriction visible in the artifact and the README rather than deciding it.
    """
    v = cfg.get("distribution_restriction")
    if v is None:
        return
    if not isinstance(v, str) or not v.strip():
        raise SystemExit(
            f"ERROR: {pkg_dir.name}/package.yml: distribution_restriction must "
            f"be a non-empty string (or absent), got {v!r}.")


def _check_run_exports(cfg: dict, pkg_dir: Path) -> None:
    """`run_exports`: what a package this repo builds exports to ITS consumers.

    Either a list (rendered as weak exports) or a mapping with `weak:` and/or
    `strong:` lists. cumm is the case: `cumm >=0.7.11,<0.8.0 cuda128_*` as a
    weak export lets spconv -- and anyone else -- inherit the flavour glob
    instead of hand-writing it, the same argument ARCHITECTURE.md makes about
    conda-torch's pytorch. Written as recipe jinja if it must follow the cell.
    """
    v = cfg.get("run_exports")
    if v is None:
        return
    if isinstance(v, list):
        v = {"weak": v}
    if (not isinstance(v, dict) or not v
            or set(v) - {"weak", "strong"}
            or not all(isinstance(lst, list) and lst and all(
                isinstance(s, str) and s.strip() for s in lst) for lst in v.values())):
        raise SystemExit(
            f"ERROR: {pkg_dir.name}/package.yml: run_exports must be a list of "
            f"specs (weak) or a mapping of weak:/strong: -> non-empty lists of "
            f"specs, got {cfg.get('run_exports')!r}.")


def _check_conda_platforms(cfg: dict, pkg_dir: Path) -> None:
    """`conda_platforms`: the subdirs the .conda is PUBLISHED on.

    The wheel is still built and published for every platform the matrix
    emits; this only withholds the .conda where it could not be installed or
    imported -- flex-gemm imports triton at load and triton has no win-64
    conda build. generate_matrix.py turns it into the job's `publish_conda`
    flag, which the workflow's upload and fragment steps gate on.
    """
    v = cfg.get("conda_platforms")
    if v is None:
        return
    if not isinstance(v, list) or not v or not all(isinstance(x, str) for x in v):
        raise SystemExit(
            f"ERROR: {pkg_dir.name}/package.yml: conda_platforms must be a "
            f"non-empty list of conda subdirs, got {v!r}.")
    unknown = sorted(set(v) - KNOWN_SUBDIRS)
    if unknown:
        raise SystemExit(
            f"ERROR: {pkg_dir.name}/package.yml: conda_platforms names {unknown}; "
            f"known subdirs are {sorted(KNOWN_SUBDIRS)}.")


def _check_verify(cfg: dict, pkg_dir: Path) -> None:
    """The parts of `verify` the recipe renders (the rest is verify_conda's).

    `verify.op` becomes tests/verify_op.py INSIDE the artifact, so
    `rattler-build test --package-file` can run it anywhere; `verify.import`
    feeds the imports test; `verify.allow_dso` is rattler-build's
    missing_dso_allowlist (a binary that links libcuda.so.1 / nvcuda.dll
    links a driver library no conda package provides); `verify.op_requires`
    lists packages the op needs beyond the artifact's own run deps;
    `verify.imports` lists the submodules that must import on a GPU-less
    runner (the python test renders import_name first, then these).
    """
    v = cfg.get("verify") or {}
    if not isinstance(v, dict):
        raise SystemExit(f"ERROR: {pkg_dir.name}/package.yml: verify must be a mapping.")
    for key in ("allow_dso", "op_requires", "imports"):
        lst = v.get(key)
        if lst is None:
            continue
        if not isinstance(lst, list) or not lst or not all(
                isinstance(s, str) and s.strip() for s in lst):
            raise SystemExit(
                f"ERROR: {pkg_dir.name}/package.yml: verify.{key} must be a "
                f"non-empty list of strings, got {lst!r}.")
    op = v.get("op")
    if op is not None and (not isinstance(op, str) or not op.strip()):
        raise SystemExit(
            f"ERROR: {pkg_dir.name}/package.yml: verify.op must be python source.")
    imp = v.get("import")
    if imp is not None and (not isinstance(imp, str) or not imp.strip()):
        raise SystemExit(
            f"ERROR: {pkg_dir.name}/package.yml: verify.import must be a module name.")


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
    _check_license_files(cfg, pkg_dir)
    _check_about_links(cfg, pkg_dir)
    _check_cuda_host_deps(cfg, pkg_dir)
    _check_pypi_project(cfg, pkg_dir)
    _check_distribution_restriction(cfg, pkg_dir)
    _check_run_exports(cfg, pkg_dir)
    _check_conda_platforms(cfg, pkg_dir)
    _check_verify(cfg, pkg_dir)

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
