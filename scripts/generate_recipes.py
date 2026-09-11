#!/usr/bin/env python3
"""Render packages/<name>/package.yml -> recipes/<name>/recipe.yaml.

Generated recipes are COMMITTED so they stay reviewable and runnable by hand,
and CI runs this with --check to assert regeneration is a no-op. The prior
experiment hand-wrote three recipes that were ~90% identical text and had
already drifted apart (a force-source flag set in one and not the others, two
divergent Windows blocks); 42 of them would be worse.

Every file the build runs is generated beside the recipe and covered by
--check: recipe.yaml, build.sh and build_win.py (the shared scripts with this
package's build_env substituted in -- file-backed, so rattler-build never
renders them through minijinja), verify_op.py (package.yml verify.op, shipped
inside the artifact as its own test), and the verbatim siblings nvcc-wrap.sh
and nonet.py. The README's package table is generated the same way.

Usage:
    generate_recipes.py                 # regenerate all
    generate_recipes.py --package foo   # regenerate one
    generate_recipes.py --check         # fail if any committed recipe is stale
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "templates" / "recipe.yaml.j2"
BUILD_SH = REPO / "scripts" / "build_snippets" / "build.sh"
NVCC_WRAP = REPO / "scripts" / "build_snippets" / "nvcc-wrap.sh"
NONET = REPO / "scripts" / "build_snippets" / "nonet.py"
BUILD_WIN = REPO / "scripts" / "build_snippets" / "build_win.py"
README = REPO / "README.md"
README_BEGIN = "<!-- packages:begin -->"
README_END = "<!-- packages:end -->"


def load_packages(only: str | None) -> list[tuple[str, dict]]:
    """Package configs via the shared loader (scripts/package_loader.py)."""
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        from package_loader import PACKAGES_DIR, load_package  # type: ignore
    except ImportError:
        sys.exit("scripts/package_loader.py not found — it is the schema owner "
                 "for packages/*/package.yml and must exist before recipes can "
                 "be generated")
    out = []
    # Loaded one folder at a time so that --package <x> validates <x> alone: a
    # schema error in some other package must not stop this one rendering.
    for d in sorted(PACKAGES_DIR.iterdir()):
        if not d.is_dir() or not (d / "package.yml").exists():
            continue
        if only and only != d.name:
            continue
        out.append((d.name, load_package(d)))
    if only and not out:
        sys.exit(f"no package folder matching {only!r}")
    return out


def conda_name(cfg: dict) -> str:
    """Conda names are the hyphenated PyPI name.

    `purls` map a conda package to its PyPI identity, and that mapping is what
    stops a pixi solve installing a second copy from PyPI on top of ours. An
    underscore name (`flash_attn`) is not the PyPI project name and breaks it.
    """
    return cfg.get("conda_name") or (cfg.get("pypi_name") or cfg["name"]).replace("_", "-")


def _license(cfg: dict) -> str:
    """SPDX identifier, or a parseable placeholder plus a loud warning.

    rattler-build validates this against the SPDX list, so a bare "UNKNOWN"
    fails the build outright. Shipping a real artifact with an unspecified
    licence is also how conda-torch ended up redistributing NVIDIA binaries
    with the EULA stripped, so the gap is surfaced rather than papered over.
    """
    lic = cfg.get("license")
    if lic:
        return lic
    print(f"WARNING: {cfg['name']}: no `license:` in package.yml -- emitting "
          f"LicenseRef-Unspecified. Set the real SPDX identifier before this "
          f"package is published.", file=sys.stderr)
    return "LicenseRef-Unspecified"


def newest_family_pair(cfg: dict) -> tuple[str, str]:
    """(version, source_rev) of the newest torch pairing, as recipe defaults.

    A family recipe takes its version from the matrix at build time, but the
    committed file still needs a real default so it is reviewable and can be
    built by hand. The newest pairing is the least surprising choice.
    """
    fv = cfg["family_versions"]
    newest = max(fv, key=lambda v: [int(x) for x in v.split(".")])
    return fv[newest]["version"], fv[newest]["source_rev"]


def build_env_block(cfg: dict) -> str:
    """package.yml `build_env` -> shell exports, substituted into build.sh.

    These cannot go in the recipe's `build.script.env`: rattler-build sets
    those values literally, so a `$PREFIX/include` written there reaches the
    build as that string, not as a path. Rendering them into the script puts
    them where $PREFIX is a real directory. Values are emitted inside double
    quotes so $PREFIX expands and word-splitting cannot bite.
    """
    env = cfg.get("build_env") or {}
    if not env:
        return "# (none declared)"
    return "\n".join(f'export {k}="{v}"' for k, v in env.items())


def _build_sh(cfg: dict) -> str:
    """The shared build script with this package's build_env substituted in."""
    text = BUILD_SH.read_text().rstrip("\n")
    hook = "# CUW_BUILD_ENV_HOOK"
    if hook not in text:
        sys.exit("scripts/build_snippets/build.sh lost its # CUW_BUILD_ENV_HOOK "
                 "marker -- package.yml build_env would be silently dropped")
    return text.replace(hook, build_env_block(cfg))


def _build_win(cfg: dict) -> str:
    """The win-64 build script with this package's build_env substituted in.

    Deliberately parallel to _build_sh: a package declaring `build_env` must get
    it on both platforms or neither, and the failure mode if one is forgotten is
    identical and silent. The substitution is a Python dict literal on the line
    build_win.py marks with CUW_BUILD_ENV_HOOK; build_win.py applies it with
    os.path.expandvars, which on Windows expands `$VAR`, `${VAR}` and `%VAR%`
    alike -- so a value written once for both platforms (`$PREFIX/include`)
    needs no batch translation, and the .bat shim that used to do it is gone.

    build_env_win OVERRIDES build_env per variable, rather than replacing the
    block. The two platforms need the same variables pointing at different
    places: conda's Unix-shaped tree lives under %PREFIX%/Library on Windows,
    so `FFMPEG_ROOT: $PREFIX` is right on Linux and points at a directory
    with no include/ on win-64. torchaudio's CMake found no libavutil/avutil.h
    there and the whole build died at configure -- which is the good failure;
    the bad one is a package that silently builds without a backend it
    declares.
    """
    text = BUILD_WIN.read_text().rstrip("\n") + "\n"
    hook = "CUW_BUILD_ENV = {}  # CUW_BUILD_ENV_HOOK"
    if hook not in text:
        sys.exit("scripts/build_snippets/build_win.py lost its CUW_BUILD_ENV_HOOK "
                 "line -- package.yml build_env would be silently dropped on win-64")
    env = dict(cfg.get("build_env") or {})
    env.update(cfg.get("build_env_win") or {})
    literal = json.dumps({str(k): str(v) for k, v in env.items()}, sort_keys=True)
    return text.replace(hook, f"CUW_BUILD_ENV = {literal}  # from package.yml build_env")


def _verify_op(cfg: dict) -> str:
    """package.yml `verify.op` as a standalone script, shipped in the artifact.

    rattler-build copies it into info/tests/ so `rattler-build test
    --package-file <artifact>` can run the package's own minimal op on any
    machine with a GPU. The header records where it came from so a reader of
    the artifact does not have to find this repo to know what it asserts.
    """
    op = ((cfg.get("verify") or {}).get("op") or "").rstrip("\n")
    if not op:
        op = (f"import {cfg.get('import_name') or cfg['name']}  # no verify.op "
              f"declared; import is the whole test")
    return (f"# {cfg['name']}: the package's own minimal GPU op, from\n"
            f"# packages/{cfg['name']}/package.yml `verify.op`. Generated by\n"
            f"# scripts/generate_recipes.py; edit package.yml, not this file.\n"
            f"{op}\n")


def _readme_table(packages: list) -> str:
    """The README package table, between README_BEGIN and README_END.

    One row per package.yml: what it is, its licence, and -- the reason the
    table exists -- any distribution restriction its licence imposes, which
    a consumer must be able to see without opening an artifact.
    """
    rows = ["| package | version | license | PyPI purl | distribution restriction |",
            "|---|---|---|---|---|"]
    for _folder, cfg in packages:
        fam = cfg.get("family_versions")
        version = "per torch pairing" if fam else str(cfg.get("version", ""))
        purl = cfg.get("pypi_project") or "none"
        restriction = cfg.get("distribution_restriction") or ""
        if restriction:
            restriction = "**" + " ".join(restriction.split()) + "**"
        rows.append(f"| `{conda_name(cfg)}` | {version} | {cfg.get('license', '')} "
                    f"| {purl} | {restriction} |")
    return "\n".join(rows) + "\n"


def _readme_with_table(packages: list) -> str:
    text = README.read_text()
    if README_BEGIN not in text or README_END not in text:
        sys.exit(f"README.md lost its {README_BEGIN} / {README_END} markers -- "
                 f"the package table has nowhere to go")
    head, rest = text.split(README_BEGIN, 1)
    _old, tail = rest.split(README_END, 1)
    return f"{head}{README_BEGIN}\n{_readme_table(packages)}{README_END}{tail}"


def _dedupe(items: list) -> list:
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _yamlstr(value) -> str:
    """A YAML double-quoted scalar. jinja's tojson would escape `>` and `'` as
    \u003e and \u0027, which YAML reads correctly and no human does; a run
    dep selector like `match(pytorch, ">=2.7,<2.14")` should stay legible."""
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def render(folder: str, cfg: dict, env) -> str:
    tmpl = env.get_template(TEMPLATE.name)
    family = bool(cfg.get("family_versions"))
    if family:
        default_version, default_rev = newest_family_pair(cfg)
    else:
        default_version = cfg.get("version", "0.0.0")
        default_rev = cfg.get("source_rev") or cfg.get("source_tag", "")
    from package_loader import (HOST_RUN_EXPORT_SUBSUMES,  # type: ignore
                                IGNORED_CUDA_RUN_EXPORTS)
    verify = cfg.get("verify") or {}
    host_names = {d.split()[0] for d in (cfg.get("host_deps") or []) if isinstance(d, str)}
    run_deps = cfg.get("run_deps") or []
    subsumed = sorted(d for d in run_deps if isinstance(d, str)
                      and d.strip() in HOST_RUN_EXPORT_SUBSUMES and d.strip() in host_names)
    keep = set(cfg.get("keep_run_exports") or [])
    run_exports = cfg.get("run_exports")
    if isinstance(run_exports, list):
        run_exports = {"weak": run_exports}
    return tmpl.render(
        conda_name=conda_name(cfg),
        family=family,
        version=default_version,
        source_repo=cfg.get("source_repo", ""),
        source_rev=default_rev,
        arch_list=cfg.get("arch_list", ""),
        jobs=cfg["jobs"],
        nvcc_threads=cfg["nvcc_threads"],
        links_torch=cfg.get("links_torch", True),
        force_source_build=cfg.get("force_source_build") or {},
        host_deps=cfg.get("host_deps") or [],
        host_deps_linux=cfg.get("host_deps_linux") or [],
        host_deps_win=cfg.get("host_deps_win") or [],
        run_deps=cfg.get("run_deps") or [],
        constrains=cfg.get("constrains") or [],
        build_deps=cfg.get("build_deps") or [],
        shard_sources=cfg.get("shard_sources") or [],
        shard_partition=cfg.get("shard_partition") or "",
        build_subdir=cfg.get("build_subdir") or "",
        verify_imports=_dedupe([verify.get("import") or cfg.get("import_name") or cfg["name"]]
                               + list(verify.get("imports") or [])),
        op_requires=verify.get("op_requires") or [],
        allow_dso=verify.get("allow_dso") or [],
        subsumed_run_deps=subsumed,
        ignored_run_exports=[p for p in IGNORED_CUDA_RUN_EXPORTS if p not in keep],
        run_exports=run_exports or {},
        license_files=cfg["license_files"],
        repository=cfg.get("repository")
        or f"https://github.com/{cfg.get('source_repo', '')}",
        documentation=cfg.get("documentation") or "",
        pypi_project=cfg.get("pypi_project") or "",
        distribution_restriction=cfg.get("distribution_restriction") or "",
        homepage=cfg.get("homepage", f"https://github.com/{cfg.get('source_repo','')}"),
        license=_license(cfg),
        summary=cfg.get("summary", cfg["name"]),
    ) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package")
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if a committed recipe differs from a fresh render")
    args = ap.parse_args()

    try:
        from jinja2 import Environment, FileSystemLoader, StrictUndefined
    except ImportError:
        sys.exit("jinja2 is required: pip install jinja2")

    # Non-default delimiters: rattler-build recipes are themselves jinja and
    # use ${{ ... }} / {% ... %}. If the generator used the same markers it
    # would try to evaluate rattler's expressions at generation time and fail
    # on undefined build-time variables like cuda_compiler_version.
    env = Environment(loader=FileSystemLoader(str(TEMPLATE.parent)),
                      variable_start_string="<<", variable_end_string=">>",
                      block_start_string="<%", block_end_string="%>",
                      comment_start_string="<#", comment_end_string="#>",
                      undefined=StrictUndefined, keep_trailing_newline=True)
    env.filters["yamlstr"] = _yamlstr

    # Everything the build runs lives beside the recipe, because $RECIPE_DIR
    # is the only path the build can rely on inside the sandbox, and every one
    # of those files is covered by --check. It used to check recipe.yaml
    # alone, which meant a change to build_win.py, nvcc-wrap.sh or nonet.py
    # left every committed copy stale and CI green -- and the copy beside the
    # recipe is the one the build actually runs. An edit that was never
    # regenerated would simply not take effect, which is the hardest kind of
    # change to debug: the source says one thing and the build does another.
    #
    # Two kinds: verbatim siblings, and per-package renders (build.sh and
    # build_win.py carry the package's build_env; verify_op.py its op).
    verbatim = {"nvcc-wrap.sh": NVCC_WRAP, "nonet.py": NONET}
    executable = {"nvcc-wrap.sh", "nonet.py", "build.sh"}

    stale = []
    packages = load_packages(args.package)
    for folder, cfg in packages:
        out_dir = REPO / "recipes" / conda_name(cfg)
        generated = {
            "recipe.yaml": render(folder, cfg, env),
            "build.sh": _build_sh(cfg) + "\n",
            "build_win.py": _build_win(cfg),
            "verify_op.py": _verify_op(cfg),
        }
        generated.update({name: src.read_text() for name, src in verbatim.items()})
        # A stale build.bat from before the script became file-backed would be
        # picked up by rattler-build as the default win-64 script if the
        # recipe ever lost its `file:` line; it has no business existing.
        leftovers = [out_dir / "build.bat"]
        if args.check:
            for name, text in generated.items():
                path = out_dir / name
                have = path.read_text() if path.is_file() else ""
                if have != text:
                    stale.append(path.relative_to(REPO))
                    sys.stdout.writelines(difflib.unified_diff(
                        have.splitlines(True), text.splitlines(True),
                        fromfile=f"{path.relative_to(REPO)} (committed)",
                        tofile=f"{path.relative_to(REPO)} (regenerated)"))
            for left in leftovers:
                if left.exists():
                    stale.append(left.relative_to(REPO))
                    print(f"{left.relative_to(REPO)} should not exist")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, text in generated.items():
            path = out_dir / name
            path.write_text(text)
            if name in executable:
                path.chmod(0o755)
        for left in leftovers:
            if left.exists():
                left.unlink()
        print(out_dir.relative_to(REPO) / "recipe.yaml")

    # The README package table, from the same package set. Only when every
    # package was loaded: a --package run renders one recipe, not a table
    # that would silently drop the other 43.
    if not args.package:
        want = _readme_with_table(packages)
        if args.check:
            if README.read_text() != want:
                stale.append(README.relative_to(REPO))
                print("README.md package table is stale")
        elif README.read_text() != want:
            README.write_text(want)
            print("README.md (package table)")

    if stale:
        print(f"\n{len(stale)} recipe(s) are stale — run scripts/generate_recipes.py "
              f"and commit the result", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
