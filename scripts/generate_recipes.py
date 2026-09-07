#!/usr/bin/env python3
"""Render packages/<name>/package.yml -> recipes/<name>/recipe.yaml.

Generated recipes are COMMITTED so they stay reviewable and runnable by hand,
and CI runs this with --check to assert regeneration is a no-op. The prior
experiment hand-wrote three recipes that were ~90% identical text and had
already drifted apart (a force-source flag set in one and not the others, two
divergent Windows blocks); 42 of them would be worse.

Usage:
    generate_recipes.py                 # regenerate all
    generate_recipes.py --package foo   # regenerate one
    generate_recipes.py --check         # fail if any committed recipe is stale
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "templates" / "recipe.yaml.j2"
BUILD_SH = REPO / "scripts" / "build_snippets" / "build.sh"
NVCC_WRAP = REPO / "scripts" / "build_snippets" / "nvcc-wrap.sh"
NONET = REPO / "scripts" / "build_snippets" / "nonet.py"
BUILD_BAT = REPO / "scripts" / "build_snippets" / "build.bat"
BUILD_WIN = REPO / "scripts" / "build_snippets" / "build_win.py"


def load_packages(only: str | None) -> list[tuple[str, dict]]:
    """Package configs via the shared loader (scripts/package_loader.py)."""
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        from package_loader import iter_packages  # type: ignore
    except ImportError:
        sys.exit("scripts/package_loader.py not found — it is the schema owner "
                 "for packages/*/package.yml and must exist before recipes can "
                 "be generated")
    out = []
    for folder, cfg in iter_packages():
        if only and only not in (folder, cfg.get("name")):
            continue
        out.append((folder, cfg))
    if only and not out:
        sys.exit(f"no package matching {only!r}")
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


def _build_bat(cfg: dict) -> str:
    """The win-64 entry point with this package's build_env substituted in.

    Deliberately parallel to _build_sh: a package declaring `build_env` must get
    it on both platforms or neither. The two hooks render different syntax --
    `export K="V"` against `set "K=V"` -- so the substitution cannot be shared,
    but the failure mode if one is forgotten is identical and silent, which is
    why both are checked.
    """
    text = BUILD_BAT.read_text().rstrip("\n")
    hook = ":: CUW_BUILD_ENV_HOOK"
    if hook not in text:
        sys.exit("scripts/build_snippets/build.bat lost its :: CUW_BUILD_ENV_HOOK "
                 "marker -- package.yml build_env would be silently dropped on win-64")
    # build_env_win OVERRIDES build_env per variable, rather than replacing the
    # block. The two platforms need the same variables pointing at different
    # places: conda's Unix-shaped tree lives under %PREFIX%\Library on Windows,
    # so `FFMPEG_ROOT: $PREFIX` is right on Linux and points at a directory
    # with no include/ on win-64. torchaudio's CMake found no libavutil/avutil.h
    # there and the whole build died at configure -- which is the good failure;
    # the bad one is a package that silently builds without a backend it
    # declares. The loader refuses a win override for a variable Linux does not
    # set, so this cannot become a place to hide a Windows-only variable.
    env = dict(cfg.get("build_env") or {})
    env.update(cfg.get("build_env_win") or {})
    block = "\n".join(f'set "{k}={_bat_value(v)}"' for k, v in env.items()) \
        or ":: (no build_env declared)"
    return text.replace(hook, block)


_SHELL_VAR = re.compile(r"\$\{(\w+)\}|\$(\w+)")


def _bat_value(value) -> str:
    """Translate a build_env value's shell variable references to batch syntax.

    package.yml is written once for both platforms, and its values reference the
    build environment the way build.sh does -- `$PREFIX`, `${SRC_DIR}`. Copied
    verbatim into a .bat that sets the LITERAL string "$PREFIX", silently, and a
    build configured against a path that does not exist is a far worse failure
    than one that stops here. torchaudio's `FFMPEG_ROOT: $PREFIX` is the case
    that surfaced it.

    Only simple `$VAR` and `${VAR}` are handled, which is all that can appear:
    package_loader rejects quotes, backticks, backslashes and command
    substitution in build_env values before this ever runs. Anything with a
    dollar still in it afterwards is refused rather than guessed at.

    Path separators are deliberately NOT rewritten. Windows accepts forward
    slashes in paths, and blanket-converting them would corrupt any value that
    is not a path.
    """
    out = _SHELL_VAR.sub(lambda m: f"%{m.group(1) or m.group(2)}%", str(value))
    if "$" in out:
        sys.exit(f"package.yml build_env value {value!r} still contains '$' after "
                 f"translating to batch syntax -- it cannot be rendered for win-64. "
                 f"Use a plain $VAR or ${{VAR}} reference.")
    return out


def render(folder: str, cfg: dict, env) -> str:
    tmpl = env.get_template(TEMPLATE.name)
    family = bool(cfg.get("family_versions"))
    if family:
        default_version, default_rev = newest_family_pair(cfg)
    else:
        default_version = cfg.get("version", "0.0.0")
        default_rev = cfg.get("source_rev") or cfg.get("source_tag", "")
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
        run_deps=cfg.get("run_deps") or [],
        build_deps=cfg.get("build_deps") or [],
        import_name=cfg.get("import_name") or cfg["name"],
        homepage=cfg.get("homepage", f"https://github.com/{cfg.get('source_repo','')}"),
        license=_license(cfg),
        summary=cfg.get("summary", cfg["name"]),
        build_sh=_build_sh(cfg),
        build_bat=_build_bat(cfg),
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

    stale = []
    for folder, cfg in load_packages(args.package):
        text = render(folder, cfg, env)
        out = REPO / "recipes" / conda_name(cfg) / "recipe.yaml"
        if args.check:
            have = out.read_text() if out.is_file() else ""
            if have != text:
                stale.append(out.relative_to(REPO))
                sys.stdout.writelines(difflib.unified_diff(
                    have.splitlines(True), text.splitlines(True),
                    fromfile=f"{out.relative_to(REPO)} (committed)",
                    tofile=f"{out.relative_to(REPO)} (regenerated)"))
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        # The wrapper sits beside the recipe so $RECIPE_DIR/nvcc-wrap.sh
        # resolves inside the build sandbox, where scripts/ is not present.
        (out.parent / "nvcc-wrap.sh").write_text(NVCC_WRAP.read_text())
        (out.parent / "nvcc-wrap.sh").chmod(0o755)
        # nonet.py must sit beside the recipe too: $RECIPE_DIR is the only
        # path the build script can rely on inside the build.
        (out.parent / "nonet.py").write_text(NONET.read_text())
        (out.parent / "nonet.py").chmod(0o755)
        # Same reason as nonet.py: $RECIPE_DIR is the only path build.bat can
        # rely on inside the build. Kept a sibling file rather than embedded
        # because rattler-build renders the embedded script through minijinja,
        # and brace-hash / brace-brace / brace-percent would break the render.
        (out.parent / "build_win.py").write_text(BUILD_WIN.read_text())
        print(out.relative_to(REPO))

    if stale:
        print(f"\n{len(stale)} recipe(s) are stale — run scripts/generate_recipes.py "
              f"and commit the result", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
