"""Patch mmcv v1.7.2: always build the CUDA ops, keep the name `mmcv`, and
survive modern python/setuptools.

Ported from cuda-wheels (packages/mmcv/patches/mmcv.py) and re-read against
the pinned rev. Runs with cwd = the cloned source, once, before the tarball is
sealed; nothing here depends on the cell.

1. MMCV_WITH_OPS. setup.py:200 `if os.getenv('MMCV_WITH_OPS', '0') == '0':
   return extensions` -- the default is a pure-Python wheel with no CUDA
   compile at all, and setup.py:544 flips the distribution name to
   `mmcv-full` when ops are on. PyPI's 1.x convention is the other way round
   (`mmcv` is the full build), so the env var is forced at the top of
   setup.py and the name expression is replaced with the literal 'mmcv'.
   The env var could equally be declared in package.yml, but the name flip
   cannot, and the two belong together.

2. get_version() execs mmcv/version.py and reads `locals()['__version__']`,
   which PEP 667 (python 3.13) turned into a KeyError. Exec into an explicit
   namespace instead; identical on 3.12.

3. `from pkg_resources import ...` at module scope: setuptools 82 removed
   pkg_resources, and the recipe's Linux host env takes whatever setuptools
   conda-forge has today. Shim it onto importlib.metadata + packaging.

4. Hardcoded C++ standard flags (-std=c++14/17, /std:c++14/17) in several
   places. torch's cpp_extension appends the standard the installed torch
   needs only when the caller gave none; an explicit pin overrides it, and
   `-std=c++14` in particular is below torch's own C++17 floor. Strip them
   all and let torch choose.

Every substitution asserts on its own -- a whole-file before/after check
passes as long as ANY edit landed (the fused-ssim lesson, docs/WINDOWS.md).
"""
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import require, strip_std_flags  # noqa: E402

MARKER = "# patched-by: cuda-foundry mmcv (MMCV_WITH_OPS=1, name=mmcv)"
setup_file = pathlib.Path("setup.py")
text = setup_file.read_text()

if MARKER in text:
    print("mmcv patch: already applied")
    sys.exit(0)

# 1. force ops on, keep the name -----------------------------------------
text = f"{MARKER}\nimport os\nos.environ['MMCV_WITH_OPS'] = '1'\n\n" + text
name_re = re.compile(
    r"name\s*=\s*'mmcv'\s+if\s+os\.getenv\(\s*'MMCV_WITH_OPS'\s*,\s*'0'\s*\)"
    r"\s*==\s*'0'\s+else\s+'mmcv-full'")
text, n = name_re.subn("name='mmcv'", text)
require(n == 1, f"mmcv: the conditional name= expression matched {n} times, "
                f"expected 1 -- the wheel would be published as mmcv-full")

# 2. PEP 667 --------------------------------------------------------------
old_ver = """def get_version():
    version_file = 'mmcv/version.py'
    with open(version_file, encoding='utf-8') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']"""
new_ver = """def get_version():
    version_file = 'mmcv/version.py'
    _ns = {}
    with open(version_file, encoding='utf-8') as f:
        exec(compile(f.read(), version_file, 'exec'), _ns)
    return _ns['__version__']"""
require(old_ver in text, "mmcv: get_version() does not match the v1.7.2 text")
text = text.replace(old_ver, new_ver, 1)

# 3. pkg_resources --------------------------------------------------------
old_pr = ("from pkg_resources import DistributionNotFound, get_distribution, "
          "parse_version")
new_pr = """try:  # setuptools >= 82 removed pkg_resources
    from pkg_resources import DistributionNotFound, get_distribution, parse_version
except ModuleNotFoundError:  # cuda-foundry shim
    from importlib.metadata import PackageNotFoundError as DistributionNotFound
    from importlib.metadata import distribution as _distribution
    from packaging.version import parse as parse_version

    class _Dist:
        def __init__(self, d):
            self.version = d.version

    def get_distribution(name):
        return _Dist(_distribution(name))"""
require(old_pr in text, "mmcv: the pkg_resources import line was not found")
text = text.replace(old_pr, new_pr, 1)

# 4. C++ standard ---------------------------------------------------------
text, n_std = strip_std_flags(text)
require(n_std > 0, "mmcv: no hardcoded C++-standard flag found in setup.py -- "
                   "upstream changed; refusing to build against an unverified "
                   "flag set")

setup_file.write_text(text)
final = setup_file.read_text()
for needle, what in (("os.environ['MMCV_WITH_OPS'] = '1'", "MMCV_WITH_OPS prologue"),
                     ("name='mmcv'", "literal name"),
                     ("_ns['__version__']", "PEP 667 get_version"),
                     ("cuda-foundry shim", "pkg_resources shim")):
    require(needle in final, f"mmcv: {what} is not present in setup.py on disk")
require(not re.search(r"""['"](?:-Xcompiler=)?[-/]std[=:]c\+\+\d+['"]""", final),
        "mmcv: a C++-standard flag survived stripping")
print(f"mmcv patch: ops forced on, name pinned, PEP 667 + pkg_resources shims, "
      f"{n_std} std flag(s) dropped")
