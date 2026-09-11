"""Give simple-knn the licence text its sources say they are under.

Runs with cwd = the cloned camenduru/simple-knn checkout, once, before the
tarball is sealed (scripts/fetch_patched_sources.py) -- where the network
exists, which the build deliberately does not have.

The repo carries NO licence file. Every source header reads

    Copyright (C) 2023, Inria
    GRAPHDECO research group, https://team.inria.fr/graphdeco
    All rights reserved.
    This software is free for non-commercial, research and evaluation use
    under the terms of the LICENSE.md file.

and that LICENSE.md is the Inria/MPII "Gaussian-Splatting License" of
graphdeco-inria/gaussian-splatting, whose submodules/simple-knn this code
was extracted from. package.yml names it LicenseRef-Gaussian-Splatting-License;
the published artifact carried the name and no text, in either output.

So the text is fetched from the upstream that owns it, at the last commit
that touched the file (d9fad7b3, pinned) and sha256-checked, and written as
LICENSE.md at the source root -- a name setuptools' default license_files
glob (LICEN[CS]E*) includes in the wheel's dist-info, and the name
package.yml's license_files gives for info/licenses. Idempotent: a copy
already on disk with the pinned hash is left alone; one with any other
content is an error, not overwritten.
"""
import hashlib
import pathlib
import sys
import time
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import require  # noqa: E402

URL = ("https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/"
       "d9fad7b3450bf4bd29316315032d57157e23a515/LICENSE.md")
SHA256 = "c5ba70a2194af2aefe85dfe3da68608dcb3abd21a3aa53b55aa297c2f0b60eb3"
DST = pathlib.Path("LICENSE.md")

require(pathlib.Path("setup.py").is_file() and pathlib.Path("simple_knn.cu").is_file(),
        "simple_knn: cwd does not look like the camenduru/simple-knn checkout")
headers = pathlib.Path("simple_knn.cu").read_text(encoding="utf-8", errors="replace")
require("LICENSE.md" in headers and "GRAPHDECO" in headers,
        "simple_knn: the source header no longer points at LICENSE.md / GRAPHDECO -- "
        "re-read the licence situation before shipping this text")

if DST.is_file():
    got = hashlib.sha256(DST.read_bytes()).hexdigest()
    require(got == SHA256,
            f"simple_knn: {DST} exists with sha256 {got}, not the pinned {SHA256} -- "
            f"upstream grew a licence file of its own; re-read it before choosing")
    print(f"simple_knn patch: {DST} already present (sha256 verified)")
    sys.exit(0)

last = None
for attempt in range(5):
    try:
        with urllib.request.urlopen(URL, timeout=60) as r:
            data = r.read()
        break
    except Exception as e:  # noqa: BLE001 - retried, then reported
        last = e
        time.sleep(2 ** attempt)
else:
    raise SystemExit(f"PATCH FAILED: could not fetch {URL}: {last}")
got = hashlib.sha256(data).hexdigest()
require(got == SHA256,
        f"{URL}: sha256 {got} does not match the pinned {SHA256}; the text served at "
        f"a pinned commit should never change -- re-verify by hand")
require(b"Gaussian-Splatting License" in data and b"Inria" in data,
        "simple_knn: the fetched file is not the Gaussian-Splatting licence text")
DST.write_bytes(data)
print(f"simple_knn patch: wrote {DST} (Gaussian-Splatting License, sha256 verified, "
      f"{len(data)} bytes)")
