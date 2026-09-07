# meta/

One JSON fragment per published `.conda`, written by `tools/fragment.py`
after the artifact is uploaded. `tools/make_repodata.py` assembles the
channel's repodata from these, so regenerating the channel never has to
re-download an artifact.

Published artifacts are immutable: a fragment is written once, and
`fragment.py` hard-errors if one already exists with a different sha256.
Metadata that turns out wrong is fixed in `patches/<subdir>/patches.json`,
which overlays the served repodata without touching the fragment.
