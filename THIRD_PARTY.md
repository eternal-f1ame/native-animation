# Third-Party Components

`src/diffsynth/` is a vendored subset of DiffSynth-Studio (Apache-2.0), carried in-repo so the project runs from a single clone.

Provenance: the subset was carved from the project's former DiffSynth working fork — frozen at `<workspace>/third_party/DiffSynth-fork` — and migrated into this repository in August 2026. Upstream: https://github.com/modelscope/DiffSynth-Studio.

When the method needs deeper runtime access, copy the required modules from the frozen fork (or current upstream) into `src/diffsynth/` and note the addition here. Keep the root `LICENSE` file with the vendored code when publishing.
