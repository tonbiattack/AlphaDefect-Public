# `local/` — running the released networks on your own machine

The command-line runner behind `./setup.sh` and `./play.sh`. It detects the Steam install and
the game version, fetches the weights of the six characters from the GitHub releases and checks
their SHA-256, prepares the game assemblies and localization, builds and selfchecks the headless
engine, plays a run (greedy, or with search on v0.113.1) and exposes the shared inference API for
downstream tools.

```bash
python -m local status                       # what is ready, what is missing
python -m local setup                        # prepare everything (idempotent)
python -m local play --character DEFECT --seed ABC123 --asc 10
python -m local api --character DEFECT --steps 5
python -m local versions
python -m unittest discover -s local/tests -t .
```

The step-by-step guide, and the supported game versions, are in
[`docs/ja/local-run.md`](../docs/ja/local-run.md) (Japanese). Game files and weights are never
committed; they live under `~/.local/share/sls2` (`SLS2_HOME` to change it).

| Module | Role |
|---|---|
| `steam.py` | find the Steam libraries and the game, read its version from `release_info.json` |
| `versions.py` | which code tree and characters each game version can run; rejects the others |
| `gamefiles.py` | copy the game assemblies and extract the localization into the cache |
| `build.py` | .NET 9 SDK detection/installation, engine build, `selfcheck` |
| `weights.py` | the six networks: where they come from, SHA-256 verification, resumable download |
| `recipes.py` | the per-character `reset` options the paper's evaluation used (observation format) |
| `runner.py` | build the command, run it in the version's own tree, parse and summarize the result |
| `infer.py` | the shared inference API used by the downstream features |
| `setup.py`, `cli.py` | the orchestration behind `python -m local` |
