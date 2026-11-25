# AlphaDefect

Reinforcement-learning agents for **Slay the Spire 2** at Ascension 10. For each of the six characters, one network makes every decision of a run, in combat and outside it. It is trained with PPO against the game's own rules, run headlessly, and can guide a decision-time search in the hardest fights.

This repository holds what the paper *AlphaDefect: Expert-Level AI for Slay the Spire 2* (Chen, Yu, Komiyama and Huang, 2026; [arXiv:2611.31076](https://arxiv.org/abs/2611.31076)) reports on: the code, all trained networks, the seed lists, per-seed results, training logs, and the project's full working record.

## Results

Win rates (%) at Ascension 10 in the headless engine, greedy play and with search, with the fresh-seed retests (paper, Tables 3 and 4). Search: rollout search over sampled worlds in the two Act-3 boss fights on v0.111.0 (called PIMC in the code and in file names); information-set tree search in all boss and elite fights on v0.113.1.

| Character | Game | Network | Greedy | Search | Retest on fresh seeds (greedy / search) |
|---|---|---|---|---|---|
| Defect | v0.111.0 | distilled (student e) | 76.7 | 80.3 | 77.8 / – (its seeds had not been used for choices; greedy only) |
| Regent | v0.111.0 | distilled (student e8) | 74.1 | 77.2 | 74.9 / 80.5 |
| Silent | v0.111.0 | EMA, update 26,428 | 71.3 | 74.0 | 71.2 / 73.6 |
| Fool | v0.113.1 | EMA, update 24,181 | 62.2 | 68.0 | 62.5 / 66.7 |
| Ironclad | v0.113.1 | EMA, update 24,782 | 67.9 | 73.1 | 66.5 / – |
| Necrobinder | v0.113.1 | raw, update 25,875 | 63.1 | 67.6 | tested only on fresh seeds (RV1) |

In 866 runs of the released game, played through a mod on a public livestream (to 2026-11-22 08:38 CET), the networks won 645 against 642.5 expected from these headless results.

## Repository layout

| Path | Content |
|---|---|
| `engine/`, `python/`, `tools/` | the system for game **v0.113.1** (the Fool, the Ironclad, the Necrobinder): headless C# host with in-place state cloning, observation encoder (formats up to 29), training system, information-set tree search (`tools/future/`), evaluation and analysis tools |
| `game-v0.111.0/` | the system for game **v0.111.0** (the Defect, the Regent, the Silent), including the rollout search on sts2-sim (called PIMC in the code) ([README](game-v0.111.0/README.md)) |
| `rust-engine/` | our Rust reimplementation of the combat rules, used briefly and then withdrawn (paper, Section 4) |
| `weights/` | model card, manifests and checksums; the network files are release assets ([README](weights/README.md)) |
| `game-mod/` | the game-side bridge mod for watching the network play in the real game (loads through the game's own mod system, streams the public run state; observation/action layer still to come) ([README](game-mod/README.md)) |
| `local/` | the local runner: `python -m local` detects the Steam install and the game version, fetches and verifies the weights, prepares the game files, builds and selfchecks the engine, plays one run (greedy or with search) and gives the shared inference API ([Japanese guide](docs/ja/local-run.md)) |
| `seeds/` | the test seed lists of the paper's tables |
| `results/` | per-seed results behind Tables 3–7, Sections 3.2 and 3.5 and Appendices B (the distilled students against their starting checkpoints) and E (the Silent's network on v0.113.1), with scripts that recount them |
| `paper/` | scripts that regenerate the paper's figures and appendix tables, with their inputs ([README](paper/README.md)) |
| `data/` | learner logs, routine evaluations, per-seed capability tests and exams of the Defect, the Regent and the Silent ([README](data/README.md)) |
| `docs/` | English documentation written during the project ([index](docs/README.md)) |
| `zh/` | the original Chinese records: decision logs of every character, plans, reports and literature notes ([index](zh/README.md)) |

## Quick start: play a released network

Requirements: your own copy of Slay the Spire 2 on the matching public-beta branch (v0.113.1 or v0.111.0; the local runner below also accepts the v0.111.1 patch), the .NET 9 SDK, Python 3.12 with NumPy and PyTorch (a CPU is enough for greedy play). The repository contains no game files.

The commands below are the paper's evaluation path. If you only want to watch one network play a run on your own machine, the local runner does the same steps for you:

```bash
./setup.sh                                            # game files, .NET 9, Python/PyTorch, weights (SHA-256 checked), engine, selfcheck
./play.sh --character DEFECT --seed ABC123 --asc 10    # one Ascension-10 run on the CPU, with its result and elapsed time
```

[`docs/ja/local-run.md`](docs/ja/local-run.md) is the step-by-step guide (Japanese), including the supported game versions, the characters each one can run, the inference API for downstream tools and the troubleshooting table. Windows users can call `setup.cmd` / `play.cmd` instead.

On Windows, `launch.cmd` (or the desktop shortcuts made by `./play.sh launch --install-shortcuts`) starts either a normal game without mods or the AI watch mode with one click ([Japanese guide](docs/ja/launcher.md)).

The game-side bridge (watching in the real game window) is described in [`docs/ja/autopilot.md`](docs/ja/autopilot.md) (Japanese) and [`game-mod/README.md`](game-mod/README.md): `./play.sh mod install`, then `./play.sh autopilot --character DEFECT`. `./play.sh autopilot --simulate --character DEFECT --seed ABC123` runs the same protocol against real headless frames without the game (a complete Ascension-10 run in about 13 s).

```bash
# 1. game files (example for a Steam install seen from WSL; adjust the paths)
V=v0.113.1
G="/mnt/c/Program Files (x86)/Steam/steamapps/common/Slay the Spire 2"
mkdir -p ~/.local/share/sls2/game/$V
rsync -a "$G/data_sts2_windows_x86_64/" ~/.local/share/sls2/game/$V/
python3 -I tools/extract_pck.py "$G/SlayTheSpire2.pck" ~/.local/share/sls2/game/$V/res localization/eng/ localization/zhs/ localization/completion.json

# 2. engine
(cd engine/Sts2Headless && dotnet build -c Release -p:Sts2GameVersion=$V)
dotnet engine/Sts2Headless/bin/Release/net9.0/Sts2Headless.dll selfcheck

# 3. Python
python3.12 -m venv .venv && .venv/bin/pip install numpy torch

# 4. a network (verify the checksum)
gh release download weights-v6 -R hmty-CHEN/AlphaDefect-Public -D weights/files_v6
(cd weights/files_v6 && sha256sum -c ../SHA256SUMS_v6.txt)

# 5. greedy play on the Ironclad's test seeds of Table 3
.venv/bin/python -I tools/ability_test.py --character IRONCLAD --ckpt weights/files_v6/ironclad_ema_v24782.pt \
    --asc 10-10 --runs 1000 --seeds-file seeds/R1.json --envs 8 --device cpu --out out/ironclad_R1 \
    --req-extra '{"obs_format": 28, "loop_cut": true, "stall_turns": 20}'
```

For v0.111.0, do the same inside `game-v0.111.0/` with `V=v0.111.0`. [docs/17](docs/17_code_map.md) maps the code; [docs/25](docs/25_live_fool_search_and_engine.md) and [docs/26](docs/26_ironclad_and_necrobinder.md) describe the v0.113.1 engine, the tree search and the newest networks.

## Reproducing the tables

| Paper row | Code | Network (release, file) | Seeds | Per-seed results |
|---|---|---|---|---|
| Table 3, Defect | `game-v0.111.0/`: `tools/sim_pimc_eval.py` (greedy trunk and rollout search) | `weights-v1`, `student_e_v165018.pt` | W1, entries 6,901–8,400 (`seeds/W1_6901-8400.txt`) | `results/table3/` |
| Table 3, Regent | same | `weights-v2`, `regent_student_e8.pt` | W1, first 1,000 | `results/table3/` |
| Table 3, Silent | same | `weights-v4`, `silent_ema_v26428.pt` | W1, first 1,000 | `results/table3/` |
| Table 3, Fool | top level: `tools/future/run_search.py` (`--sims 0` greedy; `--sims 32 --rooms Boss,Elite`) | `weights-v5`, `fool_ema_v24181.pt` | P2-future, first 3,000 | `results/table3/` |
| Table 3, Ironclad | top level: `tools/ability_test.py` (greedy), `tools/future/run_search.py` (search) | `weights-v6`, `ironclad_ema_v24782.pt` | R1 | `results/table3/` |
| Table 3, Necrobinder | top level: `tools/future/run_search.py` (`--sims 0` greedy; `--sims 32 --rooms Boss,Elite`) | `weights-v6`, `necrobinder_raw_v25875.pt` | RV1, first 1,000 | `results/table3/` |
| Table 4 (retests) | as in Table 3 | as in Table 3 | RV1, first 1,000 | `results/table4/` |
| Section 3.2 (the Defect's network on v0.113.1) | top level: `tools/ability_test.py` (W1); `tools/future/run_search.py` (`--sims 0` greedy; `--sims 32 --rooms Boss,Elite`) on RV1 | `weights-v1`, `student_e_v165018.pt` | W1, first 1,000; RV1, first 1,000 | `results/defect_v0.113.1_W1/`, `results/defect_v0.113.1_RV1/` |
| Appendix E (the Silent's network on v0.113.1) | top level: `tools/future/run_search.py` (`--sims 0 --character SILENT --req-extra '{"obs_format": 23}'`) | `weights-v4`, `silent_ema_v26428.pt` | RV1, first 3,000 | `results/silent_v0.113.1_RV1/` |
| Table 5 (ablations on the Ironclad) | training continued from the Ironclad's update 18,354 with the switches in `results/ablation/switches.diff`; evaluation: top level `tools/ability_test.py` (truncation on, `obs_format` 28) | not released (the branches' checkpoints) | AB1, first 1,000 | `results/ablation/` (`python3 results/ablation/ablation_table.py`) |
| Table 6 (released game) | the livestream mod (not released) | as in Table 3 | the game's own seeds | `results/live/` (`python3 results/live/table6.py`) |
| Table 7 (where greedy runs are lost) | greedy runs of Tables 3 and 4 on RV1 (`results/losses/death_rooms.py` counts them) | as in Table 3 | RV1, first 1,000 | `results/losses/` |
| Section 3.5 (solver positions) | `results/solver_baseline/scripts/` | `weights-v1`, `student_e_v165018.pt` | 241 positions | `results/solver_baseline/` |
| Appendix B (students against their starting checkpoints) | `game-v0.111.0/`: `tools/ability_test.py`, stall rules off | `weights-v1` / `weights-v2` students; starting checkpoints in `weights-v7` | RV1 first 1,000; W1 as in Table 3 | `results/distillation/` |

Greedy evaluations are deterministic given the seed, the network and the batch size, and repeat game by game on the same hardware. Across devices, rare near-ties between candidates can flip a decision (paper, Appendix D). The paper's statistics (Wilson intervals, paired Wald intervals, exact McNemar tests) are computed by `paper/scripts/paper_stats.py`; [results/README](results/README.md) gives the commands, the expected totals and the checksums.

## How this project was made

One person, Qiyue Chen, set the goals, chose among approaches and approved every change to the training runs. AI coding assistants (mainly Claude Code) wrote nearly all of the code, ran the experiments and wrote most of the records, under that direction. [docs/16](docs/16_how_the_project_was_run.md) explains how this worked and how to read the records; the decision logs in `zh/` quote every instruction verbatim. Code comments and the original records are in Chinese; [docs/glossary.md](docs/glossary.md) maps the terms.

## License

The project's own code, documents, data and network weights are released under the [MIT license](LICENSE). Game-derived content tables, the vendored spire-mentor rules core and the Spire Codex statistics are third-party material, described with what is not included and what was redacted in [NOTICE.md](NOTICE.md). Slay the Spire 2 is a game by Mega Crit; this project is not affiliated with or endorsed by Mega Crit.

## Citation

The paper is [arXiv:2611.31076](https://arxiv.org/abs/2611.31076). See also [CITATION.cff](CITATION.cff).

```bibtex
@misc{chen2026alphadefect,
  title         = {AlphaDefect: Expert-Level AI for Slay the Spire 2},
  author        = {Chen, Qiyue and Yu, Taian and Komiyama, Junpei and Huang, Haobo},
  year          = {2026},
  eprint        = {2611.31076},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  doi           = {10.48550/arXiv.2611.31076},
  url           = {https://arxiv.org/abs/2611.31076}
}
```
