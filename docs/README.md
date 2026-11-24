# Documentation

These documents were written while the project ran (2026-10-24 to 11-21) for invited collaborators of the then-private repository. Where they mention branch `game-v0.113.0` or `main`, read: the top-level code of this repository is that branch (game v0.113.1), and `game-v0.111.0/` holds the code of `main` (game v0.111.0). Links to the private working repository and its commits are kept as records; that repository is not public.

## Read this first

**30 minutes**: [Overview](01_project_overview.md) → [Timeline and critical path](02_timeline_and_critical_path.md) → [Evidence gaps](11_evidence_gaps.md) → [Open problems](12_open_problems_and_collaboration.md).

**Going deeper**:

| # | Document | What you get |
|---|---|---|
| 01 | [Project overview](01_project_overview.md) | task, status, approach, what this project is and is not |
| 02 | [Timeline and critical path](02_timeline_and_critical_path.md) | eight phases, what actually moved the win rate, with evidence levels |
| 03 | [System and infrastructure](03_system_and_infrastructure.md) | headless engine, training system, machines, throughput, Rust engine, `sts2-sim` |
| 04 | [Observation, actions and network](04_observation_and_model.md) | frame format, information boundary, 19 format versions, parameters, design trade-offs |
| 05 | [Training, rewards, curriculum](05_training_rewards_curriculum.md) | PPO setup, two advantage chains, LR/batch history, every reward tried and removed |
| 06 | [Evaluation methodology](06_evaluation_methodology.md) | seed pools, paired tests, power, guardrails, the mistakes that shaped the protocol |
| 07 | [Experiment ledger](07_experiment_ledger.md) | ~90 experiments: design, sample size, result, verdict |
| 08 | [Incidents and rollbacks](08_incidents_and_rollbacks.md) | every rollback and what it taught |
| 09 | [The plateau](09_plateau_and_churn.md) | coin-flip seeds, policy churn, why the mean stopped rising |
| 10 | [Search, ensembles, distillation](10_search_ensembles_distillation.md) | why search distillation failed twice and pool distillation worked |
| 11 | [**Evidence gaps**](11_evidence_gaps.md) | where experiments were too few, too small or too entangled |
| 12 | [**Open problems and collaboration**](12_open_problems_and_collaboration.md) | sized problems, the owner's questions, how to get started |
| 13 | [Lessons learned](13_lessons_learned.md) | 19 lessons with pointers to evidence |
| 14 | [Research reports and literature](14_research_reports_and_literature.md) | English digest of every report, review and survey note |
| 15 | [Decision log index](15_decision_log_index.md) | one English line per decision (291) |
| 16 | [How the project was run](16_how_the_project_was_run.md) | owner + AI agents, the decision log, working rules |
| 17 | [Code map](17_code_map.md) | what is where, how to build and run, loading weights |
| 18 | [**Update of 2026-11-11**](18_update_2026-11-11.md) | what changed since the first snapshot, status of each character |
| 19 | [The Regent line](19_regent.md) | 11-06 → 11-09: setup, schedule changes, Muon trial, ensembles and EMA, student e8, search |
| 20 | [The Silent line](20_silent.md) | from 11-09: Muon from scratch, learning-rate and batch steps, format 23, a reverted trial, current readings |
| 21 | [Combat-outcome prediction and skipping easy fights](21_combat_outcome_prediction_and_skipping.md) | can the network predict fights; resolving near-certain fights in training; fidelity checks |
| 22 | [Engine updates](22_engine_updates.md) | modular observation encoder, formats 20–23, game-update compatibility, new requests |
| 23 | [Game update v0.113.0](23_game_update_v0.113.0.md) (branch `game-v0.113.0`) | what changed in the game, the eight engine fixes, aliases, formats 24–25 for the Fool, the content-prior alignment fix, the Fool line |
| 25 | [The livestream setup of 2026-11-17](25_live_fool_search_and_engine.md) (branch `game-v0.113.0`) | the Fool at v24,181, tree search in boss and elite fights and its measurements, the live adapter, the v0.113.1 engine with state cloning and speed patches |
| 26 | [The Ironclad and the Necrobinder](26_ironclad_and_necrobinder.md) (branch `game-v0.113.0`) | the two v0.113.1 lines behind `weights-v6`, observation format 28 (and 29), recycling, how to run the networks, the checks of the code |
| — | [Watching the network play in the real game (Japanese)](ja/autopilot.md) | the bridge protocol, the game-side mod, the control plane (pause/resume/stop/speed), what is implemented and what is not |
| — | [Running a network locally (Japanese)](ja/local-run.md) | setup on Windows 11 / WSL2, one Ascension-10 run on the CPU, supported game versions, the shared inference API |
| — | [Glossary](glossary.md) | English ↔ Chinese terms used in code and records |
| — | [Notes](notes/) | two earlier English write-ups on out-of-combat training |

