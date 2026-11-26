# Notice

## What the MIT license covers

The [MIT license](LICENSE) covers the project's own work in this repository: code, documentation, records, training logs, evaluation results and network weights. It does not cover the third-party material listed below.

## Third-party material

- **Slay the Spire 2** is a game by Mega Crit. This project is not affiliated with or endorsed by Mega Crit. The repository contains no game binaries, game assets, art, audio or decompiled game source. Running the engine requires your own copy of the game at the matching version.
  - The content catalogs (`engine/catalog/`, `game-v0.111.0/engine/catalog/`, `game-v0.114.0/engine/catalog/`) and the vocabularies list entity IDs and numerical attributes derived from the game's data, which the networks need as input. They are not covered by the MIT license.
  - In the working version of the v0.113.1 engine, `engine/Sts2Headless/PerfPatches.cs` restated two methods of the game's code (`CombatState.IterateHookListeners`, `RunState.IterateHookListeners`) so that their lists could be pooled for speed. That restated code is **not** in this release; the class is a stub, the game's own methods run, and observation encoding is slower. The full-run determinism hashes of all six characters are the same with and without it (checked 2026-11-21).
- **spire-mentor** (MIT, © Thomas Friedel): the Rust rules core under `rust-engine/vendor/`, with its license in `rust-engine/vendor/LICENSE` and provenance in `rust-engine/vendor/UPSTREAM.md`.
- **sts2-sim** (MIT, https://github.com/iRyougi/sts2-sim): not included. `engine/Sts2SimServe/patches/` and `game-v0.111.0/engine/Sts2SimServe/patches/` hold our patches against its commit `f65056f`.
- **CombatSolver** (a game mod, https://github.com/Torch1230/CombatSolver): used for practice-state labels and the ceiling test; not included.
- **Spire Codex** (https://spire-codex.com): community statistics of uploaded runs, retrieved through its public API, in `engine/card_stats/` (with copies under `game-v0.111.0/` and `game-v0.114.0/`) and `paper/data/codex/` (retrieval dates in the files and in `paper/data/codex/fetched_utc.txt`). They belong to their source and are not covered by the MIT license.

## Not included

- The livestream tooling: the bridge mod that plays the released game, the live policy driver, overlay, avatar and switch-over scripts. `python/sts2rl/tree_live.py` and `python/sts2rl/headless_search.py` are included because the documents describe them, but they do not run without the driver and the mod.
- Scripts that published dashboards and replays to the owner's web server, and the tooling that backed up training data to the owner's cloud storage.
- Raw per-run logs of most tests, replays, full training checkpoints with optimizer state, and anything containing game files.

## Redactions

Machine hostnames, IP addresses, ports, SSH key file names, a dashboard domain, the owner's personal e-mail address and one document about the owner's network setup were removed or replaced with placeholders (`rented-host.example`, `B_LAN_IP`, `C_LAN_IP`, `PORT_A`, `dashboard.example`, `RELAY_IP`, `PLATFORM_GATEWAY_IP`, `mcp-host.example`, `MCP_HOSTNAME`, `relay-provider`, `D_LAN_IP`, `D_USER`, `PORT_BRIDGE`, `B2A_KEY`, `KEY_AB`, `KEY_BOARD`, `KEY_BACKUP`, `KEY_RENTED`, `KEY_D`, `KEY_FILE`, `OWNER_EMAIL`). Scripts under `tools/remote/` therefore do not run as-is; they are kept as the record of how training was run.

## How the code was written

The code, the experiments and most of the documents were produced by AI coding assistants (mainly Claude Code, and Codex for side tooling) under the direction of the project's owner; see [docs/16](docs/16_how_the_project_was_run.md). Code comments and the original records are in Chinese.
