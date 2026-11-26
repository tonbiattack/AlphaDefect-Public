# `AlphaDefectAutopilot` — game-side bridge mod

> **Position (issue #71):** real-game AI play is standardised on **Spire Mentor**;
> this mod is kept as the **legacy / comparison / diagnosis** path and also hosts the
> Mentor client (`mentor.txt`). New real-game features should not be implemented twice —
> add them on the Mentor side. Policy and launcher details: `docs/ja/policy.md`.

A Slay the Spire 2 mod (game **v0.111.x** line, .NET 9) that connects the running game to the
AlphaDefect inference server in WSL. It is the Windows half of the watch-the-AI-play feature
(issue #5); the WSL half is `local/bridge/`.

## What this version does (0.2.0)

- loads through the game's own mod system: `[ModInitializer]` in `MegaCrit.Sts2.Core.Modding`,
  no external loader, no Steam Workshop;
- Harmony-patches `NRun._Process` so it reads the live run state on the game's main thread;
- connects to `127.0.0.1:18765` (override with `ALPHADEFECT_AUTOPILOT_HOST` / `_PORT`) and sends
  `hello`, a `probe` twice a second (act, floor, ascension, in-combat, character, HP, gold, deck
  size) and, in combat on the player's turn, a **real observation frame**;
- builds that frame with the headless engine's own encoder (`Engine/Obs/**`, `Observation.cs`,
  `Engine/Compat/**`, `Decision.cs`, `Vocab.cs`) compiled into this assembly, with
  `HeadlessShim.cs` supplying the few `RunSession`/`UiFlows` members it reads. The vocabulary the
  encoder needs is shipped next to the DLL as `vocab.txt`;
- builds the same combat candidates the headless engine builds (hand × target, potions, end turn)
  and, when the server answers with an index, executes it (`CardModel.TryManualPlay`,
  `PotionModel.EnqueueManualUse`, `EndPlayerTurnAction`);
- obeys `control` messages: `paused`, `running`, `stopped`; logs everything to the game log;
- can start a run itself on request from the server (`{"t":"start"}`), through the game's own
  `NGame.StartNewSingleplayerRun`, then enters the first combat directly with
  `RunManager.EnterRoomDebug` (the server's `--autostart "DEFECT|0|SEED|weak"` does this);
- runs its tick on the Godot `SceneTree.ProcessFrame` signal, so it also works while the game sits
  in the main menu (the `NRun._Process` patch remains as a fallback).

## What it does not do yet

- **out-of-combat decisions**: rewards, map, shop, events and card selection are not driven; the
  game waits for the player there (the mod only reports state);
- **overlay**: no on-screen text of the chosen action or its candidates;
- the in-game execution path has not been observed in a real fight yet (the session that built it
  ended with the game at the main menu).

## Observation format and character

The encoder emits the character's default format (Defect 19, Regent 20, Silent 23 on v0.111.x). The
server sends its checkpoint's format in `welcome`; the mod applies it with
`Observation.FormatOverride`, and the server refuses a frame whose format does not match its
weights, so the network is never fed another character's numbers. On the v0.111.x line only Defect,
Regent and Silent have weights; Ironclad/Fool/Necrobinder belong to v0.113.x.

## Build and install

```bash
# from the repository root; builds against the game assemblies that `./setup.sh` cached
python -m local mod install        # builds, copies dll+json into <game>/mods/AlphaDefectAutopilot/
python -m local mod status         # installed? built for the running game version?
python -m local mod remove
```

Or by hand:

```bash
cd game-mod/AlphaDefectAutopilot
dotnet build -c Release -p:Sts2GameDir="$HOME/.local/share/sls2/game/v0.111.1"
cp bin/Release/AlphaDefectAutopilot.{dll,json} \
   "/mnt/c/Program Files (x86)/Steam/steamapps/common/Slay the Spire 2/mods/AlphaDefectAutopilot/"
```

The game reads `<exe dir>/mods/**/<id>.json` and loads `<id>.dll` when `has_dll` is true. The
first time, the game shows its mod-loading consent screen; enable the mod in **Settings → Modding**
if it is listed as disabled. `--nomods` on the game's command line skips mods entirely.

## Debugging

- the game log (`%APPDATA%\Godot\app_userdata\Slay the Spire 2\logs\godot.log`) gets
  `[AlphaDefect] …` lines from every step;
- the server side prints `connected`, `hello`, `probe` and `control` events and writes the same
  to `~/.local/share/sls2/logs/autopilot-*.jsonl`;
- if nothing arrives, check that WSL's `127.0.0.1` is reachable from Windows
  (`curl http://127.0.0.1:18765` is not enough — it is a raw TCP socket; check the server is
  listening and WSL's localhost forwarding is on).

## Protocol

See `local/bridge/protocol.py` for the authoritative definitions: one JSON object per line over
TCP, version 1, `hello` / `probe` / `state` / `done` / `log` / `bye` from the mod and
`welcome` / `action` / `control` / `error` from the server.
