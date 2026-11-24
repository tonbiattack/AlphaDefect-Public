# `AlphaDefectAutopilot` — game-side bridge mod

A Slay the Spire 2 mod (game **v0.111.x** line, .NET 9) that connects the running game to the
AlphaDefect inference server in WSL. It is the Windows half of the watch-the-AI-play feature
(issue #5); the WSL half is `local/bridge/`.

## What this version does

- loads through the game's own mod system: `[ModInitializer]` in `MegaCrit.Sts2.Core.Modding`,
  no external loader, no Steam Workshop;
- Harmony-patches `NRun._Process` so it reads the live run state on the game's main thread;
- connects to `127.0.0.1:8765` (override with `ALPHADEFECT_AUTOPILOT_HOST` / `_PORT`) and sends
  `hello` and, twice a second, a `probe`: act, floor, ascension, in-combat, character, HP, gold,
  deck size — all of it public information, nothing from the draw pile or the RNG;
- obeys `control` messages: `paused`, `running`, `stopped`; logs everything to the game log.

## What it does not do yet

- **observation frames**: the mod does not yet build the engine's observation frame (the encoder
  in `engine/Sts2Headless/Engine/Obs/`). `local/bridge` decides on frames, so until this is added
  the server receives `probe` and never sends an `action`.
- **action execution**: no decision is executed in the game yet.
- **overlay**: no on-screen text of the chosen action or its candidates.

The next step is to compile the encoder into this assembly. It needs a small shim, not a rewrite:
the encoder reads the run through `RunSession.State`, `RunSession.Steps`, `RunSession.IsVictory`,
`RunSession.RestHealInfo` and `UiFlows.CombatsWon`, so a bridge-local static class with those
members can stand in for the headless session. `engine/Directory.Build.targets` already treats the
`Obs/**` and `Compat/**` sources as shareable with a bridge mod for exactly this reason. After that,
the remaining piece is mapping a chosen candidate index back to the game call that performs it,
one decision type at a time (combat first, then rewards, map, shop, events).

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
  (`curl http://127.0.0.1:8765` is not enough — it is a raw TCP socket; check the server is
  listening and WSL's localhost forwarding is on).

## Protocol

See `local/bridge/protocol.py` for the authoritative definitions: one JSON object per line over
TCP, version 1, `hello` / `probe` / `state` / `done` / `log` / `bye` from the mod and
`welcome` / `action` / `control` / `error` from the server.
