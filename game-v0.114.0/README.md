# Code for game v0.114.0 (public beta of 2026-11-25)

Slay the Spire 2 v0.114.0 is a public-beta update released on 2026-11-25. Most of it reworks the Fool: a new card (Optimism), Twin Shot in place of Rapid Shot, and about forty reworked or rebalanced Fool cards and tarot cards. It also changes how damage is resolved and previewed. This folder is the top-level system of this repository (game v0.113.1) adapted to v0.114.0, for training and evaluation on the new version.

**No result of the paper comes from this folder, and no network has been trained on v0.114.0 yet.** For the paper's tables use the top level (v0.113.1) or `game-v0.111.0/`.

## What this folder is

The top-level `engine/`, `python/` and `tools/` of this repository, as released, plus the adaptation commits of the working repository's branch `game-v0.114.0` (2026-11-25; `02ff022c`, `b7c546d6`, `6a4ef8f0`, `db72c58a`, `a4c5481c`). Two kinds of files were left out:

- files that are not part of this release anyway (the livestream mod's `Entry.cs`, `python/play_live.py`);
- `engine/Sts2SimServe/`, because [sts2-sim](https://github.com/iRyougi/sts2-sim) has not been updated to v0.114.0.

The redactions of [NOTICE.md](../NOTICE.md) and the `PerfPatches.cs` stub are kept unchanged.

The records are in Chinese, in [`zh/游戏更新v0.114.0/`](../zh/游戏更新v0.114.0/):

| File | Content |
|---|---|
| `更新内容盘点.md` | every rule change from v0.113.1 to v0.114.0 with its place in the game code, including changes the patch notes do not mention, and a checklist for porting the update to sts2-sim |
| `适配报告.md` | how the engine was adapted, the checks and their numbers, open questions |
| `影响评估.md` | the generated impact report: content-catalog changes field by field, game classes the engine references |
| `附件/逻辑文件差异_v0.113.1至v0.114.0.tsv` | the list of changed game-logic files |

## What changed in the code

- **Game interfaces.** Three members the engine used were removed from the game: `ArrowStatus.Destroyed`, `MelodyCmd.MelodyCardsPlayedThisTurn` and `FreeCardPower`. The engine no longer uses them; the Fool's combat counter for destroyed arrows is now always 0.
- **Vocabulary and catalog.** `CARD.OPTIMISM` and `CARD.TWIN_SHOT` are appended (2,557 → 2,559 entries; old IDs unchanged), and `engine/catalog/v0.114.0/catalog.json` is added. Twin Shot is aliased to Rapid Shot's ID. Its `OnPlay` is identical to Rapid Shot's; only the numbers and the rarity changed, and the game's own save migration renames one to the other. Released networks therefore keep their embedding for it.
- **Fool rules written into the observation.** Several observation features compute what a play gains, such as candidate features 12 and 13 (energy gained, cards entering the hand). They follow the new rules in both the format-26/27 path (`Gains26.cs`) and the format-29 path (`Gains29.cs`, `Knap29.cs`). The rules concerned are Vitriol, Florescence, Trick Shot, Arrow Stab, Quietus, The Fool, Optimism, Cho Chorus with Wah Blast, Ki Verse, Self Deprecate and The Hierophant.
- **Damage preview.** The game's targeted damage preview now already includes block and HP-loss hooks (Hardened Shell, Hard to Kill, Slippery, The Boot). The observation treats single-hit damage as the value before those hooks and applies them itself. `Engine/Obs/RawPreview.cs` therefore keeps the preview at the v0.113.1 basis while frames are encoded. Hard to Kill changed from a damage cap to an HP-loss cap; the observation applies the cap at the v0.113.1 basis wherever it computes single-hit damage.
- **Removed IDs in old data.** Player snapshots and deck specs that still contain `RAPID_SHOT` are renamed through the game's own migration tables (`Snapshot.cs`, `CombatSpec.cs`). Before this fix, such decks failed to reset, and snapshots silently turned the card into `DEPRECATED_CARD`.
- **Loading checkpoints trained on v0.113.x.** The content prior grows from 1,195 to 1,196 columns. `content.legacy_catalogs()` now tries every older catalog in `engine/catalog/`, so all loaders realign the columns by feature name. Before, most loaders only padded zero columns at the end, which misaligned the columns without an error.

Frame layouts and format numbers are unchanged.

## Building and running

Requirements are as in the top-level [README](../README.md#quick-start-play-a-released-network), with game files of v0.114.0 (public-beta branch).

```bash
V=v0.114.0
# game files prepared as in the top-level README, into ~/.local/share/sls2/game/$V
cd game-v0.114.0/engine/Sts2Headless && dotnet build -c Release -p:Sts2GameVersion=$V
dotnet bin/Release/net9.0/Sts2Headless.dll selfcheck
```

Run the Python tools from inside `game-v0.114.0/` so that they pick up this folder's `engine/` and catalog.

## Checks

Reference engine: the top level of this repository built for v0.113.1. Details and the remaining differences are in `适配报告.md`.

| Check | Result |
|---|---|
| Build | 0 errors; the same four nullable-reference warnings as v0.113.1 |
| `selfcheck 3` | compatibility layer 184 entries (61 patches), none missing; smoke test, 20 runs per character including format-29 frames: no exceptions, no stalls, no out-of-vocabulary keys |
| Byte-for-byte frames against the v0.113.1 engine in fights whose rules did not change (same seeds and actions; Slippery, Hardened Shell, Hard to Kill; Ironclad in format 29, Necrobinder in format 28, Silent, Defect) | 3,216 frames in five settings, 0 differ |
| Fool, candidate features 12/13 against what actually happened (`tools/obs113/gains26_probe.py`, 200 fights) | feature 12: 52/52 relevant steps agree; feature 13: 451/458 in format 27 and in format 29 (v0.113.1 engine on v0.113.1: 465/475; v0.114.0 before the observation fixes: 400/476). The remaining 7 are random effects (The Hermit, Wheel of Fortune, The Devil, The Fool), Repeat Shot with X = 0, and Optimism, where the probe counts the net hand change |
| Fool, damage, kill and block predictions against what actually happened (`tools/obs113/pred_probe.py`, 300 fights) | 14,772 of 14,867 predictions agree (99.36%); end-of-turn damage forecast 2,040 of 2,044 |
| Fool, gold and red card glow (`tools/obs113/glow27_probe.py`, 60 fights) | 0 disagreements |
| State cloning, round trip (`clone-test roundtrip`) | 60 fights (1,280 frames) and 24 runs (1,400 frames), 0 differ |
| Loading the Fool's `weights-v5` network | content prior realigned (1,190 of 1,195 columns matched, 6 new columns zero); the same result with and without the stored feature names |

In the working repository, the same changes on its newest line also passed the full state-cloning acceptance and a whole-run smoke test. The numbers are in `适配报告.md`.
