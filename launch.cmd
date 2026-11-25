@echo off
rem ============================================================
rem  AlphaDefect ランチャー（Windows からダブルクリックで起動）
rem
rem    launch.cmd            メニューを出す
rem    launch.cmd ai         AI 観戦（推論サーバ＋MOD）
rem    launch.cmd normal     通常プレイ（MOD なし）
rem    launch.cmd status     状態
rem    launch.cmd stop       推論サーバを止める
rem
rem  WSL2 の中の `./play.sh launch ...` を呼びます。
rem ============================================================
chcp 65001 >nul
setlocal

if not "%~1"=="" (
  wsl.exe --cd "%~dp0." -- bash -lc "./play.sh launch %*"
  goto :end
)

:menu
cls
echo ============================================
echo  AlphaDefect ランチャー
echo ============================================
echo   1. AI 観戦（推論サーバ＋MOD）
echo   2. 通常プレイ（MOD なし）
echo   3. 状態を見る
echo   4. 推論サーバを止める
echo   5. Mentor 観戦（キャラ・アセンション・シードを指定）
echo   Q. 終了
echo.
set "CHOICE="
set /p "CHOICE=番号を入力: "
if /i "%CHOICE%"=="1" (
  wsl.exe --cd "%~dp0." -- bash -lc "./play.sh launch ai"
  goto :end
)
if /i "%CHOICE%"=="2" (
  wsl.exe --cd "%~dp0." -- bash -lc "./play.sh launch normal"
  goto :end
)
if /i "%CHOICE%"=="3" (
  wsl.exe --cd "%~dp0." -- bash -lc "./play.sh launch --status"
  echo.
  pause
  goto :menu
)
if /i "%CHOICE%"=="5" (
  set "MCHAR="
  set "MASC="
  set "MSEED="
  set /p "MCHAR=キャラ (DEFECT / REGENT / SILENT / auto / select) [auto]: "
  if "%MCHAR%"=="" set "MCHAR=auto"
  set /p "MASC=アセンション 0-10 [0]: "
  if "%MASC%"=="" set "MASC=0"
  set /p "MSEED=シード（整数、空でランダム）: "
  if "%MSEED%"=="" (
    wsl.exe --cd "%~dp0." -- bash -lc "./play.sh launch mentor --character '%MCHAR%' --ascension %MASC%"
  ) else (
    wsl.exe --cd "%~dp0." -- bash -lc "./play.sh launch mentor --character '%MCHAR%' --ascension %MASC% --seed %MSEED%"
  )
  echo.
  pause
  goto :menu
)
if /i "%CHOICE%"=="4" (
  wsl.exe --cd "%~dp0." -- bash -lc "./play.sh launch --stop"
  echo.
  pause
  goto :menu
)
if /i "%CHOICE%"=="q" goto :end
goto :menu

:end
endlocal
