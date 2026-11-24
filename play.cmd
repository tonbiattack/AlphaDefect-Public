@echo off
rem WSL の中の play.sh を呼ぶ（Windows のエクスプローラーや PowerShell から実行できる）。
rem 使い方: play.cmd --character DEFECT --seed ABC123 --asc 10
rem         play.cmd api --character DEFECT --steps 5
chcp 65001 >nul
wsl.exe --cd "%~dp0." -- bash -lc "./play.sh %*"
