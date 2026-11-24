@echo off
rem WSL の中の setup.sh を呼ぶ（Windows のエクスプローラーや PowerShell から実行できる）。
rem 使い方: setup.cmd [--characters DEFECT,IRONCLAD] [--force]
chcp 65001 >nul
wsl.exe --cd "%~dp0." -- bash -lc "./setup.sh %*"
