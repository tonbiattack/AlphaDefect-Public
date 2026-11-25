@echo off
rem AlphaDefect ランチャー（GUI）。ダブルクリックで起動します。
powershell.exe -NoProfile -ExecutionPolicy Bypass -STA -File "%~dp0launcher.ps1"
