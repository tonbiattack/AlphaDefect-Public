@echo off
rem AlphaDefect GUI launcher (double-click). ASCII only: cmd.exe cannot
rem read UTF-8/CJK text in a batch file reliably, so keep this file plain.
setlocal
rem %~dp0 is a UNC path when run from Explorer; pushd maps a drive so the
rem script can run without the "UNC paths are not supported" side effects.
pushd "%~dp0" 2>nul
powershell.exe -NoProfile -ExecutionPolicy Bypass -STA -File "%~dp0launcher.ps1"
popd 2>nul
endlocal
