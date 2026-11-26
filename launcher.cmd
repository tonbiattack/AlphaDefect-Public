@echo off
rem AlphaDefect GUI launcher. ASCII only: cmd.exe cannot read UTF-8 batch text.
setlocal
pushd "%~dp0" 2>nul
where python >nul 2>nul && (python "%~dp0launcher.py" & goto :done)
where py >nul 2>nul && (py -3 "%~dp0launcher.py" & goto :done)
echo Python 3 was not found. Install it, or use launcher.ps1 (legacy).
pause
:done
popd 2>nul
endlocal
