@echo off
setlocal
set "currentDir=%~dp0"
for /r "%currentDir%" %%d in (log __pycache__) do (
    if exist "%%d" (
        rd /s /q "%%d"
    )
)
endlocal
