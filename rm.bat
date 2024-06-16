@echo off
setlocal
set "currentDir=%~dp0"
for /r "%currentDir%" %%d in (log) do (
    if exist "%%d" (
        rd /s /q "%%d"
    )
)
endlocal
