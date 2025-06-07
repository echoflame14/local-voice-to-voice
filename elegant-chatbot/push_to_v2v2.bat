@echo off
echo Pushing elegant chatbot changes to v2v2 repository...
echo.

REM First, ensure we have the v2v2 remote
git remote | findstr /C:"v2v2" >nul
if errorlevel 1 (
    echo Adding v2v2 remote...
    git remote add v2v2 https://github.com/echoflame14/v2v2.git
)

echo.
echo Current branch: CursorChanges
echo Remote: https://github.com/echoflame14/v2v2
echo.

echo You'll need to authenticate with GitHub.
echo If using HTTPS, enter your GitHub username and personal access token.
echo.

git push v2v2 CursorChanges

echo.
echo Done! Check https://github.com/echoflame14/v2v2/tree/CursorChanges
pause