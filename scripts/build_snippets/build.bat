:: win-64 build entry point. Deliberately thin: everything of substance lives in
:: scripts/build_snippets/build_win.py, which generate_recipes.py copies next to
:: the recipe the same way it copies nonet.py.
::
:: The split is not stylistic. rattler-build renders THIS text through minijinja
:: before running it, and minijinja opens a comment on brace-hash, an expression
:: on brace-brace and a statement on brace-percent. When the render fails it
:: reports only "Script failed to execute" -- no script output, no "Running build
:: script" header, because the script never existed to be run. A sibling .py file
:: is copied verbatim and never rendered, so it is free of that hazard; keeping
:: this file short enough to eyeball for those three digraphs is the whole point.
::
:: Batch's own percent-variables are fine here -- it is brace-percent that opens
:: a minijinja statement, not percent alone.
@echo on
setlocal

:: package.yml `build_env` is substituted here by generate_recipes.py, the same
:: way build.sh's # CUW_BUILD_ENV_HOOK works. Without it a package that declares
:: build_env would have it silently applied on Linux and silently dropped here.
:: CUW_BUILD_ENV_HOOK

if not defined PYTHON (
  echo ::error::PYTHON is not set -- rattler-build normally provides it 1>&2
  exit /b 1
)

"%PYTHON%" "%RECIPE_DIR%\build_win.py"
if errorlevel 1 (
  echo ::error::win-64 build script failed 1>&2
  exit /b 1
)

endlocal
exit /b 0
