# pyproject.toml setuptools needs a src layout marker or explicit package list.
# We add it separately to avoid re-writing the whole pyproject.
# But pytest needs to find 'src' package — easiest: let pytest auto-detect via rootdir + add conftest at root.
