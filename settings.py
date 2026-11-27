"""Load settings: your own config.json if it exists (it's git-ignored, since it holds your team ID),
otherwise config.example.json. FPL_ENTRY_ID / FPL_LEAGUE_ID environment variables override both."""
import json
import os


def load_config():
    path = "config.json" if os.path.exists("config.json") else "config.example.json"
    cfg = json.load(open(path))
    for key, env in (("entry_id", "FPL_ENTRY_ID"), ("league_id", "FPL_LEAGUE_ID")):
        if os.environ.get(env):
            cfg[key] = int(os.environ[env])
    return cfg


if __name__ == "__main__":  # used by run.sh: python settings.py <key>
    import sys
    print(load_config().get(sys.argv[1]) or "")
