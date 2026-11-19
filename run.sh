#!/bin/sh
# Weekly run: fresh data -> retrain + predict -> pull your squad -> plan.
# Usage: ./run.sh            pulls your squad using entry_id in config.json
#        ./run.sh --manual   keeps squad.json as it is (e.g. transfers made before a deadline)
cd "$(dirname "$0")" || exit 1
./refresh_data.sh || { echo "Could not download FPL data"; exit 1; }
.venv/bin/python fetch_cups.py || echo "Could not update European/cup fixtures, using the last saved ones"
.venv/bin/python odds.py || echo "Could not update bookmaker odds, using the last saved ones"
HORIZON=$(.venv/bin/python -c "import json;print(json.load(open('config.json'))['horizon'])")
ENTRY=$(.venv/bin/python -c "import json;print(json.load(open('config.json')).get('entry_id') or '')")
.venv/bin/python model.py "$HORIZON" || exit 1
if [ "$1" != "--manual" ] && [ -n "$ENTRY" ]; then .venv/bin/python state.py "$ENTRY" || exit 1; fi
.venv/bin/python planner.py
