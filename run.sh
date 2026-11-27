#!/bin/sh
# Weekly run: fresh data -> retrain + predict -> pull your squad -> plan.
# Usage: ./run.sh            pulls your squad using entry_id (config.json or FPL_ENTRY_ID)
#        ./run.sh --manual   keeps squad.json as it is (e.g. transfers made before a deadline)
cd "$(dirname "$0")" || exit 1
./refresh_data.sh || { echo "Could not download FPL data"; exit 1; }
.venv/bin/python fetch_cups.py || echo "Could not update European/cup fixtures, using the last saved ones"
.venv/bin/python odds.py || echo "Could not update bookmaker odds, using the last saved ones"
HORIZON=$(.venv/bin/python settings.py horizon)
ENTRY=$(.venv/bin/python settings.py entry_id)
.venv/bin/python model.py "$HORIZON" || exit 1
if [ "$1" != "--manual" ] && [ -n "$ENTRY" ]; then .venv/bin/python state.py "$ENTRY" || exit 1; fi
.venv/bin/python rivals.py || echo "Could not update rival squads, using the last saved ones"
.venv/bin/python planner.py
