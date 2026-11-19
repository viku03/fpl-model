#!/bin/sh
# Re-download live FPL data (prices, injuries/news, fixtures, per-match history) into data/
cd "$(dirname "$0")" || exit 1
UA="Mozilla/5.0"
curl -sf -A "$UA" -o data/boot.json https://fantasy.premierleague.com/api/bootstrap-static/ || exit 1
curl -sf -A "$UA" -o data/fix.json https://fantasy.premierleague.com/api/fixtures/ || exit 1
.venv/bin/python -c "import json;print('\n'.join(str(e['id']) for e in json.load(open('data/boot.json'))['elements']))" \
  | xargs -P 16 -I{} curl -s -A "$UA" -o data/es/{}.json https://fantasy.premierleague.com/api/element-summary/{}/
