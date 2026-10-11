#!/bin/zsh
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
QUEUE="$1"
[[ "$QUEUE" != /* ]] && QUEUE="$ROOT/$QUEUE"
mkdir -p "$ROOT/results/logs"
cd "$ROOT/src"
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  if [[ "$line" == *"--name "* ]]; then
    name=$(echo "$line" | sed -E 's/.*--name ([^ ]+).*/\1/')
  else
    name=$(echo "$line" | awk '{print $1}' | sed 's/.py$//')
  fi
  echo "[$(date +%H:%M:%S)] START $line"
  PYTHONUNBUFFERED=1 ../venv/bin/python -W ignore ${=line} >> "$ROOT/results/logs/${name}.log" 2>&1
  echo "[$(date +%H:%M:%S)] END   $name (exit $?)"
done < "$QUEUE"
