#!/bin/bash
set -euo pipefail

cd /data/users/atavory/local_gp
exec python3 /data/users/atavory/busy_wait/busy_wait.py run \
  /data/users/atavory/local_gp/bandha_hpo_strengthening_busy_wait_20261008 \
  --agent-cmd "python3 /data/users/atavory/local_gp/bandha_hpo_strengthening_busy_wait_agent_20261008.py {prompt_file}" \
  --mission "Strengthen HPO-aware aggregation evidence. Run a real E089 fixed-compute parallel-HPO baseline beyond E088 ultra, update docs and the paper draft, commit, push, and continue until HPO_STRENGTHENING_BUSY_DONE.md validates." \
  --max-wall-sec 0 \
  --turn-timeout-sec 9000 \
  --max-turns 0 \
  --recent-log-lines 260 \
  --keep-going-on-error
