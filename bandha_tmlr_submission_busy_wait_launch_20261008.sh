#!/bin/bash
set -euo pipefail

cd /data/users/atavory/local_gp
exec python3 /data/users/atavory/busy_wait/busy_wait.py run \
  /data/users/atavory/local_gp/bandha_tmlr_submission_busy_wait_20261008 \
  --agent-cmd "python3 /data/users/atavory/local_gp/bandha_tmlr_submission_busy_wait_agent_20261008.py {prompt_file}" \
  --mission "Make the HPO-aware aggregation project TMLR-submission-ready. Run E090 fixed-compute stale-default recovery or the next strongest fixed-compute HPO win, produce a TMLR submission packet and narrowed draft, update docs, commit, push, and continue until TMLR_SUBMISSION_BUSY_DONE.md validates." \
  --max-wall-sec 0 \
  --turn-timeout-sec 10800 \
  --max-turns 0 \
  --recent-log-lines 320 \
  --keep-going-on-error
