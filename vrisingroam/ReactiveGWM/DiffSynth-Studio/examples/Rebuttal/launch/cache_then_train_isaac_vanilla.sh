#!/usr/bin/env bash

set -euo pipefail

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${LAUNCH_DIR}/prepare_isaac_cache.sh"
"${LAUNCH_DIR}/train_isaac_vanilla.sh" "$@"
