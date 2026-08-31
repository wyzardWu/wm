#!/usr/bin/env bash

set -euo pipefail
launch_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CACHE_VARIANT=v3
exec bash "${launch_dir}/prepare_variant_t5_cache.sh"
