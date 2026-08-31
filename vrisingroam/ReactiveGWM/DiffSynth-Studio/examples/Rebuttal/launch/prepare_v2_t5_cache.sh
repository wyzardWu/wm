#!/usr/bin/env bash

set -euo pipefail
launch_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CACHE_VARIANT=v2
exec bash "${launch_dir}/prepare_variant_t5_cache.sh"
