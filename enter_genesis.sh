#!/usr/bin/env bash
# Strict mode: fail fast and surface errors.
set -Eeuo pipefail
trap 'die "Command failed (line $LINENO): ${BASH_COMMAND:-unknown}"' ERR

say() { printf '%s\n' "$*" >&2; }
die() { say "ERROR: $*"; exit 1; }

# Preconditions
command -v xhost  >/dev/null 2>&1 || die "xhost not found in PATH."
command -v docker >/dev/null 2>&1 || die "docker not found in PATH."
docker info >/dev/null 2>&1 || die "Docker daemon not reachable. Is Docker running?"

# X auth
export DISPLAY=":1"  # WARN: Assuming ":1" because that worked in my setup. Use `who` to find it in your setup.
say "Using DISPLAY=${DISPLAY}"

# Run xhost
# Check exit status and the if 'unable to open display' is returned
xhost_output="$(xhost +local:root 2>&1)" || { say "$xhost_output"; die "xhost +local:root failed."; }
if printf '%s' "$xhost_output" | grep -qi 'unable to open display'; then
  say "$xhost_output"
  die "Cannot open X display ${DISPLAY}."
fi
say "X access granted to local:root."

# Start dev-genesis container and enter bash shell
container="dev-genesis"

# Start the container only if it's not already running
if ! docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null | grep -q '^true$'; then
  docker start "$container" >/dev/null 2>&1 || die "Failed to start container '$container'. Does it exist?"
  say "Container '$container' started."
else
  say "Container '$container' already running."
fi

say "Attaching to '$container'..."
exec docker exec -it "$container" /bin/bash
