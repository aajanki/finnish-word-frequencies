#!/bin/sh
set -eu

chown appuser /app/results

gosu appuser "$@"
