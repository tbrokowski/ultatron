#!/usr/bin/env bash
# Prevent core_nid* crash dumps from filling the working directory.
ulimit -c 0 2>/dev/null || true
