#!/usr/bin/env python3
"""Inspect recordings of play from the host (numpy only, no Triton, no torch).

    tools/recording.py info <dir> [--episodes N]
    tools/recording.py dump <dir> [--agent A] [--episode E] [--rows N]
    tools/recording.py compare <recorded> <replayed> [--agent-a A] [--agent-b B]

The format and the reader live in triton/common/maniple/recording.py (shared with the trainer).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "triton", "common"))

from maniple.recording import main  # noqa: E402

if __name__ == "__main__":
    main()
