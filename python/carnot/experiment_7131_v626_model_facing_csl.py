"""Run delayed fixed-schema memory inside a bounded Qwen decision loop.

Spec refs: REQ-SELF-7131 and SCENARIO-SELF-7131-*.

The model never sees the current exact outcome. Each arm first seals its
prompt. The exact checker then scores the generated action. Only a later event
can read a note or signed record admitted after that outcome closed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import gc
import hashlib
import json
import os
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

from carnot import gpu_lease_phase_journal as lease_api
from carnot.experiment_6966_gguf_gguf_gg_openai import impossible
