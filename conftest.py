"""conftest.py — sets the working directory so _read() finds the .dta files."""
import os
from pathlib import Path

# table_1.py resolves DATA relative to __file__ (src/), so no chdir needed.
# This file just ensures pytest discovers tests/ from the project root.
