"""table_1.py
============
Replication of Table 1 (Market Values of Financial Assets) from
Koijen & Yogo (2020), restricted to the paper's sample end-year (2020).

Data is read from API parquets produced by the pull_* scripts — no
proprietary .dta files required.

Run:
    python src/table_1.py
"""

from table_1_latest import main

if __name__ == "__main__":
    main(year=2020)
