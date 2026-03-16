"""table_2.py
============
Replication of Table 2 (Top Ten Investors by Asset Class) from
Koijen & Yogo (2020), restricted to the paper's sample end-year (2020).

Data is read from API parquets produced by the pull_* scripts — no
proprietary .dta files required.

Run:
    python src/table_2.py
"""

from table_2_latest import main

if __name__ == "__main__":
    main(year=2020)
