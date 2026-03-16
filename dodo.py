"""dodo.py
========
PyDoit pipeline for the Koijen & Yogo (2020) replication project.

This file orchestrates the full end-to-end workflow: pulling raw data from
BIS, OECD, IMF, and World Bank APIs; cleaning and tidying the data; building
the replication tables (Table 1 and Table 2); running the test suite; and
generating summary-statistics exhibits and LaTeX reports.  Each ``task_*``
function corresponds to one doit task; run ``doit list`` to see all available
tasks and ``doit`` (no arguments) to execute the default pipeline.
"""

#######################################
## Configuration and Helpers for PyDoit
#######################################
## Make sure the src folder is in the path
import sys

sys.path.insert(1, "./src/")

import shutil
from os import environ
from pathlib import Path

from settings import config

DOIT_CONFIG = {"backend": "sqlite3", "dep_file": "./.doit-db.sqlite"}


BASE_DIR = config("BASE_DIR")
DATA_DIR = config("DATA_DIR")
MANUAL_DATA_DIR = config("MANUAL_DATA_DIR")
OUTPUT_DIR = config("OUTPUT_DIR")
OS_TYPE = config("OS_TYPE")
USER = config("USER")

## Helpers for handling Jupyter Notebook tasks
environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

# fmt: off
## Helper functions for automatic execution of Jupyter notebooks
def jupyter_execute_notebook(notebook_path):
    return f"jupyter nbconvert --execute --to notebook --ClearMetadataPreprocessor.enabled=True --inplace {notebook_path}"
def jupyter_to_html(notebook_path, output_dir=OUTPUT_DIR):
    return f"jupyter nbconvert --to html --output-dir={output_dir} {notebook_path}"
def jupyter_to_md(notebook_path, output_dir=OUTPUT_DIR):
    """Requires jupytext"""
    return f"jupytext --to markdown --output-dir={output_dir} {notebook_path}"
def jupyter_clear_output(notebook_path):
    """Clear the output of a notebook"""
    return f"jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --inplace {notebook_path}"
# fmt: on


def mv(from_path, to_path):
    """Move a file to a folder"""
    from_path = Path(from_path)
    to_path = Path(to_path)
    to_path.mkdir(parents=True, exist_ok=True)
    if OS_TYPE == "nix":
        command = f"mv {from_path} {to_path}"
    else:
        command = f"move {from_path} {to_path}"
    return command


def copy_file(origin_path, destination_path, mkdir=True):
    """Create a Python action for copying a file."""

    def _copy_file():
        origin = Path(origin_path)
        dest = Path(destination_path)
        if mkdir:
            dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, dest)

    return _copy_file


##################################
## Begin rest of PyDoit tasks here
##################################


def task_config():
    """Create empty directories for data and output if they don't exist"""
    return {
        "actions": ["ipython ./src/settings.py"],
        "targets": [DATA_DIR, OUTPUT_DIR],
        "file_dep": ["./src/settings.py"],
        "clean": [],
    }


def task_pull():
    """Pull data from BIS, OECD, IMF, and World Bank"""
    yield {
        "name": "bis",
        "doc": "Pull BIS domestic (WS_NA_SEC_DSS) and IDS foreign-currency (WS_DEBT_SEC2_PUB) data",
        "actions": [
            "ipython ./src/settings.py",
            "ipython ./src/pull_bis.py",
        ],
        "targets": [
            DATA_DIR / "bis_dds_q.parquet",
            DATA_DIR / "bis_ids_foreign_q.parquet",
        ],
        "file_dep": ["./src/settings.py", "./src/pull_bis.py"],
        "clean": [],
    }
    yield {
        "name": "oecd_t720",
        "doc": "Pull OECD Table 720 balance sheet data",
        "actions": [
            "ipython ./src/settings.py",
            "ipython ./src/pull_oecd.py",
        ],
        "targets": [DATA_DIR / "oecd_t720.parquet"],
        "file_dep": ["./src/settings.py", "./src/pull_oecd.py"],
        "clean": [],
    }
    yield {
        "name": "imf",
        "doc": "Pull IMF PIP/CPIS bilateral and reserve positions",
        "actions": [
            "ipython ./src/settings.py",
            "ipython ./src/pull_imf.py",
        ],
        "targets": [
            DATA_DIR / "pip_bilateral_positions.parquet",
            DATA_DIR / "pip_bilateral_positions_reserve.parquet",
            DATA_DIR / "pip_currency_aggregates.parquet",
            DATA_DIR / "pip_local_foreign_allocated.parquet",
        ],
        "file_dep": ["./src/settings.py", "./src/pull_imf.py"],
        "clean": [],
    }
    yield {
        "name": "wb",
        "doc": "Pull World Bank WDI data",
        "actions": [
            "ipython ./src/settings.py",
            "ipython ./src/pull_WB.py",
        ],
        "targets": [DATA_DIR / "wb_data360_wdi_selected.parquet"],
        "file_dep": ["./src/settings.py", "./src/pull_WB.py"],
        "clean": [],
    }

def task_tidy_data():
    """Consolidate raw API parquets into tidy_amounts and tidy_bilateral parquets"""
    return {
        "actions": ["ipython ./src/tidy_data.py"],
        "targets": [
            DATA_DIR / "tidy_amounts.parquet",
            DATA_DIR / "tidy_bilateral.parquet",
        ],
        "file_dep": [
            "./src/tidy_data.py",
            DATA_DIR / "bis_dds_q.parquet",
            DATA_DIR / "oecd_t720.parquet",
            DATA_DIR / "wb_data360_wdi_selected.parquet",
            DATA_DIR / "pip_bilateral_positions.parquet",
        ],
        "task_dep": ["pull:bis", "pull:oecd_t720", "pull:imf", "pull:wb"],
        "clean": True,
    }


def task_build_tables():
    """Build Table 1 and Table 2 from processed data"""
    yield {
        "name": "table1",
        "doc": "Build Table 1 (Market Values of Financial Assets)",
        "actions": ["ipython ./src/table_1.py"],
        "targets": [OUTPUT_DIR / "table_1.txt"],
        "file_dep": ["./src/table_1.py"],
        "task_dep": ["pull:bis", "pull:oecd_t720", "pull:imf", "pull:wb"],
        "clean": True,
    }
    yield {
        "name": "table2",
        "doc": "Build Table 2 (Top Ten Investors by Asset Class)",
        "actions": ["ipython ./src/table_2.py"],
        "targets": [OUTPUT_DIR / "table_2.txt"],
        "file_dep": ["./src/table_1.py", "./src/table_2.py"],
        "task_dep": ["build_tables:table1"],
        "clean": True,
    }


def task_build_latest_tables():
    """Build Table 1 and Table 2 for the latest available year from API parquets"""
    yield {
        "name": "table1",
        "doc": "Build Table 1 for the latest year (auto-detected from parquets)",
        "actions": ["ipython ./src/table_1_latest.py"],
        "targets": [],   # filename includes year, detected at runtime
        "file_dep": [
            "./src/table_1_latest.py",
            DATA_DIR / "oecd_t720.parquet",
            DATA_DIR / "bis_dds_q.parquet",
            DATA_DIR / "pip_bilateral_positions.parquet",
            DATA_DIR / "wb_data360_wdi_selected.parquet",
        ],
        "task_dep": ["pull:bis", "pull:oecd_t720", "pull:imf", "pull:wb"],
        "clean": True,
        "uptodate": [False],  # always re-run (year detection is dynamic)
    }
    yield {
        "name": "table2",
        "doc": "Build Table 2 for the latest year",
        "actions": ["ipython ./src/table_2_latest.py"],
        "targets": [],
        "file_dep": [
            "./src/table_1_latest.py",
            "./src/table_2_latest.py",
            DATA_DIR / "pip_bilateral_positions.parquet",
        ],
        "task_dep": ["pull:imf"],
        "clean": True,
        "uptodate": [False],
    }


def task_test():
    """Run pytest on all test suites"""
    return {
        "actions": ["python -m pytest src/ -v --tb=short"],
        "file_dep": [
            "./src/test_table_1.py",
            "./src/test_table_2.py",
            "./src/test_misc_tools.py",
            "./src/table_1_latest.py",
            DATA_DIR / "bis_dds_q.parquet",
            DATA_DIR / "oecd_t720.parquet",
            DATA_DIR / "pip_bilateral_positions.parquet",
        ],
        "task_dep": ["pull:bis", "pull:oecd_t720", "pull:imf", "pull:wb"],
        "uptodate": [False],
        "clean": [],
    }


def task_generate_chart():
    """Run generate_chart.py to produce the chart"""
    script_path = "./src/generate_chart.py"
    output_file = OUTPUT_DIR / "obs_value_by_sector.html"  

    return {
        "actions": [f"ipython {script_path}"],
        "file_dep": [script_path],
        "targets": [output_file],
        "clean": True,
    }


def task_summary_stats():
    """Generate summary statistics LaTeX table and PNG chart from tidy data"""
    return {
        "actions": ["ipython ./src/summary_stats.py"],
        "targets": [
            OUTPUT_DIR / "summary_stats_table.tex",
            OUTPUT_DIR / "summary_stats_chart.png",
        ],
        "file_dep": [
            "./src/summary_stats.py",
            DATA_DIR / "tidy_amounts.parquet",
        ],
        "task_dep": ["tidy_data"],
        "clean": True,
    }


def task_compile_summary():
    """Compile the summary statistics LaTeX report to PDF"""
    return {
        "actions": [
            "latexmk -xelatex -halt-on-error -cd ./reports/report_summary.tex",
            "latexmk -xelatex -halt-on-error -c -cd ./reports/report_summary.tex",
        ],
        "targets": ["./reports/report_summary.pdf"],
        "file_dep": [
            "./reports/report_summary.tex",
            OUTPUT_DIR / "summary_stats_table.tex",
            OUTPUT_DIR / "summary_stats_chart.png",
        ],
        "task_dep": ["summary_stats"],
        "clean": True,
    }


notebook_tasks = {
    "01_example_notebook_interactive_ipynb": {
        "path": "./src/01_example_notebook_interactive_ipynb.py",
        "file_dep": [],
        "targets": [],
    },
    "koijen_yogo_2020_tour_ipynb": {
        "path": "./src/koijen_yogo_2020_tour_ipynb.py",
        "file_dep": [
            "./src/table_1.py",
            "./src/table_2.py",
            OUTPUT_DIR / "table_1.txt",
            OUTPUT_DIR / "table_2.txt",
        ],
        "targets": [],
    },
}


# fmt: off
def task_run_notebooks():
    """Preps the notebooks for presentation format.
    Execute notebooks if the script version of it has been changed.
    """
    for notebook in notebook_tasks.keys():
        pyfile_path = Path(notebook_tasks[notebook]["path"])
        notebook_path = pyfile_path.with_suffix(".ipynb")
        yield {
            "name": notebook,
            "actions": [
                """python -c "import sys; from datetime import datetime; print(f'Start """ + notebook + """: {datetime.now()}', file=sys.stderr)" """,
                f"jupytext --to notebook --output {notebook_path} {pyfile_path}",
                jupyter_execute_notebook(notebook_path),
                jupyter_to_html(notebook_path),
                mv(notebook_path, OUTPUT_DIR),
                """python -c "import sys; from datetime import datetime; print(f'End """ + notebook + """: {datetime.now()}', file=sys.stderr)" """,
            ],
            "file_dep": [
                pyfile_path,
                *notebook_tasks[notebook]["file_dep"],
            ],
            "targets": [
                OUTPUT_DIR / f"{notebook}.html",
                *notebook_tasks[notebook]["targets"],
            ],
            "clean": True,
        }
# fmt: on

###############################################################
## Task below is for LaTeX compilation
###############################################################


def task_compile_latex_docs():
    """Compile the LaTeX documents to PDFs"""
    file_dep = [
        "./reports/report_example.tex",
        "./reports/my_article_header.sty",
        "./reports/slides_example.tex",
        "./reports/my_beamer_header.sty",
        "./reports/my_common_header.sty",
        "./reports/report_simple_example.tex",
        "./reports/slides_simple_example.tex",
        "./src/example_plot.py",
        "./src/example_table.py",
    ]
    targets = [
        "./reports/report_example.pdf",
        "./reports/slides_example.pdf",
        "./reports/report_simple_example.pdf",
        "./reports/slides_simple_example.pdf",
    ]

    return {
        "actions": [
            # My custom LaTeX templates
            "latexmk -xelatex -halt-on-error -cd ./reports/report_example.tex",  # Compile
            "latexmk -xelatex -halt-on-error -c -cd ./reports/report_example.tex",  # Clean
            "latexmk -xelatex -halt-on-error -cd ./reports/slides_example.tex",  # Compile
            "latexmk -xelatex -halt-on-error -c -cd ./reports/slides_example.tex",  # Clean
            # Simple templates based on small adjustments to Overleaf templates
            "latexmk -xelatex -halt-on-error -cd ./reports/report_simple_example.tex",  # Compile
            "latexmk -xelatex -halt-on-error -c -cd ./reports/report_simple_example.tex",  # Clean
            "latexmk -xelatex -halt-on-error -cd ./reports/slides_simple_example.tex",  # Compile
            "latexmk -xelatex -halt-on-error -c -cd ./reports/slides_simple_example.tex",  # Clean
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": True,
    }

sphinx_targets = [
    "./docs/index.html",
]


def task_build_chartbook_site():
    """Compile Sphinx Docs"""
    notebook_scripts = [
        Path(notebook_tasks[notebook]["path"])
        for notebook in notebook_tasks.keys()
    ]
    file_dep = [
        "./README.md",
        "./chartbook.toml",
        *notebook_scripts,
    ]

    return {
        "actions": [
            "chartbook build -f",
        ],  # Use docs as build destination
        "targets": sphinx_targets,
        "file_dep": file_dep,
        "task_dep": [
            "run_notebooks",
        ],
        "clean": True,
    }
