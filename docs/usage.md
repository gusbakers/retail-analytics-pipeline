# Running the project

## GitHub Codespaces

1. Open this repository (or the review branch), select **Code → Codespaces → Create codespace**.
2. The dev container installs requirements-dev.txt, including optional SHAP and notebook tools.
3. In the terminal, run:

    python main.py --source real --explain

The public UCI ZIP is downloaded into data/raw/, cached after reading, and verified with hashes. Raw data and the generated DuckDB/processed files are not committed. The first load includes reading the two-year Excel workbook; subsequent runs reuse the verified cache.

To use only the original demand simulation, without downloading the real dataset:

    python main.py --source demo --explain

Real mode also writes the daily-order capacity-planning case to reports/operations/. Weekly SKU research remains under reports/metrics/. Demo reports go to reports/demo/ so they do not overwrite the real-sales snapshot. Pricing and feedback remain explicitly simulated in either mode.

## Python environment

From the repository root:

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements-dev.txt
    python -m pytest
    python main.py --source real --explain

For the CLI alone: **python -m pip install -r requirements.txt**. Add explanations with **python -m pip install -e '.[explain]'**.

## Notebooks

Open the four numbered files with the Python kernel belonging to the installed environment. Notebook 01 loads/audits real sales and prepares the weekly SKU and daily-order panels. Notebook 02 compares daily operations, the preserved weekly simulation and SKU research with separate targets and selected blends. Notebooks 03 and 04 generate their independent scenarios. Explanations sit below the code cells.

The committed metrics/figures are a recorded execution, not live production monitoring. Rerunning main.py regenerates them and records package versions, source checksums, horizon and cutoff in reports/metrics/run.json.

## Databricks

The notebooks now depend on the repository's shared package. Import the **whole repository** through a Git folder instead of uploading one standalone notebook. Install the package/dependencies for that compute and run the notebooks in sequence. Resolve the checkout root by pyproject.toml.

The loader writes standard Python paths beneath the checkout; this version was verified in Python/Codespaces-style execution, **not deployed or tested in a remote Databricks workspace**. If the workspace restricts writes/network access, adapt the root/storage paths to a volume with permission. It does not create a paid Job or any cloud infrastructure.

## After practicing

Stop the Codespace from GitHub **Codespaces → Stop codespace**. If you no longer need its disk, delete that Codespace after keeping any changes in GitHub. This project starts no persistent server, Airflow scheduler or cloud deployment.

## Recorded verification

Both CLI modes and all SHAP reconstructions were executed. Notebook code cells were executed sequentially in a Python process with captured IPython display outputs; this does not claim a launched Jupyter kernel or a remote Databricks deployment. Captured outputs are committed for review; rerun in the documented environment to reproduce them.
