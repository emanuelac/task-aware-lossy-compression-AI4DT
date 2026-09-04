# Reproducibility

Every number, table, and figure in the paper derives from the per-dataset
`results/<dataset>/<stamp>/summary.csv` files produced by the offline
pipeline and read back by `notebook/results_AI4DT.ipynb`. Nothing is
hand-typed. Reproducing from scratch is three steps.

## 1. Environment

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## 2. Data

The five datasets are public benchmarks from the UCI ML Repository. Download
each and place the raw files under `data/raw/`, then run its prep script,
which writes the single-stream CSV the pipeline expects:

```bash
python3 scripts/prep_uci_har.py          	# UCI HAR https://archive.ics.uci.edu/dataset/240
python3 scripts/prep_opportunity.py		# OPPORTUNITY https://archive.ics.uci.edu/dataset/226
python3 scripts/prep_hydraulic.py       	# Hydraulic Systems https://archive.ics.uci.edu/dataset/447
python3 scripts/prep_beijing.py           	# Beijing Air Quality https://archive.ics.uci.edu/dataset/501
python3 scripts/prep_metro.py            	# Metro Traffic https://archive.ics.uci.edu/dataset/492
```

Each prep script's module docstring lists the exact raw filenames it expects
under `data/raw/`.

## 3. Run the evaluation

```bash
python3 scripts/run_eval.py --dataset hydraulic   # one dataset
./scripts/run_all.sh                              # all five
```

Output: `results/<dataset>/<stamp>/summary.csv` — one row per
`(task, subtask, method, train_fold, eval_fold)`, carrying the raw task
metrics, the storage columns, and the normalised `loss_tilde` /
`storage_tilde` / `multi_obj`. All randomness (AE seeds, reference-set
subsampling) is seeded, so a re-run is deterministic.

## 4. Regenerate tables and figures

Open `notebook/results_AI4DT.ipynb` and **Run All**. It reads the latest
`summary.csv`.

