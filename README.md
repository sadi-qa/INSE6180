# Association Mining Project (Apriori, FP-growth, Eclat, and TriHybrid-PPAM)

This project contains implementations of three classic frequent pattern mining algorithms plus a new hybrid algorithm **TriHybrid‑PPAM** that combines ideas from all three.

## Folder structure

```
association_mining_project/
  data/
    market_basket_50.csv
  src/
    utils/        # CSV loader, rule generation
    apriori/      # Apriori + AprioriTid + hash-tree
    fpgrowth/     # FP-tree + FP-growth miner
    eclat/        # Vertical tid-lists + Eclat
    hybrid/       # TriHybrid-PPAM (new algorithm)
    bench/        # Benchmarks + plotting
  main.py         # One CLI entrypoint
  requirements.txt
```

## 1) Environment setup (from scratch)

### 1. Install Python
Install Python 3.10+ (3.11 recommended). Verify:

```bash
python --version
pip --version
```

### 2. Create and activate a virtual environment

Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows (Git Bash / MINGW):
```bash
python -m venv .venv
source .venv/Scripts/activate
```

### 3. Install requirements
```bash
pip install -r requirements.txt
```

## 2) Run a single algorithm

Examples:

```bash
python main.py --algo apriori --minsup 0.3 --minconf 0.7
python main.py --algo fpgrowth --minsup 0.3 --minconf 0.7
python main.py --algo eclat --minsup 0.3 --minconf 0.7
python main.py --algo trihybrid --minsup 0.3 --minconf 0.7
```

## 3) Run benchmark + generate plots

```bash
python main.py --algo benchmark --plot
```

Outputs:
- `benchmark_results.csv`
- `support_vs_runtime_all.png`
- `support_vs_itemsets_all.png`

## Notes on the new algorithm (TriHybrid‑PPAM)

TriHybrid‑PPAM is a **pair‑pruned projected miner**:
- uses **Eclat tid-lists** to compute frequent pairs quickly,
- uses **FP-growth style projected mining** for larger patterns,
- uses an **adaptive choice** (FP-growth vs Eclat) per projection depending on density.

## New improved algorithm: A3Miner (adaptive selector)

A3Miner chooses an engine (Apriori, FP-growth, Eclat, or TriHybrid-PPAM) based on dataset shape.
Run it via:

```bash
python main.py --algo a3miner --minsup 0.3 --minconf 0.7
```

## Validation vs mlxtend

Validate outputs against mlxtend:

```bash
python validate_mlxtend.py --engine apriori --minsup 0.3
python validate_mlxtend.py --engine fpgrowth --minsup 0.3
python validate_mlxtend.py --engine eclat --minsup 0.3
python validate_mlxtend.py --engine a3miner --minsup 0.3
```
