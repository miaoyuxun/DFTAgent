# XC Functional Optimization Benchmark

Benchmark suite for evaluating and optimizing XC functional parameters using PySCF.

## Data Source

**GMTKN55** — Goerigk & Grimme's 55-subset main-group thermochemistry benchmark.
Cloned from [grimme-lab/GMTKN55](https://github.com/grimme-lab/GMTKN55).

55 subsets covering:
- Atomization energies (W4-11)
- Electron affinities / ionization potentials (G21EA, G21IP)
- Reaction energies (G2RC, DC13)
- Barrier heights (BH76, BHPERI, BHROT27)
- Non-covalent interactions (S22, S66, RG18)
- Conformational energies (PCONF21, SCONF, MCONF)
- Isomerization / radical stabilization (ISO34, RSE43)

## Directory Layout

```
benchmark/
├── README.md
├── GMTKN55/              ← reference data + molecular geometries
├── scripts/
│   ├── run_benchmark.py  ← PySCF runner: compute DFT energies vs reference
│   └── analyze_results.py← compare & rank results across functionals
└── results/              ← output JSON (gitignored)
```

## Quick Start

```bash
# Run PBE on key subsets
python benchmark/scripts/run_benchmark.py --xc "PBE,PBE" --basis def2-TZVP --all

# Run B3LYP
python benchmark/scripts/run_benchmark.py --xc "B3LYP" --basis def2-TZVP --all

# Run a single subset
python benchmark/scripts/run_benchmark.py --xc "PBE,PBE" --subset W4-11 --basis def2-TZVP

# Compare results
python benchmark/scripts/analyze_results.py --compare results_pbe.json results_b3lyp.json
```

## Key Subsets for XC Optimization

| Subset | Category | N reactions | What it tests |
|--------|----------|-------------|---------------|
| W4-11 | Atomization | 14 | XC balance for total energies |
| G21IP | Ionization | 19 | Exchange quality |
| G21EA | Electron affinity | 19 | Correlation quality |
| BH76 | Barriers | 76 | Exact exchange fraction |
| S22 | NCI (small) | 22 | Dispersion / mid-range correlation |
| S66 | NCI (medium) | 66 | Dispersion + electrostatics |
| RSE43 | Radical stability | 43 | Spin-symmetry / delocalization |
| ISO34 | Isomerization | 34 | Subtle energy differences |
| DC13 | Difficult cases | 13 | Multi-reference character |

## Functional Rungs & Tunable Parameters

| Rung | Example | Key tunable params |
|------|---------|-------------------|
| GGA | PBE | mu, kappa (enhancement factor) |
| GGA-hybrid | B3LYP, PBE0 | alpha (exact exchange fraction) |
| RSH | CAM-B3LYP, LC-wPBE | alpha, beta, omega |
| MGGA | SCAN, r2SCAN | mu, kappa (via enhancement) |
| MGGA-hybrid | SCAN0, wB97X | alpha, omega + GGA params |
