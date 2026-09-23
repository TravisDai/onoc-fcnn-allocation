# onoc-fcnn-allocation
[![DOI](https://zenodo.org/badge/1382142586.svg)](https://doi.org/10.5281/zenodo.22904099)

Simulator and reproduction artefact for *Collective-Aware Core Allocation for Fully Connected
Neural Network Training on a Ring Optical Network-on-Chip* (under review at Future Generation
Computer Systems).

Everything the paper reports is generated from this repository: every table, every figure and
every number quoted in the text. There are no hand-typed results.

The study asks two questions. How many cores should each layer of a fully connected network own
on a wavelength-slotted optical ring, and which backward collective should be used: reduce the
partial errors (**R**), or replicate the transposed weights and broadcast (**T**)? Section 6.8
then rebuilds the network on a physically constrained, Hummingbird-style organisation in which
every optical transmission stays within an 8 dB power split.

## Quick start

```bash
git clone https://github.com/TravisDai/onoc-fcnn-allocation.git
cd onoc-fcnn-allocation
pip install -r requirements.txt

python -m pytest -q          # 23 tests, about 3 minutes
python physhb.py             # physical-layer numbers of Section 3.6, instant
python make_tables.py        # rebuild every table, figure and macro from results/
```

`make_tables.py` works on the CSVs already in `results/`, so you can regenerate the paper's
tables and figures without re-running any experiment. To recompute the CSVs themselves, see
[Running the experiments](#running-the-experiments).

Python 3.11 with numpy, pandas, matplotlib, numba and pytest. No GPU, no optical hardware and
no network access are needed. Everything runs on one machine.

## What is being simulated

A ring of `m` cores connected by a multiple-writer multiple-reader optical network-on-chip.
Each waveguide carries `λ_max` wavelength channels. A transmission occupies one wavelength slot
and can be received by any number of cores that tap the waveguide.

Training one mini-batch iteration of a fully connected network on that ring means:

1. **Ownership.** Each layer's neurons are partitioned across a chosen number of cores, `m_i`.
   More cores mean less computation each but more sources sharing the wavelength channels.
2. **Forward.** Every owner of layer `i` needs all activations of layer `i-1`, which is a
   broadcast per source.
3. **Backward.** Either **R**, a reduce-scatter of partial errors, or **T**, a broadcast of the
   errors where each owner also stores the rows of the next layer's weights that leave its own
   neurons. T does more arithmetic and uses more memory but replaces the reduction by a
   broadcast, which the optical ring carries once.
4. **Allocation.** Choose `m_1 … m_ℓ` to minimise the iteration time. The chain structure makes
   an exact dynamic program possible for two mapping families; a coordinate search covers the
   rest.

Section 6.8 replaces the shared, tunably split ring by single-writer *lanes*: each cluster hub
owns lanes that are tapped by the next six clusters, and every sixth cluster regenerates the
data and forwards it on its own lanes. That keeps each split at 7.8 dB and each link budget
closed at 40 Gb/s, which the shared-ring model does not.

## Repository layout

| Path | Role |
|---|---|
| `onocsim.py` | Schedule-level simulator (numba). Ownership, computation, slot service, energy, SRAM liveness, relaxed initialiser, exact chain DP, coordinate search, analytical mesh bound. |
| `refmodel.py` | Independent, slow, dictionary-based implementation of the same equations. Shares no code with `onocsim.py`; used only for cross-checking. |
| `physhb.py` | Physical layer of Section 3.6: loss of one transmission to `F` receivers, the 8 dB splitting limit, the rate-dependent link budget, WDM crosstalk. |
| `hbsim.py` | Hummingbird-style clustered network of Section 6.8: hubs, dedicated lanes, fixed taps, relays, cluster fabric, laser power from the link budget. |
| `hbref.py` | Independent reference for one transition of that network, including the relay chains. |
| `experiments.py`, `ring_size.py` | Main experiment matrix; ring-size study. Write `results/*.csv`. |
| `hb_experiments.py`, `run_hb.sh` | Section 6.8 experiments: `main`, `lanes`, `rate`, `clusters`, `dcfg`. Write `results/hb_*.csv`. |
| `make_tables.py` | Turns the CSVs into `paper/generated/*.tex` and `paper/figs/fig_*.pdf`. |
| `parse_traces.py`, `traces/compute_traces.csv` | Profiler traces of the C/GSL/BLAS implementation: per-core forward and backward kernel times for every network, period, batch size and core count, 100 to 300 repetitions each, compressed from 5.4 GB of logs. |
| `test_*.py` | 23 tests: simulator against the reference, DP against brute force, the propositions by exhaustive check on small rings, the physical-layer constraints. |
| `paper/` | Figures, and the generated tables and macros that the manuscript includes. The manuscript source itself is not in this repository while the paper is under review. |

## Running the experiments

Times are for two cores of an Intel Core i5 3200 host with 32 GB of memory.

| Command | What it produces | Time |
|---|---|---|
| `python -m pytest -q` | Correctness of everything below | ~3 min |
| `python experiments.py main` | `results/main.csv`: all six networks, four batch sizes, both collectives, both wavelength counts, three mapping families, four allocation methods | ~30 min |
| `python experiments.py enoc` | `results/enoc.csv`: the electrical-mesh reference of Section 6.7 | ~5 min |
| `python experiments.py sens` | `results/sens.csv`: sensitivity to reconfiguration time, bit rate, core throughput and flit serialisation | ~5 min |
| `python ring_size.py` | `results/ringsize.csv`: rings of 64, 256 and 1,000 cores | ~5 min |
| `python experiments.py dp` | `results/dp.csv`: exact optima that the coordinate search is measured against | 2-3 h |
| `./run_hb.sh` | `results/hb_*.csv`: the Hummingbird-style network, lane-count, lane-rate and cluster-count studies | ~12 min |
| `python make_tables.py` | `paper/generated/*.tex`, `paper/figs/fig_*.pdf` | ~20 s |

`run_all.sh` runs the first five in order. Each experiment writes its log to `results/`.

`make_tables.py` writes the tables and the macro file that the manuscript includes into
`paper/generated/`, and the figures into `paper/figs/`. The manuscript source is not part of this
repository while the paper is under review.

Figure 9 is drawn in TikZ; rebuild it with `pdflatex figs/src/hb_network.tex` and copy the PDF
into `paper/figs/`.

## How the results reach the paper

`make_tables.py` writes two kinds of output. Tables go to `paper/generated/tab_*.tex`, which the
manuscript includes directly. Every number quoted in the running text is a LaTeX macro in
`paper/generated/results_macros.tex`, so a sentence such as "T is 1.3-5.2x faster than R" is
written in the source as macros and cannot drift from the data.

| Paper | Code |
|---|---|
| Eq. (5), per-neuron FLOPs of R and T | `onocsim.flops_per_neuron`, `compute_tables` |
| Eqs. (6)-(10), emitters, payloads, slot service | `onocsim.transition` (mode 0 broadcast, mode 1 segmented reduction) |
| Eq. (11), network energy | `onocsim.transition`, `evaluate` |
| Eq. (12) and Table 2, SRAM by liveness | `onocsim.sram_peak` |
| Eq. (13), iteration time | `onocsim.evaluate` |
| Eqs. (15)-(16), loss of a transmission and link budget | `physhb.il_db`, `budget_db`, `f_max` |
| Eq. (17) and the scaling corollary | `onocsim.relaxed_init` |
| Proposition 1 and the chain DP | `onocsim.pair_matrix`, `chain_dp` |
| Proposition 2, mapping-family metrics | `onocsim.schedule_metrics`, `local_fraction` |
| Section 6.7, electrical mesh bound | `refmodel.mesh_transition`, `mesh_eval` |
| Section 6.8, transition with relays | `hbsim.htransition` |

## How the model is validated

* **Two implementations.** `refmodel.py` and `hbref.py` implement the same equations with plain
  Python dictionaries and share no code with the numba simulators. The tests check agreement on
  hundreds of random cases, including energy, SRAM and schedule metrics.
* **Exact optima.** The chain dynamic program is checked against brute force on small instances,
  and the coordinate search is measured against the dynamic program on the full grid.
* **The propositions** are checked exhaustively on rings of 3 to 10 cores with 2 to 5 layers.
* **The physical layer** is checked constraint by constraint: every window used closes the link
  budget, no split exceeds 8 dB, every destination is covered within the fan-out.

## Computation model

The main results are trace-driven: `Model(n, cfg, trace_net="NN2")` reads
`traces/compute_traces.csv` and builds per-load times. Forward kernel times are used unchanged.
The profiled backward kernel forms the weight gradient, so the extra products of R and T are
charged at the efficiency measured at the same per-core load. The sensitivity study uses an
analytical FLOP model instead, over a sixteenfold range of core throughput, and reaches the same
conclusions.

To regenerate the trace CSV from the raw profiler logs, run
`python parse_traces.py "trace NN1-3.zip" ...`. Four NN6 log files are named B34, B52, B36 and
B72 but contain batch-32 runs; the parser corrects them.

## Parameters worth knowing

* `D_cfg`, the microring reconfiguration time per slot, is the least certain parameter. The base
  value is 10 ns and the paper sweeps it from 1 ns to 1 µs.
* `P_laser = 0.645 mW` is the optical launch power per active wavelength.
* The physical layer of Section 3.6 applies the 8 dB limit to the **splitting loss only**, which
  caps a transmission at six receivers whatever the devices. The total budget then follows from
  the laser power and the receiver sensitivity: `-22.3 dBm` OMA at 10 Gb/s, scaled by 15 dB per
  decade of bit rate.

## Citation

Cite the software by its concept DOI, which always resolves to the newest release:

```bibtex
@misc{dai2026onocsim,
  author    = {Dai, Fei},
  title     = {onoc-fcnn-allocation: simulator for collective-aware core allocation
               on a ring optical network-on-chip},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22904099},
  note      = {\url{https://github.com/TravisDai/onoc-fcnn-allocation}}
}
```

Release v1.0.1, the version the paper was built from, is archived at
[10.5281/zenodo.22909487](https://doi.org/10.5281/zenodo.22909487).
The paper it accompanies is under review at Future Generation Computer Systems.

## Licence

MIT, see [LICENSE](LICENSE).
