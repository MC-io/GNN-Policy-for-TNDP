# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Research code (GPL, by Andrew Holliday) for the **Transit Network Design Problem**: a GNN policy trained with REINFORCE generates sets of bus routes over a city graph, and is compared against / combined with metaheuristics (Bee Colony Optimization, a hyperheuristic). Evaluation is on the Mumford benchmark (Mandl, Mumford0–3) and on synthetic city graphs.

Version control is **Mercurial** (`.hg/`), not git. The README recommends a `pre-commit = pytest` hook in `.hgrc`.

## Environment

- Pinned stack in `environment.yml` (conda env `ptlearn`): Python 3.8, PyTorch 1.12, PyG 2.1, hydra-core 1.2, optuna 3.0. `cc_requirements.txt` is the pip equivalent used on Compute Canada clusters. The system Python on this machine (3.13) does not have these installed; code relies on old APIs (e.g. `np.asscalar`, torch 1.12 behaviour), so use the pinned environment.
- **The repo root must be on `PYTHONPATH`.** Modules import each other as top-level packages (`from torch_utils import ...`, `from simulation.citygraph_dataset import ...`, `import learning.utils`), and scripts are run as `python learning/<script>.py` from the repo root. The slurm jobs set `PYTHONPATH` explicitly for this reason.
- Training defaults to CUDA; pass `experiment.cpu=true` to run on CPU.

## Commands

```bash
# tests (run from repo root with PYTHONPATH=.)
pytest
pytest tests/test_torch_utils.py
pytest tests/test_torch_utils.py::test_basic_floyd_warshall

# generate a synthetic training dataset (use equal --min/--max: batching graphs of
# different sizes is known to be buggy)
python simulation/citygraph_dataset.py datasets/20_nodes/mixed mixed -n 32768 --min 20 --max 20

# train (hydra; config cfg/train.yaml)
python learning/inductive_route_learning.py dataset.kwargs.path=datasets/20_nodes/mixed +run_name=my_run
#   add experiment/cost_function=op or =pp to train on a single operator/passenger perspective

# evaluate a trained model on a Mumford city
python learning/eval_route_generator.py +model.weights=output/inductive_my_run.pt \
    eval.dataset.path=/path/to/mumford/Instances +eval=mandl

# BCO, Neural BCO, hyperheuristic baselines
python learning/bee_colony.py eval.dataset.path=/path/to/Instances +eval=mandl
python learning/bee_colony.py --config-name neural_bco_mumford +model.weights=W.pt eval.dataset.path=... +eval=mandl
python learning/hyperheuristics.py eval.dataset.path=... +eval=mandl

# run a script over all Mumford cities
scripts/mumford_eval.sh learning/bee_colony.py bco_mumford [extra hydra args]
```

Note: the README refers to `learning/inductive_route_training.py` and `scripts/generate_datasets.sh` calls `learning/citygraph_dataset.py`; the actual paths are `learning/inductive_route_learning.py` and `simulation/citygraph_dataset.py`.

Outputs: TensorBoard logs in `training_logs/<run_name>`, weights in `output/<run_name>.pt` (run name is prefixed, e.g. `inductive_`, `bco_`, `neural_bco_`), per-epoch checkpoints in `output/<run_name>_checkpoints/`, route sets from BCO in `output_routes/`. Mumford data is downloaded separately (URL in README); pretrained weights are at https://www.cim.mcgill.ca/~mrl/projs/transit_learning/.

## Configuration (hydra)

All current entry points use `@hydra.main(config_path="../cfg", ...)`. Config composition:

- Top-level configs (`train.yaml`, `eval_model_*.yaml`, `bco_*.yaml`, `neural_bco_*.yaml`, `hyperheuristic*.yaml`, `hyperparam_opt.yaml` for the optuna sweeper) pull in defaults groups:
  - `experiment/standard.yaml` — seed, `logdir`, `cpu`, `symmetric_routes`, and the `cost_function` subgroup (`mine` = weighted demand/route time with `variable_weights` sampled during training; `op`/`pp` = operator/passenger-only weightings; `nikolic` = Nikolić-style cost).
  - `model/*.yaml` — `common` (embed_dim, dropout, nonlin), `backbone_gn` (graph net type), and the `route_generator` subgroup (`biased` → `PathCombiningRouteGenerator`, `unbiased` → `UnbiasedPathCombiner`, `nodewalker` → `NodeWalker`).
  - `eval/*.yaml` — added with `+eval=<city>`; sets `n_routes` list, route length bounds, and `dataset` (type `mumford` + city, or `pickle`).
- `.hgignore` ignores `*.yaml` (and `*.csv`, `*.pt`, `*.pkl`, `*.png`, …), so new config files must be explicitly `hg add`ed.

## Architecture

The shared setup path is `learning/utils.py:process_standard_experiment_cfg`, used by every hydra entry point. It seeds RNGs, picks the device, creates the `SummaryWriter`, builds the model via `build_model_from_cfg` (maps `model.route_generator.type` string → class in `learning/models.py`, `backbone_gn.net_type` string → graph net class) and loads `model.weights` if given, and builds the cost module from `experiment.cost_function.type`. When adding a new model or cost type, register it there.

Data flow:

1. **City graphs** — `simulation/citygraph_dataset.py`. `CityGraphData` is a PyG `HeteroData` with node type `STOP_KEY='stop'` and edge types `STREET_KEY` and `DEMAND_KEY` (`ROUTE_KEY` is added by models). It also carries dense per-graph matrices (`demand`, `drive_times`, `street_adj`, `nexts` from Floyd–Warshall) that are *not* concatenated when batching (`__cat_dim__` returns `None`). Node features are 8-dim (the training loop asserts this). `CityGraphData.from_mumford_data` loads Mumford instances; `DynamicCityGraphDataset` generates synthetic graphs (mst, knn, voronoi, grid, circulant, small-world, `mixed`); `CityGraphDataset` wraps a pickled `graph_list.pkl` and caches a collated `processed/collated.pt`. Old pickled datasets need `CityGraphData` importable for unpickling.
2. **Models** — `learning/models.py` (~4.6k lines, many legacy classes). The active ones subclass `RouteGeneratorBase`; their `forward(data, n_routes, min_route_len, max_route_len, cost_weights)` returns a `PlanResults` namedtuple with `routes`, `route_logits`, `stop_logits`. `FeatureNorm` modules collect stats during epoch 0 and are frozen via `update_and_freeze_feature_norms()`.
3. **Cost** — `simulation/transit_time_estimator.py`: `MyCostModule` / `NikolicCostModule` (`torch.nn.Module`s) compute batched route-set costs from routes + graph data; `MyCostModule.sample_variable_weights` draws per-graph demand/route-time weights that are also fed to the model.
4. **Training** — `learning/inductive_route_learning.py:train`: REINFORCE with a selectable baseline (`neural` default, `rolling average`, `greedy`, `none`). With the neural baseline, epoch 0 only trains the baseline. Picks a random `n_routes` in the `eval_n_routes` range each batch; keeps the best model by validation cost. Also usable as an optuna objective (`setup_and_train(cfg, trial)`, `learning/tune_hyperparams.py`).
5. **Baselines / search** — `learning/bee_colony.py` (BCO; with `neural_bees`/`init_from_model` it becomes Neural BCO, using the model to propose route replacements) and `learning/hyperheuristics.py`. Both are run through `learning/utils.py:test_method`, which iterates a dataloader and logs/prints costs.

Route sets are passed around as padded tensors: `torch_utils.get_batch_tensor_from_routes` converts nested lists to a `(batch, n_routes, max_len)` tensor padded with `-1`. `torch_utils.py` also holds the batched `floyd_warshall` and path/feature aggregation utilities.

### Legacy code

An earlier MATSim / "timeless simulator" pipeline still lives alongside the above and uses argparse rather than hydra: `simulation/simulator.py` (`DisaggregatedDemandSim`, `ConstantDemandSim`, reading MATSim XML envs like those in `tests/envs/`), `simulation/matsim_*.py`, `simulation/timeless_sim.py`, `world/`, `config_utils.py`, and in `learning/`: `learn_routes.py`, `bagloee.py`, `proxy.py` (has a broken `from models import`), `build_dataset.py`, `replay_buffer.py`, `learn_schedule.py`, `learn_tsp.py`. `tests/test_simulators.py` exercises both the legacy simulator and the current cost modules.

`scripts/` contains sweep shell scripts and plotting/table scripts for the ITSC and NeurIPS papers; `slurm_jobs/` are Compute Canada job files with hard-coded cluster paths (some still use the pre-hydra argparse CLI).
