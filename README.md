# Efficient Reinforcement Learning for Online Multi-Workflow Scheduling in Edge Computing - Code Repository  
**Submission to IEEE Internet of Things Journal**  


[中文版](./README_zh.md) | [English Version](./README.md)


---

## 1. Introduction
This repository provides code to reproduce all experiments in the paper
*Efficient Reinforcement Learning for Online Multi-Workflow Scheduling in Edge Computing*.


**This repository contains:**
- Emulator of a Heterogeneous Edge Computing Environment
    - CPU-GPU heterogeneous affinity constraints
    - Random arrival of multi-workflow requests based on Poisson intervals
    - TFN (Triangular Fuzzy Number)-based performance and network fluctuations
    - Evaluation of scheduling results
        - Makespan, measured in milliseconds(ms)
        - Latency, measured in milliseconds(ms)
        - Energy, measured in 0.1 microjoules(0.1uJ)
- 100 Scheduler Configurations
    - 2 Meta-heuristic methods (Consolidating Scheduling)
    - 10 Heuristic methods (Dynamic-Programming Scheduling)
    - 16 Heuristic methods (Ready-Task Scheduling)
    - 72 Reinforcement Learning methods (Ready-Task Scheduling)


Running the scripts will reproduce Tables 2–6, Figure 2 and Figure 5.

To save time, we provide pre trained models and raw results.

You can start directly from [7. Generate Tables and Figure](#7-generate-tables-and-figure) (expected to take 1 minute).

Or you can start from [6. Model Evaluation](#6-model-evaluation) (expected to take 2 hours).

To start from the beginning, delete `./results` and start from [3. Install Dependencies](#3-install-dependencies) (expected to take 5 days).


---

## 2. Code Structure  
```
.
├── results/
│   └── <method_parameters>/
│       ├── auto_save.pth      # The auto-saved latest checkpoint.
│       ├── auto_save.json     # The auto-saved latest training log.
│       ├── best_model_<iteration>.pth   # The auto-saved best checkpoint.
│       ├── best_model_<iteration>.json  # The auto-saved best training log.
│       └── best_model_<iteration> <dataset> <from_seed> <to_seed>.json  # Raw evaluation result.
├── build_env_workflows.py     # Generates workflow datasets
├── env_workflows_0.json       # The single-workflow dataset (not used)
├── env_workflows_1.json       # The balanced workflow dataset
├── env_workflows_2.json       # The long-tailed workflow dataset
├── environment.py             # Edge computing environment and heuristic schedulers
├── requirements.txt           # Python dependencies
├── rl_env.py                  # The RL scheduling environment
├── rl_grpo.py                 # The RL scheduler (GRPO + DrGRPO)
├── rl_ppo.py                  # The RL scheduler (PPO)
├── RQ1.py                     # Main program
├── RQ1_test_raw.csv           # Raw evaluation results for human
├── RQ1_test.json              # Raw evaluation results for program
├── workflows_ali2018_origin.ipynb  # Script to pre-process ali2018 dataset
├── workflows_ali2018_origin.json   # Pre-processed ali2018 dataset
└── README.md                  # This file
```

---

## 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 4. Data Preparation  
Partition and preprocessing of Alibaba's 2018 workflow dataset:
```bash
python build_env_workflows.py
```
Output files will be stored in:
```
.
├── env_workflows_0.json       # The single-workflow dataset (not used)
├── env_workflows_1.json       # The balanced workflow dataset
└── env_workflows_2.json       # The long-tailed workflow dataset
```


---

## 5. Model Training
### Step 1: Learning rate search
This step trains models under different learning rates.

- This step is time-consuming (expected to take 6480 minutes).
    - MLP model training takes about 8 minutes
    - self-attention model training takes about 16 minutes
    - Transformer model training takes about 30 minutes
- Total runs: 2×3×2×3×2×5=360
    - 2 datasets: W1=balanced dataset, W2=long-tailed dataset
    - 3 policy models: v1=MLP, v2=self-attention, v3=Transformer
    - 2 reference models: HEFT=HEFT-based, self=frozen-policy-based
    - 3 training algorithm: DrGRPO, GRPO, PPO
    - 2 observation: O0=11D observation, O1=9D observation
    - 5 learning rate: 0.01, 0.001, 0.0001, 0.00001, 0.000001
```bash
python RQ1.py batch_train
```
Output files will be stored in:
```
.
└── results/
    └── <method_parameters>/
        ├── auto_save.pth      # The auto-saved latest checkpoint.
        ├── auto_save.json     # The auto-saved latest training log.
        ├── best_model_<iteration>.pth   # The auto-saved best checkpoint.
        └── best_model_<iteration>.json  # The auto-saved best training log.
```


### Step2: Plot Learning Curves
Visualize the training results to identify the best performing learning rate.
```bash
python RQ1.py draw_curves
```
Output files will be stored in:
```
.
└── img/
    └── <algorithm>_<dataset>_<observation>_<policy>_<reference>.svg
```

### Step3: Select Best Learning Rate and Best Checkpoint
Set the learning rate and checkpoint name in `RQ1.py` (`batch_test`, around line 255).

---

## 6. Model Evaluation
Evaluate all RL methods, heuristic methods, and metaheuristic methods on the two workflow datasets.

If you start from this step, please follow these steps: 
1. Download [best_models.zip](https://github.com/cqian-cs/RL-Scheduling/releases/download/v1/best_models.zip) (File Size: 235MB) and unzip it to `./results` folder.
2. Run command: `python RQ1.py batch_test overwrite=True` 

- This step takes about 2 hours:
    - 2 meta heuristic methods take about 1.5 hours in total
    - 26 heuristic methods take about 40 seconds in total
    - 72 RL methods take about 0.5 hours in total

Output files will be stored in:
```
.
├── results/
|   └── <method_parameters>/
|       └── best_model_<iteration> <dataset> <from_seed> <to_seed>.json  # Raw evaluation result.
├── RQ1_test_raw.csv         # Raw evaluation results for human
└── RQ1_test.json            # Raw evaluation results for program
```

- The description of method names is as follows:
    - `NSGA_`: consolidating scheduling, which schedules all tasks of all workflows at the same time.
    - `DP_`: dynamic-planning scheduling, which schedules a whole workflow upon request arrival.
    - `BE_`: best-effort scheduling (i.e., ready-task scheduling), where request arrival event or task completion event triggers the scheduler to schedule currently ready tasks.
    - `<RL algorithm>_<policy model>_<reference model>_<training dataset>_<observation version>`: RL-based scheduling:
        - RL algorithm: DrGRPO, GRPO, PPO
        - policy model: v1=MLP, v2=self-attention, v3=Transformer
        - reference model: HEFT=HEFT-based, self=frozen-policy-based
        - training dataset: W1=balanced dataset, W2=long-tailed dataset
        - observation version: empty=9D observation, O0=11D observation


## 7. Generate Tables and Figure

- Figure 2: `python RQ1.py plot_TFN`
- Figure 5: `python RQ1.py draw_fig_curve`
- Tables 2–6: `python RQ1.py show_tables`



## 8. Acknowledgements  
*   **Real-World Workflow Dataset:**  
    *   [cluster-trace-v2018](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/trace_2018.md) -- Cluster data collected from production clusters in Alibaba for cluster management research.
*   **Heterogeneous Edge Computing Environment Reference:**
    *   [TFN-based Fluctuation Model](https://ieeexplore.ieee.org/abstract/document/10622004/) -- IEEE TASE'24, “Cost-Driven Scheduling for Workflow Decision Making Systems in Fuzzy Edge-Cloud Environments”.
    *   [Topology of CPU-GPU nodes](https://www.usenix.org/conference/nsdi25/presentation/yang) -- NSDI'25, “GPU-Disaggregated Serving for Deep Learning Recommendation Models at Scale”.
    *   [Edge CPU Specification](https://www.intel.cn/content/www/cn/zh/products/sku/240361/intel-xeon-6756e-processor-96m-cache-1-80-ghz/specifications.html) -- Intel(R) Xeon(R) 6756E.
    *   [Edge GPU Specification](https://www.nvidia.com/content/dam/en-zz/solutions/data-center/a2/pdf/a2-datasheet.pdf) -- NVIDIA A2.
*   **Code References:**  
    *   [PPO-vs-GRPO](https://github.com/emparu/PPO-vs-GRPO) -- Comparison of GRPO and PPO in LunarLander and CartPole environments.
*   **Third-party Libraries:**  
    *   [pytorch](https://pytorch.org/) -- Tensors and Dynamic neural networks in Python with strong GPU acceleration.
    *   [pymoo](https://github.com/anyoptimization/pymoo) – pymoo: Multi-objective Optimization in Python.
    *   [numpy](https://numpy.org/) -- The fundamental package for scientific computing with Python.
    *   [matplotlib](https://matplotlib.org/) -- Matplotlib: Visualization with Python.
    *   (and others listed in requirements.txt)  

