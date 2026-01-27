# Efficient Reinforcement Learning for Online Multi-Workflow Scheduling in Edge Computing - 论文源码  
**Submission to IEEE Internet of Things Journal**  


[中文版](./README_zh.md) | [English Version](./README.md)


---

## 1. 简介
本仓库提供论文《Efficient Reinforcement Learning for Online Multi-Workflow Scheduling in Edge Computing》的复现代码。

**本仓库包含：**
- 模拟异构边缘计算环境
    - CPU-GPU异构亲和性约束
    - 基于泊松间隔的多工作流请求随机到达
    - 基于三角模糊数（TFN）的计算性能波动、通信性能波动
    - 调度结果评价
        - 整体用时（Makespan，单位：毫秒）
        - 个体用时（Latency，单位：毫秒）
        - 能耗（Energy，单位：0.1微焦耳）
- 100个调度方法配置
    - 2个元启发式方法（合并调度）
    - 10个启发式方法（动态规划调度）
    - 16个启发式方法（就绪任务调度）
    - 72个强化学习方法（就绪任务调度）


运行脚本可复现表2~表6、图2、图5。

为节省时间，我们提供了预训练模型与原始结果。

您可以直接从 [7. 生成图表](#7-生成图表)开始（预计用时1分钟）。

或者从[6. 模型评估](#6-模型评估)开始（预计用时2小时）。

如果从头开始，请删除`./results`文件夹，并从[3. 安装依赖](#3-安装依赖)开始（预计用时5天）。

---

## 2. 代码结构
```
.
├── results/
│   └── <method_parameters>/
│       ├── auto_save.pth      # 模型检查点（每训练10轮自动保存）
│       ├── auto_save.json     # 模型训练记录
│       ├── best_model_<iteration>.pth   # 最优模型的检查点（每训练到Reward新记录就自动保存）
│       ├── best_model_<iteration>.json  # 最优模型的训练日志
│       └── best_model_<iteration> <dataset> <from_seed> <to_seed>.json  # 原始评估结果
├── build_env_workflows.py     # 生成工作流数据集
├── env_workflows_0.json       # 单工作流数据集（没用到）
├── env_workflows_1.json       # 多工作流数据集（平衡分布）
├── env_workflows_2.json       # 多工作流数据集（长尾分布）
├── environment.py             # 边缘计算环境+启发式调度器
├── requirements.txt           # Python依赖
├── rl_env.py                  # 强化学习环境
├── rl_grpo.py                 # 强化学习调度器 (GRPO + DrGRPO)
├── rl_ppo.py                  # 强化学习调度器 (PPO)
├── RQ1.py                     # 主程序
├── RQ1_test_raw.csv           # 原始评估结果（表格）
├── RQ1_test.json              # 原始评估结果（JSON）
├── workflows_ali2018_origin.ipynb  # 预处理阿里2018工作流数据集的代码
├── workflows_ali2018_origin.json   # 预处理后的阿里2018工作流数据集
└── README.md                  # 本文件
```

---

## 3. 安装依赖
```bash
pip install -r requirements.txt
```

---

## 4. 数据准备
阿里2018工作流数据集的划分与预处理：
```bash
python build_env_workflows.py
```
输出：
```
.
├── env_workflows_0.json       # 单工作流数据集（没用到）
├── env_workflows_1.json       # 多工作流数据集（平衡分布）
└── env_workflows_2.json       # 多工作流数据集（长尾分布）
```


---

## 5. 模型训练
### 第一步：调整学习率
这一步在不同的学习率下训练模型。

- 这一步很耗时（预计用时6480分钟）：
    - 每次MLP模型训练大约8分钟
    - 每次自注意力模型训练大约16分钟
    - 每次Transformer模型训练大约30分钟
- 总共训练 2×3×2×3×2×5=360 个模型：
    - 2种工作流数据集：W1=balanced dataset, W2=long-tailed dataset
    - 3种策略模型：v1=MLP, v2=self-attention, v3=Transformer
    - 2种参照模型：HEFT=HEFT-based, self=frozen-policy-based
    - 3种训练算法：DrGRPO, GRPO, PPO
    - 2种观测特征：O0=11D observation, O1=9D observation
    - 5种学习率：0.01, 0.001, 0.0001, 0.00001, 0.000001
```bash
python RQ1.py batch_train
```
输出：
```
.
└── results/
    └── <method_parameters>/
        ├── auto_save.pth      # 模型检查点（每训练10轮自动保存）
        ├── auto_save.json     # 模型训练记录
        ├── best_model_<iteration>.pth   # 最优模型的检查点（每训练到Reward新记录就自动保存）
        └── best_model_<iteration>.json  # 最优模型的训练记录
```


### 第二步：画学习曲线
通过画学习曲线，来识别每种模型配置的最优学习率。
```bash
python RQ1.py draw_curves
```
输出：
```
.
└── img/
    └── <algorithm>_<dataset>_<observation>_<policy>_<reference>.svg
```

### 第三步：选择最佳学习率和最佳检查点
在`RQ1.py`（第255行附近的`batch_test`）中设置RL方法的学习率和检查点名称。

---

## 6. 模型评估
在两种工作流数据集下，评估所有强化学习算法、启发式算法、元启发式算法。

如果从这一步开始，请执行以下步骤：
1. 下载[best_models.zip](https://github.com/cqian-cs/RL-Scheduling/releases/download/v1/best_models.zip) （文件大小：235MB），解压到`./results`里。
2. 运行命令：`python RQ1.py batch_test overwrite=True` 

- 此步骤大约2小时
    - 2个元启发式方法共用时约1.5小时
    - 26个启发式方法共用时约40秒
    - 72个强化学习方法共用时约0.5小时

输出：
```
.
├── results/
|   └── <method_parameters>/
|       └── best_model_<iteration> <dataset> <from_seed> <to_seed>.json  # 原始评估结果
├── RQ1_test_raw.csv         # 原始评估结果汇总（表格）
└── RQ1_test.json            # 原始评估结果汇总（JSON）
```

- 方法命名格式：
    - `NSGA_`: 合并调度，一次性调度所有工作流的所有任务。
    - `DP_`: 动态规划调度，每到达一个工作流，就调度这个工作流。
    - `BE_`: 就绪任务调度，每当有任务准备就绪，就调度这个任务。
    - `<RL算法>_<策略模型>_<参照模型>_<训练数据集>_<观测特征>`: 强化学习调度：
        - RL算法：DrGRPO, GRPO, PPO
        - 策略模型：v1=MLP, v2=self-attention, v3=Transformer
        - 参照模型：HEFT=HEFT-based, self=frozen-policy-based
        - 训练数据集：W1=balanced dataset, W2=long-tailed dataset
        - 观测特征：empty=9D observation, O0=11D observation


## 7. 生成图表

- 图2: `python RQ1.py plot_TFN`
- 图5: `python RQ1.py draw_fig_curve`
- 表2~表6: `python RQ1.py show_tables`


## 8. 致谢  
*   **真实世界工作流数据集:**  
    *   [cluster-trace-v2018](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/trace_2018.md) -- Cluster data collected from production clusters in Alibaba for cluster management research.
*   **异构边缘计算环境参考:**
    *   [TFN性能波动模型](https://ieeexplore.ieee.org/abstract/document/10622004/) -- IEEE TASE'24, “Cost-Driven Scheduling for Workflow Decision Making Systems in Fuzzy Edge-Cloud Environments”.
    *   [CPU-GPU节点拓扑](https://www.usenix.org/conference/nsdi25/presentation/yang) -- NSDI'25, “GPU-Disaggregated Serving for Deep Learning Recommendation Models at Scale”.
    *   [边缘CPU规格](https://www.intel.cn/content/www/cn/zh/products/sku/240361/intel-xeon-6756e-processor-96m-cache-1-80-ghz/specifications.html) -- Intel(R) Xeon(R) 6756E.
    *   [边缘GPU规格](https://www.nvidia.com/content/dam/en-zz/solutions/data-center/a2/pdf/a2-datasheet.pdf) -- NVIDIA A2.
*   **代码参考:**  
    *   [PPO-vs-GRPO](https://github.com/emparu/PPO-vs-GRPO) -- Comparison of GRPO and PPO in LunarLander and CartPole environments.
*   **第三方库:**  
    *   [pytorch](https://pytorch.org/) -- Tensors and Dynamic neural networks in Python with strong GPU acceleration.
    *   [pymoo](https://github.com/anyoptimization/pymoo) -- pymoo: Multi-objective Optimization in Python.
    *   [numpy](https://numpy.org/) -- The fundamental package for scientific computing with Python.
    *   [matplotlib](https://matplotlib.org/) -- Matplotlib: Visualization with Python.
    *   (and others listed in requirements.txt)  

