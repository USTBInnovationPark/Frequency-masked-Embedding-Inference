该仓库为Frequency-masked Embedding Inference(FEI)的官方代码库，该版本为最初提交于AAAI-2025的论文对应代码

# Repository Structure
```
- config/                所有模型与训练的配置
- datasets_clsa/         分类数据集构建
- datasets_reg/          回归数据集构建
- models/                包含Baseline实现与FEI架构代码
- train/                 所有训练代码的基类，以及训练日志和结果的保存
- util/                  所有的工具方法，包含了FEI的频率遮罩代码
- experiments.py         训练和测试的运行入口
```

# Requirements
```
numpy~=1.24.3
torch~=2.0.1
scikit-learn~=1.3.0
matplotlib~=3.7.2
tsaug~=0.2.1
pandas~=2.0.3
```

# Preparing Datasets
## Classification Datasets
所有的分类数据集格式应符合train.pt/test.pt/val.pt的结构，pt文件内包含键为"samples","labels"以及值为对应的样本和标签数据的字典键值对，详细见TF-C的数据集结构。

数据集下载后应放入datasets_clsa文件夹，以Gesture数据集为例，正确目录结构应该如下：
```
- datasets_clsa
  - Gesture
    - train.pt
    - test.pt
    - val.pt
```

## Regression Datasets
回归数据集下载后无需额外处理，直接放入datasets_reg即可:
```
- datasets_clsa
  - CMAPSS
    - FD001
    - FD002
    - FD003
    - FD004
```

# Quick Start
快速开始预训练，使用如下命令：
> python ./experiment.py --task_type=p --method=FEI

预训练结束后，可在项目的`train/model_result/`内找到对应的预训练日志和结果，使用该目录作为预训练模型开展验证的方法如下：
> python ./experiment.py --model=./train/model_result/your_model_path --task_type=l --task=c --dataset=FDB --method=FEI

可通过修改--task_type、--dataset等参数调整任务类型和测试数据集，使用如下命令获取有关运行参数的更多帮助：
> python ./experiment.py -h

更多代码细节将在论文录用后、代码可公开后描述
