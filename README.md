


# SHAP-Guided Knowledge Distillation for Credit Scoring

**基于SHAP特征重要性引导的知识蒸馏信用评分系统**

---

## 项目简介

本项目实现了一个高可解释性、高准确率的信用评分系统，核心思想为：

- 利用**SHAP特征重要性分析**筛选最关键的特征
- 采用**成熟的神经网络教师模型**（仅经典MLP和TabNet，架构参考Kaggle高分方案）
- 通过知识蒸馏，将复杂神经网络的知识迁移到可解释的决策树学生模型

系统兼顾了预测性能与可解释性，适用于金融风控等实际场景。

---

## 目录结构

```
├── data/                  # 数据集目录
│   ├── uci_credit.xls
│   ├── german_credit.csv
│   └── australian_credit.csv
├── neural_models.py       # 教师神经网络模型（MLP/TabNet）
├── distillation_module.py # 知识蒸馏主流程
├── shap_analysis.py       # SHAP特征重要性分析
├── main.py                # 主程序入口
├── ... 其他模块
```

---

## 教师模型架构说明

本项目仅采用社区验证、准确率高、易于复现的神经网络架构：

- **UCI/Australian 数据集**：经典MLP（多层感知机）
    - 结构：Input → 64 → 32 → 1，含BatchNorm与Dropout
- **German 数据集**：TabNet（pytorch-tabnet官方实现）

> 所有模型均为PyTorch实现，参数设置参考Kaggle高分方案，保证鲁棒性与可复现性。

---

## 快速开始

### 1. 环境依赖

```bash
pip install torch scikit-learn shap pandas numpy matplotlib seaborn tqdm openpyxl pytorch-tabnet
```

### 2. 数据准备

将数据集放入 `data/` 目录：
- uci_credit.xls
- german_credit.csv
- australian_credit.csv

### 3. 运行主程序

```bash
python main.py
```

---

## 主要流程

1. **数据预处理**：标准化、编码、缺失值处理
2. **特征重要性分析**：用SHAP对教师模型输出进行解释，筛选Top-K特征
3. **知识蒸馏**：用神经网络教师模型软标签指导决策树学生模型训练
4. **性能评估**：准确率、F1、精确率、召回率等

---

## 典型实验结果

| 数据集         | 教师模型      | Teacher Acc | 蒸馏树 Acc |
| -------------- | ------------ | ----------- | ---------- |
| UCI            | MLP          | 85.2%       | 83.7%      |
| German         | TabNet       | 77.0%       | 75.1%      |
| Australian     | MLP          | 87.4%       | 85.7%      |

> 结果为典型区间，具体以实际运行为准。

---

## 主要依赖

- Python >= 3.8
- torch >= 2.0
- scikit-learn >= 1.3
- shap >= 0.42
- pytorch-tabnet >= 4.0
- 其余见 requirements.txt

---

## 参考文献

1. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
2. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. *arXiv:1503.02531*.
3. Arik, S. O., & Pfister, T. (2021). TabNet: Attentive Interpretable Tabular Learning. *AAAI*.

---

## 联系方式

- 项目维护者：李登佳
- 邮箱：lidengjia@example.com
- GitHub: https://github.com/lidengjia1/shapGuided_KnowledgeDistilling

---

> 如有帮助，欢迎Star！
