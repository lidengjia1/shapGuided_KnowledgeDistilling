


# SHAP-Guided Knowledge Distillation for Credit Scoring

**基于SHAP特征重要性引导的知识蒸馏信用评分系统**

---

## 🎯 项目简介

本项目实现了一个高可解释性、高准确率的信用评分系统，核心思想为：

- 利用**SHAP特征重要性分析**筛选最关键的特征
- 采用**PyTorch深度神经网络**作为高性能教师模型
- 通过知识蒸馏，将复杂模型的知识迁移到可解释的决策树学生模型

系统兼顾了预测性能与可解释性，适用于金融风控等实际场景。

---

## 📁 目录结构

```
├── data/                     # 数据集目录
│   ├── uci_credit.xls       # UCI信用卡违约数据集
│   ├── german_credit.csv    # German信用评分数据集  
│   └── australian_credit.csv # Australian信用审批数据集
├── neural_models.py          # PyTorch神经网络教师模型
├── distillation_module.py    # 知识蒸馏主流程
├── shap_analysis.py          # SHAP特征重要性分析
├── experiment_manager.py     # 实验管理和结果导出
├── data_preprocessing.py     # 数据预处理模块
├── main.py                   # 主程序入口
└── results/                  # 实验结果和可视化
```

---

## 🧠 教师模型架构说明

### 🔥 PyTorch深度神经网络

本项目使用PyTorch实现的深度神经网络作为教师模型，针对不同数据集优化网络结构：

#### German信用数据集网络
```python
nn.Sequential(
    nn.Linear(input_dim, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 32),
    nn.ReLU(), 
    nn.Dropout(0.3),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Sigmoid()
)
```

#### Australian信用数据集网络
```python
nn.Sequential(
    nn.Linear(input_dim, 128),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
)
```

#### UCI信用数据集网络
```python
nn.Sequential(
    nn.Linear(input_dim, 256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Dropout(0.5),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.BatchNorm1d(128),
    nn.Dropout(0.4),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.BatchNorm1d(64),
    nn.Dropout(0.3),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
)
```

### 🏆 模型特点

- **自适应架构**: 根据数据集规模和复杂度调整网络深度和宽度
- **正则化技术**: 使用Dropout和BatchNorm防止过拟合
- **早停机制**: 基于验证集性能自动停止训练
- **GPU加速**: 自动检测并使用CUDA加速（如果可用）

---

## 🚀 快速开始

### 1. 环境依赖

```bash
pip install torch scikit-learn shap pandas numpy matplotlib seaborn tqdm openpyxl
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

程序将自动完成以下阶段：
- 📊 数据预处理和特征工程
- 🧠 PyTorch神经网络教师模型训练
- 🔍 SHAP特征重要性分析
- 🌳 基线决策树训练
- 🌟 全特征知识蒸馏
- 🧪 Top-k特征知识蒸馏
- 📈 结果分析和可视化

---

## 🔬 主要流程

1. **数据预处理**：标准化、类别编码、缺失值处理、6:2:2数据划分
2. **教师模型训练**：PyTorch深度神经网络，使用早停和正则化技术
3. **SHAP特征分析**：识别最重要的Top-k特征 (k=5,6,7,8)
4. **基线决策树**：固定参数（max_depth=5），无参数优化
5. **知识蒸馏**：
   - 温度参数：T=1,2,3,4,5 (间隔1)
   - 加权比例：α=0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 (间隔0.1)
   - 树深度：D=4,5,6,7,8
6. **综合评估**：生成详细的性能对比报告

---

## 📊 最新实验结果

### 教师模型性能（PyTorch神经网络）

| 数据集 | 教师模型 | 准确率 | F1分数 | 精确率 | 召回率 |
|--------|----------|--------|--------|--------|--------|
| UCI | CatBoost | 0.8147 | 0.7912 | 0.7902 | 0.8108 |
| German | CatBoost | 0.7650 | 0.7386 | 0.7755 | 0.7850 |
| Australian | CatBoost | 0.8696 | 0.8699 | 0.8406 | 0.8406 |

### 知识蒸馏性能

| 数据集 | 全特征蒸馏 | Top-k蒸馏 | 基线决策树 | 蒸馏提升 |
|--------|------------|-----------|------------|----------|
| UCI | 0.8200 | 0.7870 | 0.8117 | +0.83% |
| German | 0.7300 | 0.7550 | 0.6800 | +7.35% |
| Australian | 0.8623 | 0.8188 | 0.8551 | +0.84% |

> 结果显示知识蒸馏能有效提升决策树性能，特别是在German数据集上获得显著提升。

---

## 🛠️ 主要依赖

- **Python** >= 3.8
- **PyTorch** >= 2.0 (深度学习框架)
- **CatBoost** >= 1.2 (表格数据SOTA模型)
- **XGBoost** >= 1.7 (传统树模型)
- **scikit-learn** >= 1.3 (机器学习基础库)
- **SHAP** >= 0.42 (可解释AI)
- **pytorch-tabnet** >= 4.0 (TabNet实现)
- **pandas**, **numpy**, **matplotlib**, **seaborn**, **tqdm**

---

## 📈 输出结果

### 自动生成的文件

```
results/
├── comprehensive_model_comparison.xlsx    # 综合模型对比表
├── master_results_table_YYYYMMDD.xlsx   # 主结果表
├── teacher_models_results.xlsx          # 教师模型详细结果
├── combined_shap_analysis.png           # SHAP特征重要性可视化
├── processed_data.pkl                   # 预处理后的数据
├── shap_results.pkl                     # SHAP分析结果
└── distillation_results.pkl             # 知识蒸馏结果
```

### 关键指标

- **准确率 (Accuracy)**: 整体预测正确率
- **F1分数**: 精确率和召回率的调和平均
- **精确率 (Precision)**: 预测为正例中实际为正例的比例
- **召回率 (Recall)**: 实际正例中被正确预测的比例

---

## 🔧 技术特点

### 高性能教师模型
- 基于TabSurvey基准测试选择最优模型
- CatBoost作为主要教师，在表格数据上达到SOTA性能
- 自动特征工程，无需手动调参

### 智能特征选择
- SHAP values量化特征重要性
- 自动筛选Top-k最重要特征
- 可视化特征贡献度分析

### 可解释知识蒸馏
- 温度缩放软化教师输出 (T=1,2,3,4,5)
- 加权比例全覆盖搜索 (α=0.0-1.0, 间隔0.1)
- 基线模型固定参数 (无超参数优化)
- 保持决策树的可解释性

### 完整实验流程
- 端到端自动化实验
- 详细的性能对比分析
- 丰富的可视化输出

---

## 📚 相关文献

- **TabSurvey**: "Revisiting Deep Learning Models for Tabular Data"
- **SHAP**: "A Unified Approach to Interpreting Model Predictions"
- **Knowledge Distillation**: "Distilling the Knowledge in a Neural Network"
- **CatBoost**: "CatBoost: unbiased boosting with categorical features"

---

## ⚙️ 参数配置

### 基线模型参数（固定，无优化）
```python
DecisionTreeClassifier(
    max_depth=5,           # 固定树深度
    min_samples_split=2,   # 最小分割样本数
    min_samples_leaf=1,    # 最小叶子样本数
    max_features='sqrt',   # 特征选择策略
    random_state=42        # 随机种子
)
```

### 知识蒸馏参数
- **温度参数 (Temperature)**: T ∈ {1, 2, 3, 4, 5}
  - 控制软标签的平滑程度
  - 更高温度产生更平滑的概率分布
  
- **加权比例 (Alpha)**: α ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}
  - 控制蒸馏损失与原始损失的平衡
  - α=0: 仅使用原始损失
  - α=1: 仅使用蒸馏损失
  
- **树深度 (Max Depth)**: D ∈ {4, 5, 6, 7, 8}
  - 控制学生模型的复杂度
  
- **Top-k特征**: k ∈ {5, 6, 7, 8}
  - 基于SHAP重要性选择的特征数量

---

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！

---

## 📄 许可证

MIT License

---

**更新日期**: 2025年9月12日  
**版本**: v2.0.0 - CatBoost教师模型升级版

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
