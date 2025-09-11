"# 🏦 SHAP-Guided Knowledge Distillation for Credit Scoring

**基于SHAP特征重要性引导的知识蒸馏信用评分系统**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-green.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-0.42+-orange.svg)](https://shap.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/Optuna-3.4+-purple.svg)](https://optuna.org/)

## 📋 项目概述

本项目实现了一个创新的信用评分模型优化系统，通过结合**SHAP特征重要性分析**和**知识蒸馏技术**，将复杂的神经网络教师模型的知识转移到可解释的决策树学生模型中，在保持高预测性能的同时显著提升模型的可解释性。

### 🎯 核心创新点

- **🧠 多架构神经网络**: 针对不同数据集设计专门的神经网络架构
- **🔍 SHAP特征重要性**: 基于SHAP值进行特征选择和重要性排序
- **🎓 知识蒸馏**: 从复杂神经网络向可解释决策树传递知识
- **⚡ 智能优化**: 集成Optuna进行自动超参数优化
- **📊 全面评估**: 多维度性能分析和可视化

## 📁 项目结构

```
📦 shapGuided_KnowledgeDistilling/
├── 📊 data/                          # 数据集目录
│   ├── uci_credit.xls                # UCI信用数据集
│   ├── german_credit.csv             # German信用数据集
│   └── australian_credit.csv         # Australian信用数据集
├── 🧠 models/                        # 训练好的模型存储
├── 📈 results/                       # 实验结果
├── 📊 visualization/                 # 可视化图表
├── 🔧 data_preprocessing.py          # 数据预处理模块
├── 🧠 neural_models.py               # 神经网络教师模型
├── 🔍 shap_analysis.py               # SHAP特征重要性分析
├── 🎓 distillation_module.py         # 知识蒸馏核心模块
├── 📊 experiment_manager.py          # 实验管理和结果分析
├── 🌳 tree_rules_analyzer.py         # 决策树规则提取
├── 🚀 main.py                        # 主程序入口
└── 📖 README.md                      # 项目文档
```

## 🏗️ 系统架构

### 1. 数据预处理层
- **标准化处理**: Z-score标准化确保特征尺度一致
- **编码转换**: 分类变量自动编码处理
- **数据分割**: 训练/验证/测试集智能分割

### 2. 教师模型层
```python
# UCI数据集 - 多层感知机
MLP_UCI: Input → 64 → 32 → 1 → Sigmoid

# German数据集 - 径向基函数网络  
RBF_German: Input → 30 RBF Centers → Linear → Sigmoid

# Australian数据集 - 自编码器增强MLP
AE_MLP_Australian: Input → Encoder(16→8) → Decoder → Classifier → Sigmoid
```

### 3. 特征重要性分析
- **SHAP值计算**: 基于Shapley值的特征贡献分析
- **特征排序**: 按重要性对特征进行排序
- **Top-K选择**: 自适应选择最重要的K个特征

### 4. 知识蒸馏层
- **温度缩放**: 软标签概率调节 (T ∈ [1,5])
- **损失函数**: 蒸馏损失 + 硬标签损失
- **权重平衡**: α ∈ [0.1, 0.9] 控制损失权重

### 5. 学生模型优化
- **决策树**: 可解释的树状结构
- **深度控制**: max_depth ∈ [4,8]
- **Optuna优化**: 自动搜索最优超参数

## ⚙️ 核心算法

### 知识蒸馏损失函数

```python
L_total = α × L_distillation + (1-α) × L_hard

L_distillation = KL_divergence(
    softmax(logits_student / T), 
    softmax(logits_teacher / T)
) × T²

L_hard = CrossEntropy(logits_student, true_labels)
```

### SHAP特征选择算法

```python
def shap_feature_selection(model, X_train, top_k):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    importance = np.mean(np.abs(shap_values), axis=0)
    return np.argsort(importance)[-top_k:]
```

## 🚀 快速开始

### 环境要求

```bash
Python >= 3.8
torch >= 2.0.0
scikit-learn >= 1.3.0
shap >= 0.42.0
optuna >= 3.4.0
pandas >= 1.5.0
numpy >= 1.24.0
matplotlib >= 3.6.0
seaborn >= 0.12.0
tqdm >= 4.64.0
```

### 安装依赖

```bash
pip install torch scikit-learn shap optuna pandas numpy matplotlib seaborn tqdm openpyxl
```

### 运行实验

```bash
python main.py
```

## 📊 实验配置

### 超参数设置

| 参数 | 范围 | 说明 |
|------|------|------|
| `top_k` | [5, 10, 15, 20] | SHAP特征选择数量 |
| `temperature` | [1, 2, 3, 4, 5] | 知识蒸馏温度参数 |
| `alpha` | [0.1, 0.2, ..., 0.9] | 损失权重平衡参数 |
| `max_depth` | [4, 5, 6, 7, 8] | 决策树最大深度 |

### Optuna优化参数

- `min_samples_split`: [2, 20]
- `min_samples_leaf`: [1, 10] 
- `max_features`: ['sqrt', 'log2', None]
- `criterion`: ['gini', 'entropy']

## 📈 性能评估

### 评估指标

- **准确率 (Accuracy)**: 主要评判标准
- **F1分数**: 平衡精确率和召回率
- **精确率 (Precision)**: 正例预测准确性
- **召回率 (Recall)**: 正例覆盖完整性

### 输出文件

| 文件类型 | 描述 |
|----------|------|
| `📊 comparison_results.xlsx` | 四种模型性能对比 |
| `📋 master_results.xlsx` | 完整实验结果记录 |
| `📈 performance_analysis.png` | 性能分析图表 |
| `🔍 topk_parameter_analysis.png` | Top-K参数影响分析 |
| `🌳 decision_tree_rules.xlsx` | 最优决策树规则 |
| `📄 experiment_summary.txt` | 实验总结报告 |

## 🔬 技术细节

### 数据集特点

| 数据集 | 样本数 | 特征数 | 不平衡比例 | 神经网络架构 |
|--------|--------|--------|------------|--------------|
| UCI Credit | 30,000 | 23 | 22:78 | MLP (64→32→1) |
| German Credit | 1,000 | 20 | 70:30 | RBF (30 centers) |
| Australian Credit | 690 | 14 | 44:56 | AutoEncoder-MLP |

### 算法复杂度

- **SHAP分析**: O(n × m × d) - n样本, m特征, d深度
- **知识蒸馏**: O(epochs × batch_size × parameters)
- **决策树训练**: O(n × log(n) × m)

## 📊 实验结果示例

```
🏆 最佳模型性能 (UCI数据集):
   模型类型: Top-K知识蒸馏
   准确率: 0.8234
   F1分数: 0.7891
   特征数: 15 (原始23个)
   决策树深度: 6
   温度参数: 3
   权重参数: 0.7
```

## 🔧 自定义配置

### 修改神经网络架构

```python
# 在 neural_models.py 中修改
class Custom_Model(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),  # 自定义层数
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(), 
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
```

### 调整蒸馏参数

```python
# 在 main.py 中修改实验配置
config = {
    'top_k_values': [8, 12, 16, 24],      # 自定义K值
    'temperature_values': [2, 4, 6],       # 自定义温度
    'alpha_values': [0.3, 0.5, 0.7],      # 自定义权重
    'max_depth_values': [5, 7, 9]         # 自定义深度
}
```

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目基于 MIT 许可证开源 - 详见 [LICENSE](LICENSE) 文件

## 📞 联系方式

- **项目作者**: [Your Name]
- **邮箱**: [your.email@example.com]
- **GitHub**: [https://github.com/yourusername/shapGuided_KnowledgeDistilling](https://github.com/yourusername/shapGuided_KnowledgeDistilling)

## 🙏 致谢

- [SHAP](https://github.com/slundberg/shap) - 用于模型解释性分析
- [Optuna](https://github.com/optuna/optuna) - 用于超参数优化
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [Scikit-learn](https://scikit-learn.org/) - 机器学习工具包

## 📚 参考文献

1. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in neural information processing systems*, 30.

2. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. *arXiv preprint arXiv:1503.02531*.

3. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining*.

---

⭐ **如果这个项目对您有帮助，请给我们一个星标！**" 
