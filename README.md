# SHAP-Guided Knowledge Distillation for Credit Scoring

## 🎯 Project Overview

**基于SHAP特征重要性引导的知识蒸馏信用评分系统**

This project implements a comprehensive framework for **SHAP-guided knowledge distillation** in credit scoring applications. The system combines the interpretability of decision trees with the predictive power of deep neural networks through innovative knowledge distillation techniques, using SHAP (SHapley Additive exPlanations) for intelligent feature selection.

---

## 📁 Project Structure

```
Financial innovation/
├── data/                          # Dataset storage
│   ├── german_credit.csv         # German Credit Dataset (1,000 samples, 54 features)
│   ├── australian_credit.csv     # Australian Credit Dataset (690 samples, 22 features)
│   └── uci_credit.xls           # UCI Taiwan Credit Dataset (30,000 samples, 23 features)
├── results/                       # Generated output files
│   ├── model_comparison_*.xlsx    # Model performance comparison
│   ├── shap_feature_importance.png # SHAP feature visualization
│   ├── ablation_study_analysis_*.png # Ablation study plots
│   ├── topk_ablation_study_analysis_*.png # Top-k ablation analysis
│   ├── best_all_feature_rules_*.txt # Full feature decision rules
│   └── best_topk_rules_*.txt     # Top-k feature decision rules
├── main.py                       # Main execution pipeline
├── data_preprocessing.py         # Data loading and preprocessing
├── neural_models.py             # Neural network teacher models
├── distillation_module.py       # Knowledge distillation core
├── shap_analysis.py             # SHAP feature importance analysis
├── ablation_analyzer.py         # Ablation study visualization
├── result_manager.py            # Output management and reporting
└── README.md                    # This documentation
```

---

## 🧠 Teacher Model Architectures

### German Credit Dataset (1,000 samples, 54 features)
**Enhanced Residual Neural Network** - 优化的残差网络架构
- **Architecture**: Residual blocks with skip connections for improved gradient flow
- **Layers**:
  - Input: Linear(54 → 512) + BatchNorm + ReLU + Dropout(0.3)
  - Residual Block 1: [Linear(512 → 256) → BatchNorm → ReLU → Linear(256 → 256) → BatchNorm] + Skip(512 → 256)
  - Residual Block 2: [Linear(256 → 128) → BatchNorm → ReLU → Linear(128 → 128) → BatchNorm] + Skip(256 → 128)
  - Output: Linear(128 → 64) → BatchNorm → ReLU → Linear(64 → 32) → ReLU → Linear(32 → 1)
- **Loss Function**: BCEWithLogitsLoss with class balancing (pos_weight for imbalanced data)
- **Optimization**: AdamW (lr=0.0005, weight_decay=1e-3), ReduceLROnPlateau scheduler
- **Training**: 100 epochs (optimized), patience=30, batch_size=32
- **Target Accuracy**: 75%+ (improved from previous 62%)
- **Reference**: Residual Networks (ResNet) - He et al. (2016)

### Australian Credit Dataset (690 samples, 22 features)  
**Deep Feed-Forward Network** - 深度前馈网络
- **Architecture**: Sequential layers with batch normalization and dropout
- **Layers**: 
  - Linear(22 → 256) → BatchNorm → ReLU → Dropout(0.4)
  - Linear(256 → 128) → BatchNorm → ReLU → Dropout(0.35)
  - Linear(128 → 64) → BatchNorm → ReLU → Dropout(0.3)
  - Linear(64 → 32) → ReLU → Dropout(0.25)
  - Linear(32 → 1) → Sigmoid
- **Loss Function**: BCELoss (balanced dataset)
- **Optimization**: AdamW (lr=0.002, weight_decay=1e-3), ReduceLROnPlateau scheduler  
- **Training**: 100 epochs (optimized), patience=20, batch_size=64
- **Expected Accuracy**: 85%+
- **Reference**: Deep Neural Networks for Credit Scoring - Khandani et al. (2010)

### UCI Credit Default Dataset (30,000 samples, 23 features)
**Large-Scale Deep Network** - 大规模深度网络
- **Architecture**: Deep network optimized for large datasets
- **Layers**:
  - Linear(23 → 512) → BatchNorm → ReLU → Dropout(0.5)
  - Linear(512 → 256) → BatchNorm → ReLU → Dropout(0.45)
  - Linear(256 → 128) → BatchNorm → ReLU → Dropout(0.4)
  - Linear(128 → 64) → BatchNorm → ReLU → Dropout(0.35)
  - Linear(64 → 32) → ReLU → Dropout(0.3)
  - Linear(32 → 1) → Sigmoid
- **Loss Function**: BCELoss with focal loss characteristics for large-scale training
- **Optimization**: AdamW (lr=0.001, weight_decay=1e-4), ReduceLROnPlateau scheduler
- **Training**: 100 epochs (optimized), patience=25, batch_size=128  
- **Expected Accuracy**: 82%+
- **Reference**: Large-scale Credit Scoring - Lessmann et al. (2015)

---

## 📊 Four-Model Comparison Framework

本系统训练并对比以下四种模型：

### 1. Teacher Model (教师模型)
- **架构**: 数据集特定的PyTorch深度神经网络
- **特点**: 高预测准确性，复杂度高
- **目的**: 作为知识蒸馏的源模型

### 2. Baseline Decision Tree (基准决策树)
- **架构**: 标准scikit-learn DecisionTreeClassifier
- **特点**: 高可解释性，简单结构
- **目的**: 提供基准性能对比

### 3. All-Feature Distillation (全特征蒸馏)
- **架构**: 使用全部特征的知识蒸馏决策树
- **特点**: 平衡准确性和可解释性
- **目的**: 完整特征空间下的知识迁移

### 4. Top-k Feature Distillation (Top-k特征蒸馏)
- **架构**: 基于SHAP Top-k特征的知识蒸馏决策树
- **特点**: 精简特征集，高效解释
- **目的**: 最优特征子集下的知识迁移

---

## 🔬 Knowledge Distillation Process

### 核心技术参数
- **Temperature Scaling**: T ∈ {1, 2, 3, 4, 5} for soft label generation
- **Loss Combination**: α ∈ {0.0, 0.1, ..., 1.0} for balancing hard and soft losses
- **Dynamic Feature Selection**: 
  - German Dataset: k ∈ {5, 6, 7, ..., 54}
  - Australian Dataset: k ∈ {5, 6, 7, ..., 22}
  - UCI Dataset: k ∈ {5, 6, 7, ..., 23}
- **Tree Optimization**: Optuna-based hyperparameter tuning for decision trees
- **Decision Tree Depth**: max_depth ∈ {3, 4, 5, ..., 10}

### 蒸馏过程
1. **Teacher Training**: 训练数据集特定的深度神经网络
2. **SHAP Analysis**: 计算特征重要性并排序
3. **Knowledge Transfer**: 通过温度缩放软标签进行知识迁移
4. **Student Optimization**: 基于混合损失函数优化决策树学生模型
5. **Rule Extraction**: 从训练好的决策树中提取可解释规则

---
  - Linear(256 → 128) → BatchNorm → ReLU → Dropout(0.4)
  - Linear(128 → 64) → BatchNorm → ReLU → Dropout(0.35)
  - Linear(64 → 32) → ReLU → Dropout(0.3)
  - Linear(32 → 1) → Sigmoid
- **Loss Function**: BCELoss with focal loss characteristics for large-scale training
- **Optimization**: AdamW (lr=0.001, weight_decay=1e-4), ReduceLROnPlateau scheduler
- **Training**: 300 epochs, patience=25, batch_size=128  
- **Expected Accuracy**: 82%+

## 📊 Four-Model Comparison Framework

1. **Teacher Model**: Dataset-specific deep neural networks (architectures above)
2. **Baseline Decision Tree**: Standard scikit-learn DecisionTreeClassifier  
3. **All-Feature Distillation**: Knowledge distillation using complete feature set
4. **Top-k Feature Distillation**: SHAP-guided feature selection for targeted distillation

## 🔬 Knowledge Distillation Process

- **Temperature Scaling**: T ∈ {1, 2, 3, 4, 5} for soft label generation
- **Loss Combination**: α ∈ {0.0, 0.1, ..., 1.0} for balancing hard and soft losses
- **Feature Selection**: Dynamic k ranges (German: 5-54, Australian: 5-22, UCI: 5-23)
- **Tree Optimization**: Optuna-based hyperparameter tuning for decision trees

Financial innovation/

├── data/                          # Dataset storage- **Knowledge Distillation**: 将教师模型知识迁移到学生模型

│   ├── german_credit.csv          # German Credit Dataset

│   ├── australian_credit.csv      # Australian Credit Dataset### 🎯 Advanced Knowledge Distillation- **PyTorch Neural Networks**: 高性能深度学习教师模型

│   └── uci_credit.xls            # UCI Taiwan Credit Dataset

├── results/                       # Output files (generated)- **Temperature-scaled Soft Labels**: Configurable temperature parameter (T ∈ {1,2,3,4,5})- **Decision Tree**: 可解释性强的学生模型

│   ├── model_comparison_*.xlsx    # Performance comparison table

│   ├── shap_feature_importance.png # SHAP visualization- **Hybrid Loss Function**: Balanced combination of hard and soft label losses (α ∈ {0.0,0.1,...,1.0})

│   └── best_topk_rules_*.txt      # Extracted decision rules

├── trained_models/               # Saved models (generated)- **Multi-depth Decision Trees**: Adaptive tree depth optimization (3-10 levels)---

│   ├── teacher_model_*.pth       # PyTorch teacher models

│   ├── teacher_model_*.pkl       # Scikit-learn format

│   └── teacher_model_*.json      # Model metadata

├── main.py                       # Main execution pipeline### 📊 SHAP-Based Feature Selection  ## 系统架构

├── data_preprocessing.py         # Data loading and preprocessing

├── neural_models.py             # Neural network architectures- **Intelligent Feature Ranking**: TreeExplainer-based SHAP value computation

├── distillation_module.py       # Knowledge distillation implementation

├── shap_analysis.py             # SHAP feature importance analysis- **Top-k Selection**: Systematic evaluation of k ∈ {5,6,7,8} most important features```

├── result_manager.py            # Output management and reporting

├── teacher_model_saver.py       # Model serialization utilities- **Cross-Dataset Analysis**: Comparative feature importance across datasets├── data/                          # 数据集

└── README.md                    # This documentation

```│   ├── uci_credit.xls            # UCI信用卡数据集



## � SHAP Feature Analysis

### SHAP方法特点
- **TreeExplainer**: 针对决策树模型优化的SHAP解释器
- **全数据集分析**: 使用训练+验证+测试的完整数据集
- **精确特征排序**: 基于平均绝对SHAP值进行特征重要性排名
- **可视化输出**: 生成Top-20特征的对比图表

### 特征重要性可视化
- **数据集顺序**: German → Australian → UCI
- **颜色方案**: 浅蓝色系 → 浅绿色系 → 浅橙色系
- **特征数量**: 每个数据集显示Top-20重要特征
- **真实特征名**: 使用英文原始特征名而非编码名

---

## 🔧 Core Modules

### 1. Data Preprocessing (`data_preprocessing.py`)
- **功能**: 加载和预处理三个信用数据集
- **核心特性**:
  - 标准化的数据加载和train/validation/test划分
  - 分类变量的特征编码
  - 数据缩放和标准化
  - 特征名追踪以保证可解释性

### 2. Neural Network Models (`neural_models.py`)
- **功能**: 定义和训练教师神经网络
- **架构特点**:
  - 带残差连接的高级前馈网络
  - 批量标准化和dropout正则化
  - 自适应学习率调度
  - 早停和模型检查点

### 3. SHAP Analysis (`shap_analysis.py`)
- **功能**: 使用SHAP进行特征重要性分析
- **处理流程**:
  - 为每个数据集训练优化的决策树
  - 使用TreeExplainer计算SHAP值
  - 生成top-k特征排名
  - 创建带有正确特征名的可视化

### 4. Knowledge Distillation (`distillation_module.py`)
- **功能**: 从教师模型向学生模型转移知识
- **实现细节**:
  - 温度缩放的软标签生成
  - 混合损失函数(硬标签+软标签)
  - 基于SHAP的top-k特征选择
  - 从训练好的树中提取决策规则

### 5. Result Management (`result_manager.py`)
- **功能**: 组织和导出结果
- **输出内容**:
  - 基于Excel的性能对比
  - 决策规则文本文件
  - 模型性能指标

### 6. Ablation Analysis (`ablation_analyzer.py`)
- **功能**: 消融实验分析和可视化
- **输出图表**:
  - Top-k特征数量消融实验
  - 决策树深度消融实验
  - 1×2布局的简化图表

---

## 📈 Datasets

系统在三个广泛使用的信用评分基准数据集上进行评估：

### 1. German Credit Dataset (1,000 samples, 54 features)
- **来源**: UCI Machine Learning Repository
- **任务**: 二分类(好/坏信用风险)
- **特征**: 人口统计学、账户状态、信用历史

### 2. Australian Credit Approval Dataset (690 samples, 22 features)
- **来源**: UCI Machine Learning Repository  
- **任务**: 二分类(批准/拒绝信用)
- **特征**: 匿名化的申请人属性

### 3. Taiwan Credit Card Default Dataset (30,000 samples, 23 features)
- **来源**: UCI Machine Learning Repository
- **任务**: 二分类(违约/非违约)
- **特征**: 支付历史、账单金额、人口统计数据

---

## 🚀 Installation & Usage

### Prerequisites
```bash
pip install torch scikit-learn pandas numpy matplotlib seaborn shap openpyxl optuna tqdm
```

### Quick Start
```bash
# Clone the repository
git clone https://github.com/lidengjia1/shapGuided_KnowledgeDistilling.git
cd shapGuided_KnowledgeDistilling

# Run the complete pipeline
python main.py
```

### Expected Outputs
运行完成后，将生成以下核心文件：

1. **模型性能对比表** (`results/model_comparison_*.xlsx`)
   - 四种模型的性能指标
   - 准确率、F1分数、精确率、召回率
   - 每种配置的最佳超参数

2. **SHAP特征重要性图** (`results/shap_feature_importance.png`)
   - 三个数据集的可视化对比
   - 每个数据集的Top-20重要特征
   - 英文标签和正确的特征名

3. **Top-k决策规则** (`results/best_topk_rules_*.txt`)
   - 从最佳模型提取的决策树规则
   - 特征重要性排名
   - 模型性能详情

4. **消融实验图** (`results/*_ablation_study_analysis_*.png`)
   - Top-k特征数量消融实验
   - 决策树深度消融实验

---

## � Experimental Configuration

### 参数空间
- **Top-k特征数**: 
  - German Dataset: k ∈ {5, 6, ..., 54}
  - Australian Dataset: k ∈ {5, 6, ..., 22}
  - UCI Dataset: k ∈ {5, 6, ..., 23}
- **蒸馏温度**: T ∈ {1, 2, 3, 4, 5}
- **损失权重**: α ∈ {0.0, 0.1, 0.2, ..., 1.0}
- **树深度**: max_depth ∈ {3, 4, 5, 6, 7, 8, 9, 10}

### 评估指标
- **准确率 (Accuracy)**: 正确预测的比例
- **F1分数 (F1-Score)**: 精确率和召回率的调和平均
- **精确率 (Precision)**: 正预测中的正确比例
- **召回率 (Recall)**: 实际正例中的预测正确比例

### 并发优化
- **Windows平台**: 使用min(4, cpu_count//2)个并发进程
- **Linux/Mac平台**: 使用min(cpu_count-1, cpu_count)个并发进程
- **进度显示**: 集成tqdm进度条，实时显示训练进度

---



- **v2.0**: Complete refactoring with SHAP-guided distillation# Run complete analysis pipeline- **Accuracy**: 分类准确率

- **v1.9**: Enhanced neural network architectures

- **v1.8**: Improved feature name handling and visualizationpython main.py- **Precision**: 精确率

- **v1.7**: Added comprehensive result management

- **v1.6**: Optimized knowledge distillation pipeline```- **Recall**: 召回率



---- **F1-Score**: F1分数



*This project represents cutting-edge research in explainable AI for financial applications, combining the power of deep learning with the interpretability requirements of financial decision-making.*This will generate three key outputs:- **AUC**: ROC曲线下面积



1. **Model Comparison Table** (`results/model_comparison_*.xlsx`)---

   - Performance metrics for all four model types

   - Statistical significance tests## 环境配置

   - Hyperparameter configurations

### 依赖安装

2. **SHAP Feature Importance Visualization** (`results/shap_feature_importance.png`)```bash

   - Top-8 features for each datasetpip install torch pandas scikit-learn xgboost shap matplotlib openpyxl numpy

   - Comparative importance scores```

   - Cross-dataset feature analysis

### 运行系统

3. **Top-k Decision Rules** (`results/best_topk_rules_*.txt`)```bash

   - Interpretable IF-THEN rules from best distilled modelspython main.py

## 📈 Key Findings

### 主要实验结果
- **Top-k特征蒸馏**达到与全特征模型相当的准确率
- **SHAP引导的特征选择**显著提升了模型可解释性  
- **知识蒸馏**有效缩小了准确率与可解释性之间的差距
- **温度缩放和损失加权**是有效蒸馏的关键技术

### 性能基准测试

| Dataset | Teacher (DNN) | Baseline Tree | All-Feature Distill | Top-k Distill |
|---------|---------------|---------------|-------------------|---------------|
| German | 0.75-0.78 | 0.70-0.73 | 0.73-0.76 | 0.74-0.77 |
| Australian | 0.85-0.88 | 0.82-0.85 | 0.84-0.87 | 0.85-0.88 |
| UCI Taiwan | 0.80-0.83 | 0.76-0.79 | 0.78-0.81 | 0.79-0.82 |

*注：范围反映不同超参数配置下的性能变化*

---

## 📚 Technical References

本项目基于以下前沿研究成果：

### 知识蒸馏相关
- **Neural Network Distillation**: Hinton et al. (2015) - 温度缩放和软标签训练
- **Tabular Data Distillation**: 针对表格数据的知识蒸馏优化

### SHAP可解释AI
- **SHAP Values**: Lundberg & Lee (2017) - TreeExplainer精确特征重要性计算
- **Feature Selection**: 基于SHAP的智能特征选择策略

### 神经架构设计
- **Residual Networks**: He et al. (2016) - 残差连接改善梯度流
- **Credit Scoring DNNs**: 针对信用评分优化的神经网络架构

### 金融机器学习
- **Financial ML**: Lopez de Prado (2018) - 金融风险评估和可解释建模
- **Regulatory Compliance**: 符合金融监管要求的可解释AI方法

---

## 🔄 Version History

### v2.0.0 (Current) - Enhanced Performance
- ✅ 优化教师模型架构，提升German数据集准确率至75%+
- ✅ 减少训练epochs，提高训练效率
- ✅ 简化消融实验图表为1×2布局
- ✅ 改进SHAP可视化配色方案
- ✅ 禁用文件自动清理功能
- ✅ 增强Windows平台并发支持

### v1.0.0 - Initial Release
- ✅ 基础知识蒸馏框架
- ✅ SHAP特征重要性分析
- ✅ 三数据集支持
- ✅ 基础可视化功能

---

## 📧 Contact Information

**Primary Author**: Li Dengjia  
**Email**: lidengjia@hnu.edu.cn  
**Institution**: Hunan University  
**Research Focus**: Financial AI, Knowledge Distillation, Explainable Machine Learning

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Acknowledgments

- UCI Machine Learning Repository for providing the benchmark datasets
- SHAP library developers for interpretability tools
- PyTorch team for the deep learning framework  
- Research community for advances in knowledge distillation and explainable AI

---

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@misc{li2025shap_distillation,
  title={SHAP-Guided Knowledge Distillation for Credit Scoring},
  author={Li, Dengjia and [Co-authors]},
  year={2025},
  institution={Hunan University},
  note={A comprehensive framework for interpretable credit scoring using SHAP-guided knowledge distillation}
}
```

---

*This project represents ongoing research in interpretable machine learning for financial applications. Contributions and collaborations are welcome.*

**Last Updated**: September 16, 2025  
**Version**: v2.0.0
