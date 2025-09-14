# SHAP-Guided Knowledge Distillation for Financial Credit Assessment# SHAP-Guided Knowledge Distillation for Credit Scoring



## 🎯 Project Overview# SHAP-Guided Knowledge Distillation for Credit Scoring



This project implements a novel SHAP-guided knowledge distillation framework for financial credit assessment. The system uses SHAP (SHapley Additive exPlanations) feature importance analysis to guide the knowledge transfer from complex neural network teachers to interpretable decision tree students.## Overview



**Key Innovation**: Instead of using all features, our approach identifies the most important features through SHAP analysis and trains specialized distillation models using only top-k features, achieving better interpretability while maintaining competitive performance.**基于SHAP特征重要性引导的知识蒸馏信用评分系统**



## 📊 Research MethodologyThis project implements a comprehensive framework for **SHAP-guided knowledge distillation** in credit scoring applications. The system combines the interpretability of decision trees with the predictive power of deep neural networks through innovative knowledge distillation techniques, using SHAP (SHapley Additive exPlanations) for intelligent feature selection.



### Four-Model Comparison Framework---

1. **Teacher Model**: Deep Neural Network (PyTorch-based) with advanced architectures

2. **Baseline Decision Tree**: Standard decision tree trained on original data## Key Features

3. **All-Feature Distillation**: Decision tree trained with knowledge distillation using all features

4. **Top-k Feature Distillation**: Decision tree trained with knowledge distillation using only SHAP-selected top-k features## 项目简介



### Knowledge Distillation Process### 🧠 Four-Model Comparison Framework

- **Temperature Scaling**: T ∈ {1, 2, 3, 4, 5} for soft label generation

- **Loss Combination**: α ∈ {0.0, 0.1, ..., 1.0} for balancing hard and soft losses- **Teacher Model**: Deep Neural Network (PyTorch-based)本系统实现基于SHAP特征重要性分析和知识蒸馏技术的信用评分模型，将深度神经网络的预测性能与决策树的可解释性相结合。

- **Feature Selection**: k ∈ {5, 6, 7, 8} top features selected via SHAP importance

- **Tree Optimization**: Optuna-based hyperparameter tuning for decision trees- **Baseline Decision Tree**: Standard scikit-learn DecisionTreeClassifier  



## 📁 Directory Structure- **All-Feature Distillation**: Knowledge distillation using complete feature set### 核心技术



```- **Top-k Feature Distillation**: SHAP-guided feature selection for targeted distillation- **SHAP (SHapley Additive exPlanations)**: 量化特征重要性，实现特征选择

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



## 🔧 Core Modules## Datasets│   ├── german_credit.csv         # German信用数据集



### 1. Data Preprocessing (`data_preprocessing.py`)│   └── australian_credit.csv     # Australian信用数据集

- **Purpose**: Load and preprocess three credit datasets

- **Key Features**:The framework is evaluated on three well-established credit scoring benchmarks:├── trained_models/                # 训练模型存储

  - Standardized data loading with train/validation/test splits

  - Feature encoding for categorical variables├── results/                       # 实验结果输出

  - Data scaling and normalization

  - Feature name tracking for interpretability1. **German Credit Dataset** (1,000 samples, 20 features)├── data_preprocessing.py          # 数据预处理



### 2. Neural Network Models (`neural_models.py`)   - Source: UCI Machine Learning Repository├── neural_models.py              # PyTorch教师模型

- **Purpose**: Define and train teacher neural networks

- **Architectures**:   - Task: Binary classification (good/bad credit risk)├── shap_analysis.py              # SHAP特征分析

  - Advanced feedforward networks with residual connections

  - Batch normalization and dropout for regularization   - Features: Demographics, account status, credit history├── distillation_module.py        # 知识蒸馏核心模块

  - Adaptive learning rate scheduling

  - Early stopping and model checkpointing├── result_manager.py             # 结果管理器



### 3. SHAP Analysis (`shap_analysis.py`)2. **Australian Credit Approval Dataset** (690 samples, 14 features) ├── teacher_model_saver.py        # 模型保存器

- **Purpose**: Feature importance analysis using SHAP

- **Process**:   - Source: UCI Machine Learning Repository└── main.py                       # 主程序

  - Train optimized decision trees for each dataset

  - Compute SHAP values using TreeExplainer   - Task: Binary classification (approve/deny credit)```

  - Generate top-k feature rankings

  - Create visualization with proper feature names   - Features: Anonymized applicant attributes



### 4. Knowledge Distillation (`distillation_module.py`)---

- **Purpose**: Transfer knowledge from teachers to students

- **Implementation**:3. **Taiwan Credit Card Default Dataset** (30,000 samples, 23 features)

  - Temperature-scaled soft label generation

  - Combined loss function (hard + soft)   - Source: UCI Machine Learning Repository  ## 四种模型对比

  - Top-k feature selection based on SHAP

  - Decision rule extraction from trained trees   - Task: Binary classification (default/non-default)



### 5. Result Management (`result_manager.py`)   - Features: Payment history, bill amounts, demographic data本系统训练并对比以下四种模型：

- **Purpose**: Organize and export results

- **Outputs**:

  - Excel-based performance comparison

  - Decision rule text files## Architecture### 1. Teacher Model (教师模型)

  - Model performance metrics

- **架构**: PyTorch深度神经网络

## 🚀 Usage Instructions

### Neural Network Design- **特点**: 高预测精度，复杂度高

### Prerequisites

```bashBased on recent advances in deep learning for credit scoring ([arXiv:2502.00201](https://arxiv.org/abs/2502.00201), [arXiv:2411.17783](https://arxiv.org/abs/2411.17783)), our teacher models employ:- **用途**: 作为知识源指导学生模型

pip install torch scikit-learn pandas numpy matplotlib seaborn shap openpyxl optuna

```



### Quick Start- **Multi-layer Perceptrons**: 3-layer architecture with adaptive hidden dimensions### 2. Baseline Decision Tree (基准决策树)

```bash

python main.py- **Regularization**: Dropout layers and batch normalization for robustness- **架构**: 标准决策树 (max_depth=5)

```

- **Activation Functions**: ReLU activations with output sigmoid for binary classification- **特点**: 可解释性强，性能一般

### Expected Outputs

After execution, three core files will be generated:- **Optimization**: Adam optimizer with learning rate scheduling- **用途**: 对比基准

1. **Model Comparison Table** (`results/model_comparison_*.xlsx`)

   - Performance metrics for all four models

   - Accuracy, F1-score, precision, recall

   - Best hyperparameters for each configuration### Knowledge Distillation Process### 3. All Feature Distillation (全特征蒸馏)



2. **SHAP Feature Importance** (`results/shap_feature_importance.png`)1. **Teacher Training**: Deep neural network trained on full dataset- **架构**: 决策树 + 知识蒸馏

   - Visual comparison across three datasets

   - Top-8 most important features for each dataset2. **SHAP Analysis**: Feature importance computation using TreeExplainer- **特征**: 使用全部特征

   - English labels with proper feature names

3. **Feature Selection**: Top-k feature identification for distillation- **目标**: 在保持可解释性的同时提升性能

3. **Top-k Decision Rules** (`results/best_topk_rules_*.txt`)

   - Extracted decision tree rules from best models4. **Student Training**: Decision tree learning from teacher's soft predictions

   - Feature importance rankings

   - Model performance details5. **Evaluation**: Comprehensive performance comparison across all models### 4. Top-k Feature Distillation (Top-k特征蒸馏)



## 📈 Experimental Results- **架构**: 决策树 + 知识蒸馏 + SHAP特征选择



### Datasets Used## Installation- **特征**: 仅使用SHAP选出的Top-k重要特征

- **German Credit Dataset**: 1,000 samples, 20 features

- **Australian Credit Dataset**: 690 samples, 14 features  - **目标**: 在精简特征下获得最优性能

- **UCI Taiwan Credit Dataset**: 30,000 samples, 23 features

```bash

### Key Findings

- Top-k feature distillation achieves comparable accuracy to full-feature models# Clone the repository---

- SHAP-guided feature selection improves interpretability significantly

- Knowledge distillation bridges the gap between accuracy and explainabilitygit clone https://github.com/lidengjia1/shapGuided_KnowledgeDistilling.git

- Temperature scaling and loss weighting are crucial for effective distillation

cd shapGuided_KnowledgeDistilling## 实验配置

## 📧 Contact Information



**Author**: Li Dengjia  

**Email**: lidengjia@hnu.edu.cn  # Install dependencies### 参数空间

**Institution**: Hunan University  

**Research Focus**: Financial AI, Knowledge Distillation, Explainable Machine Learningpip install torch scikit-learn shap matplotlib seaborn pandas numpy openpyxl- **Top-k特征数**: k ∈ {5, 6, 7, 8}



## 📚 Technical References```- **蒸馏温度**: T ∈ {1, 2, 3, 4, 5}



This implementation incorporates recent advances in:- **损失权重**: α ∈ {0.0, 0.1, 0.2, ..., 1.0}

- **Knowledge Distillation**: Temperature scaling and soft label training

- **SHAP Analysis**: TreeExplainer for precise feature importance## Quick Start- **树深度**: depth ∈ {4, 5, 6, 7, 8}

- **Neural Architecture**: Residual connections and advanced optimization

- **Financial ML**: Credit risk assessment and interpretable modeling



## 🔄 Version History```bash### 评估指标



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

   - Feature thresholds and confidence scores```

   - Optimal k-values and hyperparameters

---

## Research Applications

## 输出结果

### Knowledge Distillation Research

- Novel application of SHAP-guided feature selection in teacher-student frameworks系统执行完成后生成三个核心文件：

- Systematic evaluation of soft label temperature effects in tabular data

- Hybrid loss function optimization for interpretable model distillation### 1. 模型性能对比表 (`model_comparison_*.xlsx`)

包含四种模型在三个数据集上的性能指标对比

### Credit Scoring Innovation  

- Explainable AI for financial decision-making### 2. SHAP特征重要性图 (`shap_feature_importance.png`)

- Regulatory compliance through interpretable decision trees可视化各数据集Top-10重要特征的SHAP值排序

- Feature importance analysis for risk factor identification

### 3. 最优决策规则 (`best_topk_rules_*.txt`)

### Model Interpretability每个数据集上性能最优的Top-k蒸馏模型的决策规则和配置参数

- SHAP-based feature ranking across diverse credit datasets

- Decision tree rule extraction for transparent predictions---

- Comparative analysis of feature importance patterns

## 技术实现

## Performance Benchmarks

### 神经网络架构

Based on extensive evaluation across three datasets:根据数据集复杂度自适应调整网络结构：



| Dataset | Teacher (DNN) | Baseline Tree | All-Feature Distill | Top-k Distill |**German数据集**: 64→32→16→1 (3层)

|---------|---------------|---------------|-------------------|---------------|**Australian数据集**: 128→64→32→1 (3层)  

| German | 0.75-0.78 | 0.70-0.73 | 0.73-0.76 | 0.74-0.77 |**UCI数据集**: 256→128→64→32→1 (4层)

| Australian | 0.85-0.88 | 0.82-0.85 | 0.84-0.87 | 0.85-0.88 |

| UCI Taiwan | 0.80-0.83 | 0.76-0.79 | 0.78-0.81 | 0.79-0.82 |### 知识蒸馏损失函数

```

*Note: Ranges reflect performance variation across different hyperparameter configurations*L_total = α × L_distillation + (1-α) × L_hard

L_distillation = KL_divergence(softmax(y_student/T), softmax(y_teacher/T))

## Technical Implementation```



### Core Modules### SHAP特征选择

- `neural_models.py`: PyTorch teacher network implementation使用TreeSHAP算法计算特征重要性，选择Top-k个最重要特征进行蒸馏训练。

- `distillation_module.py`: Knowledge distillation algorithms  

- `shap_analysis.py`: SHAP feature importance computation---

- `data_preprocessing.py`: Dataset loading and preprocessing

- `result_manager.py`: Output generation and analysis## 实验流程



### Key Algorithms1. **数据预处理**: 标准化、编码、6:2:2划分

- **SHAP TreeExplainer**: Efficient SHAP value computation for tree models2. **教师模型训练**: PyTorch神经网络训练

- **Temperature Scaling**: Softmax temperature adjustment for knowledge transfer3. **SHAP分析**: 计算特征重要性排序

- **Hybrid Loss Optimization**: Weighted combination of classification and distillation losses4. **基准模型训练**: 标准决策树

- **Grid Search Optimization**: Systematic hyperparameter exploration5. **知识蒸馏**: 全特征和Top-k特征两种策略

6. **性能评估**: 多指标对比分析

## Citation

---

If you use this work in your research, please cite:

## 引用信息

```bibtex

@article{li2024shap_distillation,```bibtex

  title={SHAP-Guided Knowledge Distillation for Interpretable Credit Scoring},@misc{shap_knowledge_distillation_2025,

  author={Li, Dengjia and [Co-authors]},  title={SHAP-Guided Knowledge Distillation for Credit Scoring},

  journal={[Journal Name]},  author={[Author Names]},

  year={2024},  year={2025},

  publisher={[Publisher]}  institution={[Institution]},

}  note={基于SHAP特征重要性引导的知识蒸馏信用评分系统}

```}

```

## Contact

## 🧠 模型架构

**Primary Author**: Dengjia Li  

**Email**: lidengjia@hnu.edu.cn  ### PyTorch深度神经网络教师模型

**Institution**: Hunan University  

**Department**: [Department Name]针对不同数据集优化的网络结构：



## License#### German信用数据集

```python

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.nn.Sequential(

    nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.3),

## Acknowledgments    nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),

    nn.Linear(32, 16), nn.ReLU(),

- UCI Machine Learning Repository for providing the datasets    nn.Linear(16, 1), nn.Sigmoid()

- SHAP library developers for interpretability tools)

- PyTorch team for deep learning framework```

- Research community for advances in knowledge distillation and explainable AI

#### Australian信用数据集

## Related Work```python

nn.Sequential(

### Recent Advances in Credit Scoring    nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.4),

- [Deep Neural Networks for Credit Scoring](https://arxiv.org/abs/2502.00201) (2025)    nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),

- [Kolmogorov-Arnold Networks for Credit Default Prediction](https://arxiv.org/abs/2411.17783) (2024)    nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),

- [Attention-based Graph Neural Networks for Loan Default](https://arxiv.org/abs/2402.00299) (2024)    nn.Linear(32, 1), nn.Sigmoid()

)

### Knowledge Distillation Research```

- [Neural Network Distillation for Tabular Data](https://arxiv.org/abs/2412.02097) (2024)

- [Monotonic Neural Models for Credit Scoring](https://arxiv.org/abs/2209.10070) (2022)#### UCI信用卡数据集

```python

### Explainable AI in Finance  nn.Sequential(

- [SHAP Applications in Financial ML](https://arxiv.org/abs/2209.09362) (2022)    nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.5),

- [Trustworthy Credit Scoring with Interpretable Models](https://arxiv.org/abs/2301.08839) (2023)    nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4),

    nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),

---    nn.Linear(64, 32), nn.ReLU(),

    nn.Linear(32, 1), nn.Sigmoid()

*This project represents ongoing research in interpretable machine learning for financial applications. Contributions and collaborations are welcome.*)
```

---

## 🔬 实验参数

- **Top-k特征**: k ∈ {5, 6, 7, 8}
- **知识蒸馏温度**: T ∈ {1, 2, 3, 4, 5}
- **加权比例**: α ∈ {0.0, 0.1, 0.2, ..., 1.0}
- **决策树深度**: depth ∈ {4, 5, 6, 7, 8}

---

## 📈 评价指标

- **Accuracy**: 准确率
- **Precision**: 精确率  
- **Recall**: 召回率
- **F1-Score**: F1分数
- **AUC**: ROC曲线下面积

---

## 🎯 核心优势

1. **高性能**: PyTorch深度神经网络提供优异的预测性能
2. **可解释性**: 通过知识蒸馏获得可解释的决策树模型
3. **特征选择**: SHAP分析识别最重要的特征
4. **精简输出**: 只生成三个核心结果文件，避免信息冗余

---

## 📄 引用

如果使用本项目，请引用：

```bibtex
@misc{shap_knowledge_distillation_2024,
  title={SHAP-Guided Knowledge Distillation for Credit Scoring},
  author={[作者姓名]},
  year={2024},
  note={基于SHAP特征重要性引导的知识蒸馏信用评分系统}
}
```
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
