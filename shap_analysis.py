"""
SHAP特征重要性分析模块
SHAP Feature Importance Analysis Module
"""

import numpy as np
# 设置matplotlib后端为非交互式，避免多线程问题
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
import optuna
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')

# 禁用Optuna日志输出
optuna.logging.set_verbosity(optuna.logging.WARNING)

plt.style.use('default')

class SHAPAnalyzer:
    """SHAP特征重要性分析器 - 基于决策树模型"""
    
    def __init__(self, processed_data):
        self.processed_data = processed_data
        self.decision_tree_models = {}
        
    def train_decision_trees(self):
        """为每个数据集训练决策树模型用于SHAP分析"""
        print("🌳 Training decision trees for SHAP analysis...")
        
        for dataset_name, data_dict in self.processed_data.items():
            print(f"   Training decision tree for {dataset_name}...")
            
            X_train = data_dict['X_train']
            X_test = data_dict['X_test']
            y_train = data_dict['y_train']
            y_test = data_dict['y_test']
            
            # 使用Optuna优化决策树参数
            def objective(trial):
                # 定义超参数搜索空间
                max_depth = trial.suggest_int('max_depth', 5, 25)
                min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
                min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
                
                # 创建决策树模型
                dt = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42
                )
                
                # 使用交叉验证评估模型
                scores = cross_val_score(dt, X_train, y_train, cv=5, scoring='accuracy', n_jobs=1)
                return scores.mean()
            
            # 创建Optuna study并优化
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=50, show_progress_bar=False)
            
            # 使用最佳参数训练最终模型
            best_params = study.best_params
            best_model = DecisionTreeClassifier(
                max_depth=best_params['max_depth'],
                min_samples_split=best_params['min_samples_split'],
                min_samples_leaf=best_params['min_samples_leaf'],
                random_state=42
            )
            best_model.fit(X_train, y_train)
            
            # 计算测试集准确率
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.decision_tree_models[dataset_name] = {
                'model': best_model,
                'accuracy': accuracy,
                'best_params': best_params,
                'best_score': study.best_value
            }
            
            print(f"     Decision tree accuracy: {accuracy:.4f}")
            print(f"     Best params: {best_params}")
            print(f"     CV score: {study.best_value:.4f}")
        
        print("✅ Decision trees trained for SHAP analysis")
        
    def compute_shap_values(self, dataset_name, top_k_range=(5, 8)):
        """使用决策树模型计算SHAP值并选择重要特征
        
        SHAP计算方法说明：
        1. 使用训练好的决策树模型进行SHAP分析
        2. 使用SHAP TreeExplainer专门针对树模型优化
        3. 对全量数据样本(训练+测试)计算SHAP值以获得更准确的特征重要性
        4. SHAP值表示每个特征对模型预测的贡献度
        5. 通过平均绝对SHAP值计算特征重要性排序
        """
        print(f"\n🔍 Computing SHAP values for {dataset_name.upper()} dataset...")
        print(f"   Method: TreeExplainer with decision tree model")
        print(f"   Computing SHAP for all data samples (train + test)")
        
        model_info = self.decision_tree_models[dataset_name]
        model = model_info['model']
        data_dict = self.processed_data[dataset_name]
        
        # 准备数据 - 使用全量样本(训练+测试)
        X_train = data_dict['X_train']
        X_test = data_dict['X_test']
        
        # 合并训练和测试数据以获得更全面的SHAP分析
        import numpy as np
        X_all = np.vstack([X_train, X_test])
        
        print(f"   Data samples: {X_train.shape[0]} train + {X_test.shape[0]} test = {X_all.shape[0]} total")
        
        # 创建SHAP TreeExplainer
        explainer = shap.TreeExplainer(model)
        
        # 计算SHAP值
        print(f"   Calculating SHAP values for {X_all.shape[0]} samples...")
        shap_values = explainer.shap_values(X_all)
        
        # 处理不同格式的SHAP输出
        if isinstance(shap_values, list):
            # 二分类问题，通常取第二个类别（正类）
            if len(shap_values) == 2:
                shap_values = shap_values[1]
            else:
                shap_values = shap_values[0]
        
        # 计算特征重要性（平均绝对SHAP值）
        feature_importance = np.mean(np.abs(shap_values), axis=0)
        
        # 确保feature_importance是一维数组并转换为float
        if feature_importance.ndim > 1:
            feature_importance = feature_importance.flatten()
        feature_importance = feature_importance.astype(float)
        
        # 创建特征重要性字典
        feature_names = data_dict['feature_names']
        importance_dict = dict(zip(feature_names, feature_importance))
        
        # 按重要性排序
        sorted_features = sorted(importance_dict.items(), key=lambda x: float(x[1]), reverse=True)
        
        print(f"   Top 8 important features for {dataset_name}:")
        for i, (feature, importance) in enumerate(sorted_features[:8]):
            print(f"     {i+1}. {feature}: {float(importance):.4f}")
        
        # 生成不同top-k的特征选择
        top_k_features = {}
        for k in range(top_k_range[0], top_k_range[1] + 1):
            top_k_features[k] = [feat[0] for feat in sorted_features[:k]]
        
        return {
            'shap_values': shap_values,
            'feature_importance': feature_importance,
            'sorted_features': sorted_features,
            'top_k_features': top_k_features,
            'explainer': explainer,
            'feature_names': feature_names
        }
    
    def create_combined_shap_visualization(self, all_shap_results):
        """创建三个数据集的SHAP对比可视化图表"""
        print(f"📊 Creating combined SHAP visualization...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        datasets = ['german', 'uci', 'australian']
        titles = ['German Credit Dataset', 'UCI Credit Dataset', 'Australian Credit Dataset']
        
        for idx, (dataset_name, title) in enumerate(zip(datasets, titles)):
            ax = axes[idx]
            shap_results = all_shap_results[dataset_name]
            
            # 获取Top 8特征
            top_features = shap_results['sorted_features'][:8]
            features, importances = zip(*top_features)
            importances = [float(x) for x in importances]
            
            # 创建条形图
            bars = ax.barh(range(len(features)), importances, color=f'C{idx}')
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features, fontsize=8)
            ax.set_xlabel('Mean |SHAP Value|', fontsize=10)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.invert_yaxis()
            
            # 添加数值标签
            for i, (bar, imp) in enumerate(zip(bars, importances)):
                ax.text(bar.get_width() + max(importances)*0.01, bar.get_y() + bar.get_height()/2, 
                       f'{imp:.3f}', va='center', fontsize=7)
        
        plt.tight_layout()
        plt.savefig('results/combined_shap_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Combined SHAP visualization saved to: results/combined_shap_analysis.png")
        
        return 'results/combined_shap_analysis.png'
