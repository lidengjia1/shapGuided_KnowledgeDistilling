# Knowledge Distillation Module - Decision Tree Only
# 知识蒸馏模块 - 仅决策树版本

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import optuna
from tqdm import tqdm
import warnings
import multiprocessing as mp
import os

warnings.filterwarnings('ignore')

# 设置matplotlib后端为非交互式，避免多线程问题
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 禁用Optuna日志输出
optuna.logging.set_verbosity(optuna.logging.WARNING)

class KnowledgeDistillator:
    """知识蒸馏系统 - 决策树蒸馏"""
    
    def __init__(self, teacher_models, processed_data, all_shap_results):
        self.teacher_models = teacher_models
        self.processed_data = processed_data
        self.all_shap_results = all_shap_results
        
    def extract_knowledge(self, dataset_name, model_type, temperature=3.0):
        """从教师模型提取知识
        
        知识蒸馏理论背景：
        教师模型输出softmax分布包含更丰富的类间关系信息
        temperature参数控制分布的平滑程度，温度越高分布越平滑
        """
        teacher_model = self.teacher_models[dataset_name]['model']
        data_dict = self.processed_data[dataset_name]
        
        X_train = data_dict['X_train']
        X_test = data_dict['X_test']
        
        # 获取教师模型的软标签(概率分布)
        train_logits = teacher_model.predict_proba(X_train)
        test_logits = teacher_model.predict_proba(X_test)
        
        # 应用温度缩放，增强知识蒸馏效果
        train_soft_labels = self._apply_temperature(train_logits, temperature)
        test_soft_labels = self._apply_temperature(test_logits, temperature)
        
        return {
            'train_soft_labels': train_soft_labels,
            'test_soft_labels': test_soft_labels,
            'teacher_logits_train': train_logits,
            'teacher_logits_test': test_logits
        }
    
    def _apply_temperature(self, logits, temperature):
        """温度缩放：logits / T，然后应用softmax
        温度T > 1 使分布更平滑，T < 1 使分布更sharp
        """
        return F.softmax(torch.tensor(logits) / temperature, dim=1).numpy()
    
    def train_student_model(self, dataset_name, model_type_name='decision_tree', 
                          k=5, temperature=3.0, alpha=0.7, max_depth=6, 
                          use_all_features=False, trial=None):
        """训练学生模型(决策树)使用知识蒸馏
        
        参数:
        - dataset_name: 数据集名称
        - model_type_name: 学生模型类型，固定为'decision_tree'
        - k: Top-k特征数量
        - temperature: 知识蒸馏温度参数
        - alpha: 蒸馏损失权重 (0=仅硬标签, 1=仅软标签)
        - max_depth: 决策树最大深度
        - use_all_features: 是否使用全特征
        - trial: Optuna trial对象(用于超参数优化)
        """
        
        data_dict = self.processed_data[dataset_name]
        
        # 特征选择
        if use_all_features:
            # 使用全特征
            X_train_selected = data_dict['X_train']
            X_test_selected = data_dict['X_test']
            selected_features = data_dict['feature_names']
            model_type = f'all_features_decision_tree_distillation'
        else:
            # 选择Top-k特征
            shap_results = self.all_shap_results[dataset_name]
            top_k_features = shap_results['top_k_features'][k]
            feature_indices = [data_dict['feature_names'].index(feat) for feat in top_k_features]
            
            X_train_selected = data_dict['X_train'][:, feature_indices]
            X_test_selected = data_dict['X_test'][:, feature_indices]
            selected_features = top_k_features
            model_type = f'top_{k}_decision_tree_distillation'
        
        y_train = data_dict['y_train']
        y_test = data_dict['y_test']
        
        # 提取教师模型知识
        knowledge = self.extract_knowledge(dataset_name, 'teacher', temperature)
        train_soft_labels = knowledge['train_soft_labels']
        test_soft_labels = knowledge['test_soft_labels']
        
        # 创建决策树学生模型
        student_model = self._create_decision_tree_student(trial, max_depth)
        
        # 知识蒸馏训练
        student_model = self._train_with_distillation(
            student_model, X_train_selected, y_train, train_soft_labels, alpha
        )
        
        # 预测和评估
        y_pred = student_model.predict(X_test_selected)
        y_pred_proba = student_model.predict_proba(X_test_selected)
        
        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # 提取决策规则
        rules = self._extract_decision_rules(student_model, selected_features)
        
        return {
            'model': student_model,
            'model_type': model_type,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'feature_count': len(selected_features),
            'selected_features': selected_features,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'rules': rules,
            'temperature': temperature,
            'alpha': alpha,
            'max_depth': max_depth,
            'hyperparameters': {
                'temperature': temperature,
                'alpha': alpha,
                'max_depth': max_depth
            }
        }
    
    def _create_decision_tree_student(self, trial, max_depth):
        """创建决策树学生模型"""
        if trial is not None:
            # Optuna超参数优化
            trial_max_depth = trial.suggest_int('max_depth', 3, 12)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
            max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        else:
            # 使用固定参数
            trial_max_depth = max_depth
            min_samples_split = 2
            min_samples_leaf = 1
            max_features = 'sqrt'
        
        return DecisionTreeClassifier(
            max_depth=trial_max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42
        )
    
    def _train_with_distillation(self, model, X_train, y_train, soft_labels, alpha):
        """使用知识蒸馏训练决策树
        
        对于决策树，我们使用软标签的概率作为样本权重
        这是一种近似的知识蒸馏方法，因为决策树不直接支持软标签
        """
        
        if alpha > 0:
            # 使用软标签的最大概率作为样本权重
            sample_weights = np.max(soft_labels, axis=1)
            # 归一化权重
            sample_weights = sample_weights / np.sum(sample_weights) * len(sample_weights)
            
            # 训练时使用样本权重
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            # 纯硬标签训练
            model.fit(X_train, y_train)
        
        return model
    
    def _extract_decision_rules(self, model, feature_names):
        """提取决策树规则"""
        from tree_rules_analyzer import DecisionTreeRulesAnalyzer
        
        extractor = DecisionTreeRulesAnalyzer()
        # 使用简单的规则提取方法
        rules = self._simple_extract_rules(model, feature_names)
        
        return {
            'rules': rules,
            'rule_count': len(rules),
            'description': f'Decision tree with {len(rules)} rules'
        }
    
    def _simple_extract_rules(self, model, feature_names):
        """简单的决策树规则提取"""
        tree = model.tree_
        rules = []
        
        def recurse(node, depth, parent_rule=""):
            indent = "  " * depth
            if tree.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_names[tree.feature[node]]
                threshold = tree.threshold[node]
                left_rule = f"{parent_rule}{name} <= {threshold:.3f}"
                right_rule = f"{parent_rule}{name} > {threshold:.3f}"
                recurse(tree.children_left[node], depth + 1, left_rule + " and ")
                recurse(tree.children_right[node], depth + 1, right_rule + " and ")
            else:
                # 叶子节点
                if parent_rule:
                    rule = parent_rule.rstrip(" and ")
                    value = tree.value[node]
                    predicted_class = np.argmax(value)
                    confidence = np.max(value) / np.sum(value)
                    rules.append(f"IF {rule} THEN class={predicted_class} (confidence={confidence:.3f})")
        
        try:
            from sklearn.tree import _tree
            recurse(0, 0)
        except:
            # 如果规则提取失败，返回简单描述
            rules = [f"Decision tree with {model.tree_.node_count} nodes"]
        
        return rules
    
    def train_baseline_decision_tree(self, dataset_name):
        """训练基础决策树（不使用蒸馏）"""
        data_dict = self.processed_data[dataset_name]
        
        X_train = data_dict['X_train']
        X_test = data_dict['X_test']
        y_train = data_dict['y_train']
        y_test = data_dict['y_test']
        feature_names = data_dict['feature_names']
        
        # 使用Optuna优化基础决策树
        def objective(trial):
            max_depth = trial.suggest_int('max_depth', 3, 12)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
            max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            
            model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return f1_score(y_test, y_pred, average='weighted')
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        # 使用最优参数训练最终模型
        best_params = study.best_params
        model = DecisionTreeClassifier(**best_params, random_state=42)
        model.fit(X_train, y_train)
        
        # 预测和评估
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # 提取决策规则
        rules = self._extract_decision_rules(model, feature_names)
        
        return {
            'model': model,
            'model_type': 'baseline_optimized_tree',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'feature_count': len(feature_names),
            'selected_features': feature_names,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'rules': rules,
            'hyperparameters': best_params,  # 改为hyperparameters以保持一致
            'best_params': best_params       # 保留原有字段以兼容
        }
    
    def run_all_feature_distillation(self, dataset_names, temperature_range, alpha_range, max_depth_range):
        """运行全特征知识蒸馏实验"""
        results = {}
        
        for dataset_name in dataset_names:
            print(f"   Processing {dataset_name.upper()} dataset...")
            results[dataset_name] = {}
            
            best_accuracy = 0  # 改为使用准确率作为评判标准
            best_result = None
            
            total_combinations = len(temperature_range) * len(alpha_range) * len(max_depth_range)
            progress_bar = tqdm(total=total_combinations, desc=f"     {dataset_name}")
            
            for temperature in temperature_range:
                for alpha in alpha_range:
                    for max_depth in max_depth_range:
                        result = self.train_student_model(
                            dataset_name=dataset_name,
                            model_type_name='decision_tree',
                            use_all_features=True,
                            temperature=temperature,
                            alpha=alpha,
                            max_depth=max_depth
                        )
                        
                        if result['accuracy'] > best_accuracy:  # 改为使用准确率
                            best_accuracy = result['accuracy']
                            best_result = result
                        
                        progress_bar.update(1)
            
            progress_bar.close()
            results[dataset_name]['best'] = best_result
            print(f"     Best Accuracy: {best_accuracy:.4f}")  # 改为显示准确率
        
        return results
    
    def run_all_feature_distillation_with_optuna(self, dataset_names, temperature_range, alpha_range, max_depth_range, n_trials=50):
        """运行全特征知识蒸馏实验 - 使用Optuna优化其他参数"""
        results = {}
        
        for dataset_name in dataset_names:
            print(f"   Processing {dataset_name.upper()} dataset with Optuna optimization...")
            results[dataset_name] = {}
            
            best_accuracy = 0
            best_result = None
            best_config = None
            
            # 遍历主要参数
            total_combinations = len(temperature_range) * len(alpha_range) * len(max_depth_range)
            progress_bar = tqdm(total=total_combinations, desc=f"     {dataset_name}")
            
            for temperature in temperature_range:
                for alpha in alpha_range:
                    for max_depth in max_depth_range:
                        
                        # 为每个主要参数组合使用Optuna优化其他参数
                        def objective(trial):
                            # Optuna优化的参数
                            min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
                            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
                            
                            try:
                                result = self.train_student_model_with_extra_params(
                                    dataset_name=dataset_name,
                                    model_type_name='decision_tree',
                                    use_all_features=True,
                                    temperature=temperature,
                                    alpha=alpha,
                                    max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf
                                )
                                return result['accuracy']
                            except Exception as e:
                                return 0.0
                        
                        # 运行Optuna优化
                        study = optuna.create_study(direction='maximize', 
                                                  sampler=optuna.samplers.TPESampler(seed=42))
                        study.optimize(objective, n_trials=min(10, n_trials//total_combinations))
                        
                        # 获取最佳结果
                        best_params = study.best_params
                        current_result = self.train_student_model_with_extra_params(
                            dataset_name=dataset_name,
                            model_type_name='decision_tree',
                            use_all_features=True,
                            temperature=temperature,
                            alpha=alpha,
                            max_depth=max_depth,
                            min_samples_split=best_params['min_samples_split'],
                            min_samples_leaf=best_params['min_samples_leaf']
                        )
                        
                        if current_result['accuracy'] > best_accuracy:
                            best_accuracy = current_result['accuracy']
                            best_result = current_result
                            best_config = {
                                'temperature': temperature,
                                'alpha': alpha,
                                'max_depth': max_depth,
                                **best_params
                            }
                        
                        progress_bar.update(1)
            
            progress_bar.close()
            results[dataset_name]['best'] = best_result
            results[dataset_name]['best_config'] = best_config
            print(f"     Best Accuracy: {best_accuracy:.4f} with config: {best_config}")
        
        return results
    
    def run_comprehensive_distillation(self, dataset_names, k_range, temperature_range, alpha_range, max_depth_range):
        """运行综合知识蒸馏实验（Top-k特征）"""
        results = {}
        
        for dataset_name in dataset_names:
            print(f"   Processing {dataset_name.upper()} dataset...")
            results[dataset_name] = {}
            
            best_accuracy = 0  # 改为使用准确率作为评判标准
            best_result = None
            best_k = None
            
            k_values = list(range(k_range[0], k_range[1] + 1))
            total_combinations = len(k_values) * len(temperature_range) * len(alpha_range) * len(max_depth_range)
            progress_bar = tqdm(total=total_combinations, desc=f"     {dataset_name}")
            
            for k in k_values:
                for temperature in temperature_range:
                    for alpha in alpha_range:
                        for max_depth in max_depth_range:
                            result = self.train_student_model(
                                dataset_name=dataset_name,
                                model_type_name='decision_tree',
                                k=k,
                                temperature=temperature,
                                alpha=alpha,
                                max_depth=max_depth,
                                use_all_features=False
                            )
                            
                            if result['accuracy'] > best_accuracy:  # 改为使用准确率
                                best_accuracy = result['accuracy']
                                best_result = result
                                best_k = k
                            
                            progress_bar.update(1)
            
            progress_bar.close()
            results[dataset_name]['best'] = best_result
            results[dataset_name]['best_k'] = best_k
            print(f"     Best Accuracy: {best_accuracy:.4f} with k={best_k}")  # 改为显示准确率
        
        return results
    
    def run_comprehensive_distillation_with_optuna(self, dataset_names, k_range, temperature_range, alpha_range, max_depth_range, n_trials=50):
        """运行综合知识蒸馏实验（Top-k特征）- 使用Optuna优化其他参数"""
        results = {}
        
        for dataset_name in dataset_names:
            print(f"   Processing {dataset_name.upper()} dataset with Optuna optimization...")
            results[dataset_name] = {}
            
            best_accuracy = 0
            best_result = None
            best_k = None
            best_config = None
            
            k_values = list(range(k_range[0], k_range[1] + 1))
            total_combinations = len(k_values) * len(temperature_range) * len(alpha_range) * len(max_depth_range)
            progress_bar = tqdm(total=total_combinations, desc=f"     {dataset_name}")
            
            for k in k_values:
                for temperature in temperature_range:
                    for alpha in alpha_range:
                        for max_depth in max_depth_range:
                            
                            # 为每个主要参数组合使用Optuna优化其他参数
                            def objective(trial):
                                # Optuna优化的参数
                                min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
                                min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
                                
                                try:
                                    result = self.train_student_model_with_extra_params(
                                        dataset_name=dataset_name,
                                        model_type_name='decision_tree',
                                        k=k,
                                        temperature=temperature,
                                        alpha=alpha,
                                        max_depth=max_depth,
                                        use_all_features=False,
                                        min_samples_split=min_samples_split,
                                        min_samples_leaf=min_samples_leaf
                                    )
                                    return result['accuracy']
                                except Exception as e:
                                    return 0.0
                            
                            # 运行Optuna优化
                            study = optuna.create_study(direction='maximize', 
                                                      sampler=optuna.samplers.TPESampler(seed=42))
                            study.optimize(objective, n_trials=min(5, n_trials//total_combinations))
                            
                            # 获取最佳结果
                            best_params = study.best_params
                            current_result = self.train_student_model_with_extra_params(
                                dataset_name=dataset_name,
                                model_type_name='decision_tree',
                                k=k,
                                temperature=temperature,
                                alpha=alpha,
                                max_depth=max_depth,
                                use_all_features=False,
                                min_samples_split=best_params['min_samples_split'],
                                min_samples_leaf=best_params['min_samples_leaf']
                            )
                            
                            if current_result['accuracy'] > best_accuracy:
                                best_accuracy = current_result['accuracy']
                                best_result = current_result
                                best_k = k
                                best_config = {
                                    'k': k,
                                    'temperature': temperature,
                                    'alpha': alpha,
                                    'max_depth': max_depth,
                                    **best_params
                                }
                            
                            progress_bar.update(1)
            
            progress_bar.close()
            results[dataset_name]['best'] = best_result
            results[dataset_name]['best_k'] = best_k
            results[dataset_name]['best_config'] = best_config
            print(f"     Best Accuracy: {best_accuracy:.4f} with k={best_k}, config: {best_config}")
        
        return results
    
    def train_student_model_with_extra_params(self, dataset_name, model_type_name='decision_tree', 
                          k=5, temperature=3.0, alpha=0.7, max_depth=6, 
                          use_all_features=False, min_samples_split=2, min_samples_leaf=1, trial=None):
        """训练学生模型(决策树)使用知识蒸馏 - 包含额外参数优化"""
        
        data_dict = self.processed_data[dataset_name]
        
        # 特征选择
        if use_all_features:
            # 使用全特征
            X_train_selected = data_dict['X_train']
            X_test_selected = data_dict['X_test']
            selected_features = data_dict['feature_names']
            model_type = f'all_features_decision_tree_distillation'
        else:
            # 选择Top-k特征
            shap_results = self.all_shap_results[dataset_name]
            top_k_features = shap_results['top_k_features'][k]
            feature_indices = [data_dict['feature_names'].index(feat) for feat in top_k_features]
            
            X_train_selected = data_dict['X_train'][:, feature_indices]
            X_test_selected = data_dict['X_test'][:, feature_indices]
            selected_features = top_k_features
            model_type = f'top_{k}_decision_tree_distillation'
        
        y_train = data_dict['y_train']
        y_test = data_dict['y_test']
        
        # 提取教师模型知识
        knowledge = self.extract_knowledge(dataset_name, 'teacher', temperature)
        train_soft_labels = knowledge['train_soft_labels']
        test_soft_labels = knowledge['test_soft_labels']
        
        # 创建决策树学生模型 - 包含额外参数
        student_model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            criterion='gini'
        )
        
        # 知识蒸馏训练
        student_model = self._train_with_distillation(
            student_model, X_train_selected, y_train, train_soft_labels, alpha
        )
        
        # 评估学生模型
        y_pred = student_model.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'model': student_model,
            'model_type': model_type,
            'selected_features': selected_features,
            'feature_count': len(selected_features),
            'hyperparameters': {
                'k': k if not use_all_features else 'all',
                'temperature': temperature,
                'alpha': alpha,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf
            }
        }
