"""
精简结果管理器
Simplified Result Manager

只生成三个核心输出：
1. 四个模型在各个数据集上的各个指标表格
2. SHAP值排序图
3. 最优topk规则提取结果
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pickle


class ResultManager:
    """精简结果管理器"""
    
    def __init__(self, results_dir='results'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
    
    def generate_model_comparison_table(self, teacher_models, baseline_results, 
                                      all_feature_results, topk_results):
        """
        生成四个模型的性能对比表格
        
        Args:
            teacher_models: 教师模型结果
            baseline_results: 基线决策树结果  
            all_feature_results: 全特征蒸馏结果
            topk_results: Top-k特征蒸馏结果
        
        Returns:
            str: 保存的Excel文件路径
        """
        print("📊 生成模型性能对比表格...")
        
        comparison_data = []
        
        for dataset_name in ['uci', 'german', 'australian']:
            # 1. 教师模型 (PyTorch神经网络)
            if dataset_name in teacher_models:
                teacher_result = teacher_models[dataset_name]
                comparison_data.append({
                    'Dataset': dataset_name.upper(),
                    'Model_Type': 'Teacher_Model',
                    'Architecture': 'PyTorch_DNN',
                    'Accuracy': teacher_result.get('accuracy', 0),
                    'Precision': teacher_result.get('precision', 0),
                    'Recall': teacher_result.get('recall', 0),
                    'F1_Score': teacher_result.get('f1', 0),
                    'AUC': teacher_result.get('auc', 0)
                })
            
            # 2. 基线决策树
            if dataset_name in baseline_results:
                baseline_result = baseline_results[dataset_name]
                comparison_data.append({
                    'Dataset': dataset_name.upper(),
                    'Model_Type': 'Baseline_DecisionTree',
                    'Architecture': 'Decision_Tree',
                    'Accuracy': baseline_result.get('accuracy', 0),
                    'Precision': baseline_result.get('precision', 0),
                    'Recall': baseline_result.get('recall', 0),
                    'F1_Score': baseline_result.get('f1', 0),
                    'AUC': baseline_result.get('auc', 0)
                })
            
            # 3. 全特征知识蒸馏
            if dataset_name in all_feature_results:
                dataset_results = all_feature_results[dataset_name]
                
                # 处理不同的结果结构
                if isinstance(dataset_results, dict) and 'best' in dataset_results:
                    # 简化结构：{dataset: {'best': result}}
                    best_all_feature = dataset_results['best']
                else:
                    # 复杂结构：需要遍历
                    best_all_feature = self._extract_best_result(dataset_results)
                
                if best_all_feature:
                    comparison_data.append({
                        'Dataset': dataset_name.upper(),
                        'Model_Type': 'All_Feature_Distillation',
                        'Architecture': 'Distilled_DecisionTree',
                        'Accuracy': best_all_feature.get('accuracy', 0),
                        'Precision': best_all_feature.get('precision', 0),
                        'Recall': best_all_feature.get('recall', 0),
                        'F1_Score': best_all_feature.get('f1', 0),
                        'AUC': best_all_feature.get('auc', 0)
                    })
            
            # 4. Top-k特征知识蒸馏
            if dataset_name in topk_results:
                dataset_results = topk_results[dataset_name]
                
                # 处理不同的结果结构
                if isinstance(dataset_results, dict) and 'best' in dataset_results:
                    # 简化结构：{dataset: {'best': result}}
                    best_topk = dataset_results['best']
                else:
                    # 复杂结构：需要遍历
                    best_topk = self._extract_best_topk_result(dataset_results)
                
                if best_topk:
                    comparison_data.append({
                        'Dataset': dataset_name.upper(),
                        'Model_Type': 'TopK_Feature_Distillation',
                        'Architecture': 'Distilled_DecisionTree',
                        'Accuracy': best_topk.get('accuracy', 0),
                        'Precision': best_topk.get('precision', 0),
                        'Recall': best_topk.get('recall', 0),
                        'F1_Score': best_topk.get('f1', 0),
                        'AUC': best_topk.get('auc', 0)
                    })
        
        # 创建DataFrame并保存
        df = pd.DataFrame(comparison_data)
        
        # 格式化数值
        numeric_cols = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC']
        for col in numeric_cols:
            df[col] = df[col].round(4)
        
        # 保存到Excel
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = os.path.join(self.results_dir, f'model_comparison_{timestamp}.xlsx')
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Model_Comparison', index=False)
        
        print(f"   ✅ 模型对比表格已保存：{excel_path}")
        return excel_path
    
    def generate_shap_visualization(self, shap_results):
        """
        生成SHAP值排序可视化图
        
        Args:
            shap_results: SHAP分析结果
            
        Returns:
            str: 保存的图片文件路径
        """
        print("📈 生成SHAP特征重要性排序图...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('SHAP特征重要性排序 (各数据集)', fontsize=16, fontweight='bold')
        
        dataset_names = ['uci', 'german', 'australian']
        dataset_labels = ['UCI信用卡', 'German信用', 'Australian信用']
        
        for i, (dataset_name, dataset_label) in enumerate(zip(dataset_names, dataset_labels)):
            if dataset_name in shap_results:
                # 获取特征重要性数据
                shap_data = shap_results[dataset_name]
                if 'feature_importance' in shap_data:
                    importance_data = shap_data['feature_importance']
                    
                    # 处理不同格式的特征重要性数据
                    if isinstance(importance_data, np.ndarray):
                        # 如果是numpy数组，创建简单的特征名称
                        features = [f'Feature_{j}' for j in range(len(importance_data))]
                        importance_df = pd.DataFrame({
                            'feature': features,
                            'importance': importance_data
                        }).sort_values('importance', ascending=False)
                    elif isinstance(importance_data, pd.DataFrame):
                        importance_df = importance_data
                    elif isinstance(importance_data, dict):
                        importance_df = pd.DataFrame(list(importance_data.items()), 
                                                   columns=['feature', 'importance']).sort_values('importance', ascending=False)
                    else:
                        continue
                    
                    # 选择前10个最重要的特征
                    top_features = importance_df.head(10)
                    
                    # 绘制水平条形图
                    axes[i].barh(range(len(top_features)), top_features['importance'], 
                               color=plt.cm.viridis(np.linspace(0, 1, len(top_features))))
                    axes[i].set_yticks(range(len(top_features)))
                    axes[i].set_yticklabels(top_features['feature'], fontsize=10)
                    axes[i].set_xlabel('SHAP重要性值', fontsize=12)
                    axes[i].set_title(f'{dataset_label}数据集', fontsize=14, fontweight='bold')
                    axes[i].grid(axis='x', alpha=0.3)
                    
                    # 反转y轴，让最重要的特征在顶部
                    axes[i].invert_yaxis()
        
        plt.tight_layout()
        
        # 保存图片
        img_path = os.path.join(self.results_dir, 'shap_feature_importance.png')
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ SHAP可视化图已保存：{img_path}")
        return img_path
    
    def extract_best_topk_rules(self, topk_results, processed_data):
        """
        提取最优Top-k规则
        
        Args:
            topk_results: Top-k蒸馏结果
            processed_data: 预处理后的数据
            
        Returns:
            str: 保存的规则文件路径
        """
        print("🌳 提取最优Top-k决策树规则...")
        
        rules_data = []
        
        for dataset_name in ['uci', 'german', 'australian']:
            if dataset_name in topk_results:
                # 找到最佳配置
                best_config = self._find_best_topk_config(topk_results[dataset_name])
                
                if best_config:
                    # 提取决策规则文本
                    tree_rules = best_config.get('tree_rules', 'N/A')
                    if tree_rules == 'N/A' or not tree_rules:
                        # 尝试从rules字段重新构建
                        rules_obj = best_config.get('rules', {})
                        if isinstance(rules_obj, dict) and 'rules' in rules_obj:
                            tree_rules = '\n'.join(rules_obj['rules']) if isinstance(rules_obj['rules'], list) else str(rules_obj['rules'])
                        else:
                            tree_rules = '无法提取决策规则'
                    
                    rules_data.append({
                        'Dataset': dataset_name.upper(),
                        'Best_K': best_config.get('k', 'N/A'),
                        'Best_Temperature': best_config.get('temperature', 'N/A'),
                        'Best_Alpha': best_config.get('alpha', 'N/A'),
                        'Best_Depth': best_config.get('max_depth', 'N/A'),
                        'Accuracy': best_config.get('accuracy', 0),
                        'F1_Score': best_config.get('f1', 0),
                        'Selected_Features': ', '.join(best_config.get('selected_features', [])),
                        'Tree_Rules': tree_rules
                    })
        
        # 保存到文本文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rules_path = os.path.join(self.results_dir, f'best_topk_rules_{timestamp}.txt')
        
        with open(rules_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("最优Top-k特征知识蒸馏决策树规则\n")
            f.write("Best Top-k Feature Knowledge Distillation Decision Tree Rules\n")
            f.write("=" * 80 + "\n\n")
            
            for rule_data in rules_data:
                f.write(f"数据集: {rule_data['Dataset']}\n")
                f.write(f"最佳配置: k={rule_data['Best_K']}, T={rule_data['Best_Temperature']}, "
                       f"α={rule_data['Best_Alpha']}, depth={rule_data['Best_Depth']}\n")
                f.write(f"性能: Accuracy={rule_data['Accuracy']:.4f}, F1={rule_data['F1_Score']:.4f}\n")
                f.write(f"选择特征: {rule_data['Selected_Features']}\n")
                f.write(f"决策规则:\n{rule_data['Tree_Rules']}\n")
                f.write("-" * 80 + "\n\n")
        
        print(f"   ✅ 最优Top-k规则已保存：{rules_path}")
        return rules_path
    
    def clean_output_files(self):
        """清理旧的输出文件，只保留最新的三个核心文件"""
        print("🧹 清理旧的输出文件...")
        
        # 需要保留的文件模式
        keep_patterns = [
            'model_comparison_',
            'shap_feature_importance.png',
            'best_topk_rules_'
        ]
        
        # 删除其他文件
        for filename in os.listdir(self.results_dir):
            file_path = os.path.join(self.results_dir, filename)
            
            # 检查是否需要保留
            should_keep = False
            for pattern in keep_patterns:
                if pattern in filename:
                    should_keep = True
                    break
            
            if not should_keep and os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                    print(f"   删除文件：{filename}")
                except Exception as e:
                    print(f"   删除文件失败 {filename}: {e}")
    
    def _extract_best_result(self, results):
        """从结果中提取最佳模型"""
        if not results:
            return None
        
        # 如果results直接是一个字典且包含评估指标，直接返回
        if isinstance(results, dict) and ('f1' in results or 'accuracy' in results):
            return results
        
        # 如果results是模型对象，返回None（无法直接提取指标）
        if not isinstance(results, dict):
            return None
        
        best_f1 = -1
        best_result = None
        
        # 处理嵌套字典结构
        try:
            # 遍历所有配置找最佳F1
            for temp, alpha_results in results.items():
                if not isinstance(alpha_results, dict):
                    continue
                for alpha, depth_results in alpha_results.items():
                    if not isinstance(depth_results, dict):
                        continue
                    for depth, result in depth_results.items():
                        if isinstance(result, dict) and 'f1' in result:
                            if result['f1'] > best_f1:
                                best_f1 = result['f1']
                                best_result = result
        except AttributeError:
            # 如果遍历失败，返回None
            return None
        
        return best_result
    
    def _extract_best_topk_result(self, results):
        """从Top-k结果中提取最佳模型"""
        if not results:
            return None
        
        # 如果results直接是一个字典且包含评估指标，直接返回
        if isinstance(results, dict) and ('f1' in results or 'accuracy' in results):
            return results
        
        # 如果results不是字典，返回None
        if not isinstance(results, dict):
            return None
        
        best_f1 = -1
        best_result = None
        
        try:
            # 遍历所有k值和配置找最佳F1
            for k, temp_results in results.items():
                if not isinstance(temp_results, dict):
                    continue
                for temp, alpha_results in temp_results.items():
                    if not isinstance(alpha_results, dict):
                        continue
                    for alpha, depth_results in alpha_results.items():
                        if not isinstance(depth_results, dict):
                            continue
                        for depth, result in depth_results.items():
                            if isinstance(result, dict) and 'f1' in result:
                                if result['f1'] > best_f1:
                                    best_f1 = result['f1']
                                    best_result = result
        except (AttributeError, TypeError):
            return None
        
        return best_result
        
        return best_result
    
    def _find_best_topk_config(self, results):
        """找到最佳Top-k配置的详细信息"""
        if not results:
            return None
        
        # 如果results直接是最佳结果字典（包含best键）
        if isinstance(results, dict) and 'best' in results:
            best_result = results['best'].copy()
            # 添加k值
            if 'best_k' in results:
                best_result['k'] = results['best_k']
            
            # 提取决策规则 - 修复规则提取逻辑
            if 'rules' in best_result:
                rules_obj = best_result['rules']
                if isinstance(rules_obj, dict):
                    if 'rules' in rules_obj and isinstance(rules_obj['rules'], list):
                        # 如果rules是列表格式，直接连接
                        best_result['tree_rules'] = '\n'.join(rules_obj['rules'])
                    elif 'description' in rules_obj:
                        # 如果只有描述，使用描述
                        best_result['tree_rules'] = rules_obj['description']
                    else:
                        # 其他情况转换为字符串
                        best_result['tree_rules'] = str(rules_obj)
                elif isinstance(rules_obj, list):
                    # 如果rules直接是列表
                    best_result['tree_rules'] = '\n'.join(rules_obj)
                else:
                    # 其他类型转换为字符串
                    best_result['tree_rules'] = str(rules_obj)
            else:
                best_result['tree_rules'] = 'No rules extracted'
            
            return best_result
        
        # 如果results直接是一个包含指标的字典
        if isinstance(results, dict) and ('f1' in results or 'accuracy' in results):
            return results
        
        # 处理嵌套结构
        if not isinstance(results, dict):
            return None
        
        best_f1 = -1
        best_config = None
        
        try:
            for k, temp_results in results.items():
                if not isinstance(temp_results, dict):
                    continue
                for temp, alpha_results in temp_results.items():
                    if not isinstance(alpha_results, dict):
                        continue
                    for alpha, depth_results in alpha_results.items():
                        if not isinstance(depth_results, dict):
                            continue
                        for depth, result in depth_results.items():
                            if isinstance(result, dict) and 'f1' in result:
                                if result['f1'] > best_f1:
                                    best_f1 = result['f1']
                                    best_config = result.copy()
                                    best_config.update({
                                        'k': k,
                                        'temperature': temp,
                                        'alpha': alpha,
                                        'max_depth': depth
                                    })
                                    # 提取决策规则
                                    if 'rules' in result and isinstance(result['rules'], dict):
                                        if 'rules' in result['rules']:
                                            best_config['tree_rules'] = '\n'.join(result['rules']['rules'])
                                        else:
                                            best_config['tree_rules'] = str(result['rules'])
        except (AttributeError, TypeError):
            return None
        
        return best_config