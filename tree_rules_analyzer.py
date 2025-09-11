"""
决策树规则分析器
Decision Tree Rules Analyzer
"""

import pandas as pd
from sklearn.tree import export_text, _tree
import warnings
warnings.filterwarnings('ignore')

class DecisionTreeRulesAnalyzer:
    """决策树规则分析器 - 提取最优蒸馏树的决策规则"""
    
    def __init__(self, results_dir='results'):
        self.results_dir = results_dir
    
    def extract_best_distillation_tree_rules(self, top_k_distillation_results, processed_data):
        """提取所有数据集的最优蒸馏树规则"""
        print("   🌳 Extracting decision tree rules from best distillation models...")
        
        all_rules_data = []
        best_trees_info = {}
        
        for dataset_name in top_k_distillation_results:
            print(f"   🔍 Processing {dataset_name.upper()} dataset...")
            
            # 找到该数据集的最优树
            best_tree_info = self._find_best_tree_for_dataset(
                dataset_name, top_k_distillation_results, processed_data
            )
            
            if best_tree_info:
                best_trees_info[dataset_name] = best_tree_info
                
                # 提取决策规则
                # 使用 selected_features 作为特征名称
                feature_names_to_use = best_tree_info.get('selected_features', best_tree_info.get('feature_names', []))
                rules = self._extract_tree_rules(
                    best_tree_info['model'], 
                    feature_names_to_use,
                    dataset_name
                )
                
                # 添加元数据
                for rule in rules:
                    rule['Dataset'] = dataset_name.upper()
                    rule['Best_Config'] = f"k={best_tree_info['k']}, T={best_tree_info['temperature']}, α={best_tree_info['alpha']}, D={best_tree_info['max_depth']}"
                    rule['Tree_Accuracy'] = best_tree_info['accuracy']
                
                all_rules_data.extend(rules)
                
        if not all_rules_data:
            print("   ⚠️  No valid decision trees found for rule extraction")
            return None, {}
        
        # 保存到Excel
        excel_path = self._save_rules_to_excel(all_rules_data, best_trees_info)
        
        # 保存各数据集最优Top-K规则为txt文件
        self._save_best_topk_rules_as_txt(best_trees_info, processed_data)
        
        print(f"   📊 Decision tree rules extracted and saved to: {excel_path}")
        return excel_path, best_trees_info
    
    def _find_best_tree_for_dataset(self, dataset_name, top_k_distillation_results, processed_data):
        """为指定数据集找到最优的蒸馏树（适配简化版本的数据结构）"""
        
        if dataset_name not in top_k_distillation_results:
            return None
            
        dataset_results = top_k_distillation_results[dataset_name]
        feature_names = processed_data[dataset_name]['feature_names']
        
        # 检查数据结构类型
        if 'best' in dataset_results:
            # 简化版本：直接使用最佳结果
            result = dataset_results['best']
            best_k = dataset_results.get('best_k', 'unknown')
            
            if result is not None and 'accuracy' in result:
                best_tree_info = {
                    'dataset': dataset_name,
                    'k': best_k,
                    'temperature': result.get('temperature', 'unknown'),
                    'alpha': result.get('alpha', 'unknown'),
                    'max_depth': result.get('max_depth', 'unknown'),
                    'accuracy': result['accuracy'],
                    'f1': result['f1'],
                    'precision': result['precision'],
                    'recall': result['recall'],
                    'model': result['model'],
                    'selected_features': result.get('selected_features', feature_names),
                    'feature_names': result.get('selected_features', feature_names),  # 添加兼容性键
                    'feature_count': result.get('feature_count', len(feature_names))
                }
                return best_tree_info
        else:
            # 完整版本：遍历所有参数组合找最优树
            best_accuracy = 0.0
            best_tree_info = None
            
            for k in dataset_results:
                for temp in dataset_results[k]:
                    for alpha in dataset_results[k][temp]:
                        for depth in dataset_results[k][temp][alpha]:
                            result = dataset_results[k][temp][alpha][depth]
                            
                            if result is not None and 'accuracy' in result:
                                if result['accuracy'] > best_accuracy:
                                    best_accuracy = result['accuracy']
                                    best_tree_info = {
                                        'dataset': dataset_name,
                                        'k': k,
                                        'temperature': temp,
                                        'alpha': alpha,
                                        'max_depth': depth,
                                        'accuracy': result['accuracy'],
                                        'f1': result['f1'],
                                        'precision': result['precision'],
                                        'recall': result['recall'],
                                        'model': result['model'],
                                        'selected_features': result.get('selected_features', feature_names),
                                        'feature_names': result.get('selected_features', feature_names),  # 添加兼容性键
                                        'feature_count': result.get('feature_count', len(feature_names))
                                    }
            return best_tree_info
        
        return None
    
    def _extract_tree_rules(self, tree_model, feature_names, dataset_name):
        """从决策树模型中提取规则"""
        rules_data = []
        
        try:
            tree = tree_model.tree_
            
            def recurse(node_id, depth, parent_rule="Root"):
                # 如果是叶节点
                if tree.children_left[node_id] == tree.children_right[node_id]:
                    # 获取叶节点的类别分布
                    value = tree.value[node_id]
                    samples = tree.n_node_samples[node_id]
                    
                    # 对于二分类，选择多数类别
                    predicted_class = 1 if value[0][1] > value[0][0] else 0
                    confidence = max(value[0]) / sum(value[0])
                    
                    rule_data = {
                        'Rule_ID': len(rules_data) + 1,
                        'Rule_Path': parent_rule,
                        'Prediction': f"Class_{predicted_class}",
                        'Confidence': f"{confidence:.3f}",
                        'Samples': samples,
                        'Depth': depth,
                        'Class_Distribution': f"[{value[0][0]:.0f}, {value[0][1]:.0f}]"
                    }
                    rules_data.append(rule_data)
                    return
                
                # 内部节点
                feature_idx = tree.feature[node_id]
                threshold = tree.threshold[node_id]
                feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f"feature_{feature_idx}"
                
                # 左子树（<= threshold）
                left_rule = f"{parent_rule} → {feature_name} ≤ {threshold:.3f}"
                recurse(tree.children_left[node_id], depth + 1, left_rule)
                
                # 右子树（> threshold）
                right_rule = f"{parent_rule} → {feature_name} > {threshold:.3f}"
                recurse(tree.children_right[node_id], depth + 1, right_rule)
            
            # 从根节点开始递归
            recurse(0, 0)
            
        except Exception as e:
            print(f"   ⚠️  Error extracting rules for {dataset_name}: {str(e)}")
            
        return rules_data
    
    def _save_rules_to_excel(self, all_rules_data, best_trees_info):
        """保存规则到Excel文件"""
        excel_path = f"{self.results_dir}/decision_tree_rules_analysis.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Sheet 1: 所有规则
            rules_df = pd.DataFrame(all_rules_data)
            rules_df.to_excel(writer, sheet_name='Decision_Rules', index=False)
            
            # Sheet 2: 最优树汇总
            summary_data = []
            for dataset_name, tree_info in best_trees_info.items():
                summary_data.append({
                    'Dataset': dataset_name.upper(),
                    'Best_k': tree_info['k'],
                    'Best_Temperature': tree_info['temperature'],
                    'Best_Alpha': tree_info['alpha'],
                    'Best_Max_Depth': tree_info['max_depth'],
                    'Tree_Accuracy': tree_info['accuracy'],
                    'Tree_F1': tree_info['f1'],
                    'Tree_Precision': tree_info['precision'],
                    'Tree_Recall': tree_info['recall'],
                    'Feature_Count': tree_info['feature_count']
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Best_Trees_Summary', index=False)
            
            # Sheet 3: 按数据集分组的规则统计
            if all_rules_data:
                rules_df = pd.DataFrame(all_rules_data)
                stats_data = []
                
                for dataset in rules_df['Dataset'].unique():
                    dataset_rules = rules_df[rules_df['Dataset'] == dataset]
                    stats_data.append({
                        'Dataset': dataset,
                        'Total_Rules': len(dataset_rules),
                        'Avg_Depth': dataset_rules['Depth'].mean(),
                        'Max_Depth': dataset_rules['Depth'].max(),
                        'Avg_Samples_Per_Rule': dataset_rules['Samples'].mean(),
                        'Total_Samples_Covered': dataset_rules['Samples'].sum()
                    })
                
                stats_df = pd.DataFrame(stats_data)
                stats_df.to_excel(writer, sheet_name='Rules_Statistics', index=False)
        
        return excel_path
    
    def generate_tree_text_representation(self, best_trees_info, processed_data):
        """生成决策树的文本表示"""
        tree_text_representations = {}
        
        for dataset_name, tree_info in best_trees_info.items():
            try:
                feature_names = tree_info.get('selected_features', processed_data[dataset_name]['feature_names'])
                tree_text = export_text(tree_info['model'], feature_names=feature_names)
                tree_text_representations[dataset_name] = tree_text
                
                # 保存到文件
                text_file_path = f"{self.results_dir}/tree_text_{dataset_name}.txt"
                with open(text_file_path, 'w', encoding='utf-8') as f:
                    f.write(f"Decision Tree Text Representation for {dataset_name.upper()}\n")
                    f.write("="*60 + "\n\n")
                    f.write(f"Best Configuration: k={tree_info['k']}, T={tree_info['temperature']}, α={tree_info['alpha']}, D={tree_info['max_depth']}\n")
                    f.write(f"Tree Accuracy: {tree_info['accuracy']:.4f}\n\n")
                    f.write(tree_text)
                    
            except Exception as e:
                print(f"   ⚠️  Error generating text representation for {dataset_name}: {str(e)}")
                tree_text_representations[dataset_name] = f"Error: {str(e)}"
        
        return tree_text_representations
    
    def _save_best_topk_rules_as_txt(self, best_trees_info, processed_data):
        """保存各数据集最优Top-K规则为txt文件"""
        print("   📝 Saving best Top-K rules as txt files...")
        
        for dataset_name, tree_info in best_trees_info.items():
            try:
                # 生成txt文件名
                txt_filename = f"{self.results_dir}/best_topk_rules_{dataset_name}.txt"
                
                # 获取特征名称
                feature_names = tree_info.get('selected_features', processed_data[dataset_name]['feature_names'])
                
                # 生成决策树文本表示
                tree_text = export_text(tree_info['model'], feature_names=feature_names, max_depth=10)
                
                # 写入txt文件
                with open(txt_filename, 'w', encoding='utf-8') as f:
                    f.write(f"最优Top-K知识蒸馏决策树规则 - {dataset_name.upper()}数据集\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(f"数据集: {dataset_name.upper()}\n")
                    f.write(f"最优配置: k={tree_info['k']}, T={tree_info['temperature']}, α={tree_info['alpha']}, D={tree_info['max_depth']}\n")
                    f.write(f"准确率: {tree_info['accuracy']:.4f}\n")
                    f.write(f"F1分数: {tree_info['f1']:.4f}\n")
                    f.write(f"精确率: {tree_info['precision']:.4f}\n")
                    f.write(f"召回率: {tree_info['recall']:.4f}\n")
                    f.write(f"使用特征数: {tree_info['feature_count']}\n")
                    f.write("\n" + "-" * 60 + "\n")
                    f.write("决策树规则:\n")
                    f.write("-" * 60 + "\n\n")
                    f.write(tree_text)
                    f.write("\n\n" + "=" * 60 + "\n")
                    f.write("特征列表:\n")
                    for i, feature in enumerate(feature_names, 1):
                        f.write(f"{i:2d}. {feature}\n")
                
                print(f"   ✅ Saved: {txt_filename}")
                
            except Exception as e:
                print(f"   ⚠️  Error saving txt for {dataset_name}: {str(e)}")
        
        print("   📝 Best Top-K rules txt files saved successfully!")
