"""
实验管理和结果导出模块
Experiment Management and Results Export Module
"""

import pandas as pd
import pickle
import os
from datetime import datetime
import numpy as np
# 设置matplotlib后端为非交互式，避免多线程问题
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import warnings
warnings.filterwarnings('ignore')

class ExperimentManager:
    """实验管理器"""
    
    def __init__(self, results_dir='results'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
    def save_models_and_data(self, teacher_models, processed_data, all_shap_results, distillation_results):
        """保存所有模型和实验数据"""
        print(f"\n💾 Saving experiment artifacts...")
        
        # 创建教师模型结果表格
        teacher_results_path = self.create_teacher_models_table(teacher_models)
        
        # 保存教师模型
        for dataset_name, model_info in teacher_models.items():
            model_path = f"{self.results_dir}/teacher_model_{dataset_name}.pkl"
            torch_model_path = f"{self.results_dir}/teacher_model_{dataset_name}.pth"
            
            # 保存模型信息
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model_type': model_info['model_type'],
                    'test_metrics': model_info.get('test_metrics', {}),
                    'training_info': model_info.get('training_info', {})
                }, f)
            
            # 保存PyTorch模型状态
            torch.save(model_info['model'].state_dict(), torch_model_path)
            print(f"   ✅ Teacher model saved: {dataset_name}")
        
        # 保存处理后的数据
        data_path = f"{self.results_dir}/processed_data.pkl"
        with open(data_path, 'wb') as f:
            pickle.dump(processed_data, f)
        print(f"   ✅ Processed data saved")
        
        # 保存SHAP结果
        shap_path = f"{self.results_dir}/shap_results.pkl"
        with open(shap_path, 'wb') as f:
            pickle.dump(all_shap_results, f)
        print(f"   ✅ SHAP results saved")
        
        # 保存知识蒸馏结果
        distillation_path = f"{self.results_dir}/distillation_results.pkl"
        with open(distillation_path, 'wb') as f:
            pickle.dump(distillation_results, f)
        print(f"   ✅ Knowledge distillation results saved")
        
        return {
            'teacher_models_path': self.results_dir,
            'teacher_results_path': teacher_results_path,
            'processed_data_path': data_path,
            'shap_results_path': shap_path,
            'distillation_results_path': distillation_path
        }
    
    def create_teacher_models_table(self, teacher_models):
        """创建教师模型详细结果表格"""
        print(f"   📊 Creating teacher models results table...")
        
        teacher_results = []
        
        for dataset_name, model_info in teacher_models.items():
            test_metrics = model_info.get('test_metrics', {})
            training_info = model_info.get('training_info', {})
            
            teacher_results.append({
                'Dataset': dataset_name.upper(),
                'Model_Type': model_info.get('model_type', 'Unknown'),
                'Architecture': model_info.get('model_type', 'Neural Network'),
                'Input_Features': len(model_info.get('feature_names', [])) if 'feature_names' in model_info else 'N/A',
                'Accuracy': test_metrics.get('accuracy', 'N/A'),
                'Precision': test_metrics.get('precision', 'N/A'),
                'Recall': test_metrics.get('recall', 'N/A'),
                'F1_Score': test_metrics.get('f1_score', 'N/A'),
                'Training_Loss': training_info.get('final_loss', 'N/A'),
                'Training_Epochs': training_info.get('epochs', 'N/A'),
                'Model_Parameters': training_info.get('parameters', 'N/A'),
                'Training_Time': training_info.get('training_time', 'N/A'),
                'Notes': f"Teacher model for {dataset_name} dataset"
            })
        
        # 转换为DataFrame并保存
        df = pd.DataFrame(teacher_results)
        excel_path = f"{self.results_dir}/teacher_models_results.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Teacher Models', index=False)
            
            # 格式化工作表
            worksheet = writer.sheets['Teacher Models']
            
            # 设置列宽
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
        
        print(f"   ✅ Teacher models table saved: {excel_path}")
        return excel_path
    
    def create_comprehensive_comparison_table(self, teacher_models, baseline_results, 
                                            all_feature_distillation_results, top_k_distillation_results):
        """创建综合对比表格 - 四种模型在三个数据集上的性能对比"""
        print(f"   📊 Creating comprehensive model comparison table...")
        
        comparison_results = []
        
        for dataset_name in ['uci', 'german', 'australian']:
            dataset_upper = dataset_name.upper()
            
            # 1. 教师模型 (Neural Network)
            teacher_info = teacher_models[dataset_name]
            test_metrics = teacher_info.get('test_metrics', {})
            comparison_results.append({
                'Dataset': dataset_upper,
                'Model_Type': 'Teacher Model (Neural Network)',
                'Architecture': teacher_info.get('model_type', 'Neural Network'),
                'Feature_Selection': 'All Features',
                'Accuracy': f"{test_metrics.get('accuracy', 0):.4f}",
                'Precision': f"{test_metrics.get('precision', 0):.4f}",
                'Recall': f"{test_metrics.get('recall', 0):.4f}",
                'F1_Score': f"{test_metrics.get('f1_score', 0):.4f}",
                'Notes': 'Neural Network Teacher Model'
            })
            
            # 2. 基础决策树 (优化但无蒸馏)
            if dataset_name in baseline_results:
                baseline_info = baseline_results[dataset_name]
                comparison_results.append({
                    'Dataset': dataset_upper,
                    'Model_Type': 'Baseline Decision Tree',
                    'Architecture': 'Decision Tree (Optimized)',
                    'Feature_Selection': 'All Features',
                    'Accuracy': f"{baseline_info.get('accuracy', 0):.4f}",
                    'Precision': f"{baseline_info.get('precision', 0):.4f}",
                    'Recall': f"{baseline_info.get('recall', 0):.4f}",
                    'F1_Score': f"{baseline_info.get('f1', 0):.4f}",
                    'Notes': 'Optimized Decision Tree without Knowledge Distillation'
                })
            
            # 3. 全特征知识蒸馏决策树
            if dataset_name in all_feature_distillation_results:
                all_feature_best = self._find_best_model_in_results(all_feature_distillation_results[dataset_name])
                if all_feature_best:
                    comparison_results.append({
                        'Dataset': dataset_upper,
                        'Model_Type': 'All-Feature Distilled Tree',
                        'Architecture': 'Knowledge Distilled Decision Tree',
                        'Feature_Selection': 'All Features',
                        'Accuracy': f"{all_feature_best.get('accuracy', 0):.4f}",
                        'Precision': f"{all_feature_best.get('precision', 0):.4f}",
                        'Recall': f"{all_feature_best.get('recall', 0):.4f}",
                        'F1_Score': f"{all_feature_best.get('f1', 0):.4f}",
                        'Notes': f"KD with T={all_feature_best.get('temperature', 'N/A')}, α={all_feature_best.get('alpha', 'N/A')}"
                    })
            
            # 4. 最优Top-k蒸馏决策树
            if dataset_name in top_k_distillation_results:
                top_k_best = self._find_best_model_in_results(top_k_distillation_results[dataset_name])
                if top_k_best:
                    comparison_results.append({
                        'Dataset': dataset_upper,
                        'Model_Type': 'Top-k Distilled Tree',
                        'Architecture': 'Knowledge Distilled Decision Tree',
                        'Feature_Selection': f"Top-{top_k_best.get('k', 'N/A')} Features",
                        'Accuracy': f"{top_k_best.get('accuracy', 0):.4f}",
                        'Precision': f"{top_k_best.get('precision', 0):.4f}",
                        'Recall': f"{top_k_best.get('recall', 0):.4f}",
                        'F1_Score': f"{top_k_best.get('f1', 0):.4f}",
                        'Notes': f"KD with k={top_k_best.get('k', 'N/A')}, T={top_k_best.get('temperature', 'N/A')}, α={top_k_best.get('alpha', 'N/A')}"
                    })
        
        # 转换为DataFrame并保存
        df = pd.DataFrame(comparison_results)
        excel_path = f"{self.results_dir}/comprehensive_model_comparison.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # 主对比表
            df.to_excel(writer, sheet_name='Model Comparison', index=False)
            
            # 按数据集分组的表格
            for dataset in ['UCI', 'GERMAN', 'AUSTRALIAN']:
                dataset_df = df[df['Dataset'] == dataset].copy()
                dataset_df = dataset_df.drop('Dataset', axis=1)  # 移除数据集列因为已经在sheet名中
                dataset_df.to_excel(writer, sheet_name=f'{dataset} Results', index=False)
            
            # 格式化所有工作表
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                
                # 设置列宽
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
        
        print(f"   ✅ Comprehensive comparison table saved: {excel_path}")
        return excel_path, df
    
    def _find_best_model_in_results(self, results):
        """在实验结果中找到最优模型（基于F1分数）"""
        best_model = None
        best_f1 = -1
        
        print(f"   🔍 Debug: Analyzing results structure...")
        
        # 处理嵌套字典结构
        if isinstance(results, dict):
            # 检查是否是Top-k格式: {k: {temp: {alpha: {depth: result}}}}
            first_key = next(iter(results.keys())) if results else None
            if first_key is not None and isinstance(results[first_key], dict):
                first_sub_key = next(iter(results[first_key].keys())) if results[first_key] else None
                
                print(f"   🔍 Debug: First key: {first_key}, First sub key: {first_sub_key}")
                
                # Top-k格式 (有k层)
                if first_sub_key is not None and isinstance(results[first_key][first_sub_key], dict):
                    print(f"   🔍 Debug: Detected Top-k format")
                    for k_key, k_results in results.items():
                        if isinstance(k_results, dict):
                            for temp_key, temp_results in k_results.items():
                                if isinstance(temp_results, dict):
                                    for alpha_key, alpha_results in temp_results.items():
                                        if isinstance(alpha_results, dict):
                                            for depth_key, result in alpha_results.items():
                                                if result is not None and isinstance(result, dict):
                                                    f1_score = result.get('f1', 0)
                                                    if f1_score > best_f1:
                                                        best_f1 = f1_score
                                                        best_model = result.copy()
                                                        # 使用结果中的实际参数值
                                                        best_model['k'] = result.get('k', k_key)
                                                        best_model['temperature'] = result.get('temperature', temp_key)
                                                        best_model['alpha'] = result.get('alpha', alpha_key)
                                                        best_model['max_depth'] = result.get('max_depth', depth_key)
                                                        print(f"   🔍 Debug: Found better model - k={best_model['k']}, T={best_model['temperature']}, α={best_model['alpha']}, F1={f1_score:.4f}")
                
                # 全特征格式 (无k层): {temp: {alpha: {depth: result}}}
                else:
                    print(f"   🔍 Debug: Detected All-feature format")
                    for temp_key, temp_results in results.items():
                        if isinstance(temp_results, dict):
                            for alpha_key, alpha_results in temp_results.items():
                                if isinstance(alpha_results, dict):
                                    for depth_key, result in alpha_results.items():
                                        if result is not None and isinstance(result, dict):
                                            f1_score = result.get('f1', 0)
                                            if f1_score > best_f1:
                                                best_f1 = f1_score
                                                best_model = result.copy()
                                                # 使用结果中的实际参数值
                                                best_model['temperature'] = result.get('temperature', temp_key)
                                                best_model['alpha'] = result.get('alpha', alpha_key)
                                                best_model['max_depth'] = result.get('max_depth', depth_key)
                                                print(f"   🔍 Debug: Found better model - T={best_model['temperature']}, α={best_model['alpha']}, F1={f1_score:.4f}")
        
        # 处理列表结构（兼容其他可能的数据格式）
        elif isinstance(results, list):
            for result in results:
                if result is not None and isinstance(result, dict):
                    f1_score = result.get('f1', 0)
                    if f1_score > best_f1:
                        best_f1 = f1_score
                        best_model = result
        
        if best_model:
            print(f"   ✅ Debug: Best model found - F1={best_f1:.4f}")
        else:
            print(f"   ❌ Debug: No valid model found in results")
        
        return best_model
    
    def create_master_results_table(self, teacher_models, baseline_results, 
                                  all_feature_distillation_results, top_k_distillation_results):
        """创建主要结果表格 - 记录各数据集的最优模型"""
        print(f"\n📊 Creating Master Results Table...")
        
        master_results = []
        
        for dataset_name in ['uci', 'german', 'australian']:
            dataset_upper = dataset_name.upper()
            
            # 1. 教师模型
            teacher_info = teacher_models[dataset_name]
            test_metrics = teacher_info.get('test_metrics', {})
            master_results.append({
                'Dataset': dataset_upper,
                'Model_Type': 'Teacher Model',
                'Architecture': teacher_info['model_type'],
                'Feature_Selection': 'All Features',
                'Feature_Count': len(teacher_info.get('feature_names', [])) if 'feature_names' in teacher_info else 'N/A',
                'Accuracy': test_metrics.get('accuracy', 'N/A'),
                'Precision': test_metrics.get('precision', 'N/A'),
                'Recall': test_metrics.get('recall', 'N/A'),
                'F1_Score': test_metrics.get('f1_score', 'N/A'),
                'Temperature': 'N/A',
                'Alpha': 'N/A',
                'Hyperparameters': 'Default Neural Network',
                'Notes': 'Neural Network Teacher Model'
            })
            
            # 2. 基础决策树（不使用知识蒸馏）
            if dataset_name in baseline_results:
                baseline_info = baseline_results[dataset_name]
                hyperparams_str = ', '.join([f"{k}={v}" for k, v in baseline_info['hyperparameters'].items()])
                master_results.append({
                    'Dataset': dataset_upper,
                    'Model_Type': 'Baseline Decision Tree',
                    'Architecture': 'Decision Tree',
                    'Feature_Selection': 'All Features',
                    'Feature_Count': baseline_info['feature_count'],
                    'Accuracy': baseline_info['accuracy'],
                    'Precision': baseline_info['precision'],
                    'Recall': baseline_info['recall'],
                    'F1_Score': baseline_info['f1'],
                    'Temperature': 'N/A',
                    'Alpha': 'N/A',
                    'Hyperparameters': hyperparams_str,
                    'Notes': 'Baseline without Knowledge Distillation'
                })
            
            # 3. 全特征知识蒸馏的最优决策树
            if dataset_name in all_feature_distillation_results:
                best_all_feature = self._find_best_result_in_nested_dict(all_feature_distillation_results[dataset_name])
                if best_all_feature:
                    hyperparams_str = ', '.join([f"{k}={v}" for k, v in best_all_feature['hyperparameters'].items()])
                    master_results.append({
                        'Dataset': dataset_upper,
                        'Model_Type': 'All-Feature Distillation Tree',
                        'Architecture': 'Decision Tree',
                        'Feature_Selection': 'All Features',
                        'Feature_Count': best_all_feature['feature_count'],
                        'Accuracy': best_all_feature['accuracy'],
                        'Precision': best_all_feature['precision'],
                        'Recall': best_all_feature['recall'],
                        'F1_Score': best_all_feature['f1'],
                        'Temperature': best_all_feature['temperature'],
                        'Alpha': best_all_feature['alpha'],
                        'Hyperparameters': hyperparams_str,
                        'Notes': f"Best All-Feature KD (T={best_all_feature['temperature']}, α={best_all_feature['alpha']})"
                    })
            
            # 4. Top-5到Top-10蒸馏的最优决策树
            for k in range(5, 11):
                if (dataset_name in top_k_distillation_results and 
                    k in top_k_distillation_results[dataset_name]):
                    
                    best_top_k = self._find_best_result_in_nested_dict(top_k_distillation_results[dataset_name][k])
                    if best_top_k:
                        hyperparams_str = ', '.join([f"{k_}={v}" for k_, v in best_top_k['hyperparameters'].items()])
                        master_results.append({
                            'Dataset': dataset_upper,
                            'Model_Type': f'Top-{k} Distillation Tree',
                            'Architecture': 'Decision Tree',
                            'Feature_Selection': f'SHAP Top-{k}',
                            'Feature_Count': best_top_k['feature_count'],
                            'Accuracy': best_top_k['accuracy'],
                            'Precision': best_top_k['precision'],
                            'Recall': best_top_k['recall'],
                            'F1_Score': best_top_k['f1'],
                            'Temperature': best_top_k['temperature'],
                            'Alpha': best_top_k['alpha'],
                            'Hyperparameters': hyperparams_str,
                            'Notes': f"Best Top-{k} KD (T={best_top_k['temperature']}, α={best_top_k['alpha']})"
                        })
        
        # 创建DataFrame
        master_df = pd.DataFrame(master_results)
        
        # 按数据集和准确率排序
        master_df = master_df.sort_values(['Dataset', 'Accuracy'], ascending=[True, False])
        
        # 保存到Excel
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = f"{self.results_dir}/master_results_table_{timestamp}.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # 主表 - 所有最优结果
            master_df.to_excel(writer, sheet_name='Master_Results', index=False)
            
            # 按数据集分组
            for dataset in ['UCI', 'GERMAN', 'AUSTRALIAN']:
                dataset_df = master_df[master_df['Dataset'] == dataset]
                dataset_df.to_excel(writer, sheet_name=f'{dataset}_Results', index=False)
            
            # 模型类型对比
            model_comparison = master_df.pivot_table(
                index=['Dataset', 'Model_Type'], 
                values=['Accuracy', 'Precision', 'Recall', 'F1_Score'],
                aggfunc='first'
            ).reset_index()
            model_comparison.to_excel(writer, sheet_name='Model_Comparison', index=False)
            
            # 最佳性能排序（所有数据集）
            best_overall = master_df.nlargest(15, 'Accuracy')
            best_overall.to_excel(writer, sheet_name='Top_15_Models', index=False)
        
        print(f"   ✅ Master results table created: {excel_path}")
        print(f"   📋 Total models recorded: {len(master_df)}")
        
        # 显示各数据集最佳模型摘要
        print(f"\n🏆 Best Models Summary:")
        for dataset in ['UCI', 'GERMAN', 'AUSTRALIAN']:
            dataset_best = master_df[master_df['Dataset'] == dataset].iloc[0]
            print(f"   • {dataset}: {dataset_best['Model_Type']} - Accuracy: {dataset_best['Accuracy']:.4f}")
        
        return excel_path, master_df
    
    def _find_best_result_in_nested_dict(self, nested_results):
        """在嵌套字典中找到最佳结果（按准确率）- 支持5层嵌套"""
        best_result = None
        best_accuracy = 0.0
        
        def recursive_search(obj):
            nonlocal best_result, best_accuracy
            
            if isinstance(obj, dict):
                if 'accuracy' in obj and obj['accuracy'] > best_accuracy:
                    best_accuracy = obj['accuracy']
                    best_result = obj
                else:
                    for value in obj.values():
                        if value is not None:  # 跳过None值
                            recursive_search(value)
            elif isinstance(obj, list):
                for item in obj:
                    if item is not None:
                        recursive_search(item)
        
        recursive_search(nested_results)
        return best_result
        """创建全面的实验结果表格"""
        print(f"\n📊 Creating comprehensive results table...")
        
        results_data = []
        
        # 教师模型结果
        print(f"   📋 Processing teacher model results...")
        for dataset_name, model_info in teacher_models.items():
            results_data.append({
                'Dataset': dataset_name.upper(),
                'Model_Type': 'Teacher',
                'Architecture': model_info['model_type'],
                'k_features': 'All',
                'Temperature': 'N/A',
                'Alpha': 'N/A',
                'Accuracy': model_info['accuracy'],
                'Precision': model_info.get('precision', 'N/A'),
                'Recall': model_info.get('recall', 'N/A'),
                'F1_Score': model_info.get('f1', 'N/A'),
                'Hyperparameters': 'Default',
                'Feature_Selection_Method': 'None'
            })
        
        # 知识蒸馏结果
        print(f"   🧠 Processing knowledge distillation results...")
        total_student_experiments = 0
        
        for dataset_name, dataset_results in distillation_results.items():
            for k, k_results in dataset_results.items():
                for temp, temp_results in k_results.items():
                    for alpha, alpha_result in temp_results.items():
                        if alpha_result is not None:
                            total_student_experiments += 1
                            
                            # 处理超参数信息
                            hyperparams = alpha_result.get('hyperparameters', {})
                            if isinstance(hyperparams, dict):
                                hyperparams_str = ', '.join([f"{k}={v}" for k, v in hyperparams.items()])
                            else:
                                hyperparams_str = str(hyperparams)
                            
                            results_data.append({
                                'Dataset': dataset_name.upper(),
                                'Model_Type': 'Student',
                                'Architecture': 'Decision_Tree',
                                'k_features': k,
                                'Temperature': temp,
                                'Alpha': alpha,
                                'Accuracy': alpha_result['accuracy'],
                                'Precision': alpha_result['precision'],
                                'Recall': alpha_result['recall'],
                                'F1_Score': alpha_result['f1'],
                                'Hyperparameters': hyperparams_str,
                                'Feature_Selection_Method': 'SHAP_Top_k'
                            })
        
        print(f"   📈 Processed {len(results_data)} total experiments")
        print(f"     • Teacher models: {len(teacher_models)}")
        print(f"     • Student models: {total_student_experiments}")
        
        # 创建DataFrame
        results_df = pd.DataFrame(results_data)
        
        # 按准确率排序
        results_df = results_df.sort_values('Accuracy', ascending=False)
        
        # 保存到Excel
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = f"{self.results_dir}/comprehensive_results_{timestamp}.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # 主结果表
            results_df.to_excel(writer, sheet_name='All_Results', index=False)
            
            # 按数据集分组的最佳结果
            best_results = results_df.groupby(['Dataset', 'Model_Type']).head(5)
            best_results.to_excel(writer, sheet_name='Best_Results', index=False)
            
            # 教师模型vs最佳学生模型对比
            teacher_results = results_df[results_df['Model_Type'] == 'Teacher']
            student_results = results_df[results_df['Model_Type'] == 'Student'].groupby('Dataset').head(1)
            comparison_df = pd.concat([teacher_results, student_results]).sort_values(['Dataset', 'Model_Type'])
            comparison_df.to_excel(writer, sheet_name='Teacher_vs_Student', index=False)
            
            # 超参数影响分析
            if total_student_experiments > 0:
                student_only = results_df[results_df['Model_Type'] == 'Student'].copy()
                param_analysis = student_only.groupby(['k_features', 'Temperature', 'Alpha'])['Accuracy'].agg(['mean', 'std', 'count']).reset_index()
                param_analysis.to_excel(writer, sheet_name='Parameter_Analysis', index=False)
        
        print(f"   ✅ Results exported to: {excel_path}")
        return excel_path, results_df
    
    def create_performance_visualization(self, master_df):
        """创建性能可视化图表 - 适配master结果表格"""
        print(f"📈 Creating performance visualizations...")
        
        # 创建多子图布局
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 各数据集最佳模型对比
        ax1 = axes[0, 0]
        best_by_dataset = master_df.groupby('Dataset')['Accuracy'].max().reset_index()
        bars1 = ax1.bar(best_by_dataset['Dataset'], best_by_dataset['Accuracy'], 
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax1.set_title('Best Model Accuracy by Dataset', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0.7, 1.0)
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. 模型类型性能对比
        ax2 = axes[0, 1]
        model_performance = master_df.groupby('Model_Type')['Accuracy'].mean().sort_values(ascending=True)
        model_performance.plot(kind='barh', ax=ax2, color='skyblue')
        ax2.set_title('Average Performance by Model Type', fontweight='bold', fontsize=12)
        ax2.set_xlabel('Average Accuracy')
        
        # 3. 特征数量对性能的影响
        ax3 = axes[1, 0]
        feature_data = master_df[master_df['Feature_Count'] != 'N/A'].copy()
        if not feature_data.empty:
            feature_data['Feature_Count'] = pd.to_numeric(feature_data['Feature_Count'])
            feature_performance = feature_data.groupby('Feature_Count')['Accuracy'].mean()
            ax3.scatter(feature_performance.index, feature_performance.values, 
                       s=100, alpha=0.7, color='orange')
            ax3.plot(feature_performance.index, feature_performance.values, 
                    linestyle='--', alpha=0.5, color='orange')
            ax3.set_title('Impact of Feature Count on Performance', fontweight='bold', fontsize=12)
            ax3.set_xlabel('Number of Features')
            ax3.set_ylabel('Average Accuracy')
            ax3.grid(True, alpha=0.3)
        
        # 4. 知识蒸馏vs非蒸馏对比
        ax4 = axes[1, 1]
        distillation_data = master_df.copy()
        distillation_data['Is_Distillation'] = distillation_data['Model_Type'].str.contains('Distillation')
        distill_performance = distillation_data.groupby('Is_Distillation')['Accuracy'].mean()
        
        labels = ['No Distillation', 'With Distillation']
        colors = ['lightcoral', 'lightgreen']
        bars4 = ax4.bar(labels, distill_performance.values, color=colors)
        ax4.set_title('Knowledge Distillation Impact', fontweight='bold', fontsize=12)
        ax4.set_ylabel('Average Accuracy')
        
        # 添加数值标签
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        viz_path = f"{self.results_dir}/master_performance_analysis.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Performance visualization saved: {viz_path}")
        return viz_path
        """创建性能可视化图表"""
        print(f"📈 Creating performance visualizations...")
        
        # 创建多子图布局
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 各数据集最佳模型对比
        ax1 = axes[0, 0]
        best_by_dataset = results_df.groupby('Dataset')['Accuracy'].max().reset_index()
        bars1 = ax1.bar(best_by_dataset['Dataset'], best_by_dataset['Accuracy'], 
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax1.set_title('Best Model Accuracy by Dataset', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0.7, 1.0)
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. 教师vs学生模型性能对比
        ax2 = axes[0, 1]
        teacher_student_comparison = results_df.groupby(['Dataset', 'Model_Type'])['Accuracy'].max().unstack()
        teacher_student_comparison.plot(kind='bar', ax=ax2, color=['#FF6B6B', '#4ECDC4'])
        ax2.set_title('Teacher vs Student Model Performance', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Accuracy')
        ax2.legend(['Student', 'Teacher'])
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Top-k特征数量对性能的影响
        ax3 = axes[1, 0]
        student_data = results_df[results_df['Model_Type'] == 'Student']
        if not student_data.empty:
            k_performance = student_data.groupby('k_features')['Accuracy'].mean()
            ax3.plot(k_performance.index, k_performance.values, marker='o', linewidth=2, markersize=8)
            ax3.set_title('Impact of k (Number of Features) on Performance', fontweight='bold', fontsize=12)
            ax3.set_xlabel('Number of Top-k Features')
            ax3.set_ylabel('Average Accuracy')
            ax3.grid(True, alpha=0.3)
        
        # 4. 温度参数对性能的影响
        ax4 = axes[1, 1]
        if not student_data.empty:
            temp_performance = student_data.groupby('Temperature')['Accuracy'].mean()
            ax4.plot(temp_performance.index, temp_performance.values, marker='s', 
                    linewidth=2, markersize=8, color='orange')
            ax4.set_title('Impact of Temperature on Performance', fontweight='bold', fontsize=12)
            ax4.set_xlabel('Temperature')
            ax4.set_ylabel('Average Accuracy')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        viz_path = f"{self.results_dir}/performance_analysis.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Performance visualization saved: {viz_path}")
        return viz_path
    
    def generate_experiment_summary(self, teacher_models, all_shap_results, top_k_distillation_results, master_df):
        """生成实验总结报告 - 适配master结果表格"""
        print(f"\n📄 Generating experiment summary report...")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 计算统计信息
        total_teacher_models = len(teacher_models)
        total_experiments = len(master_df)
        best_overall_accuracy = master_df['Accuracy'].max()
        best_model_info = master_df[master_df['Accuracy'] == best_overall_accuracy].iloc[0]
        
        # 各数据集最佳结果
        best_by_dataset = {}
        for dataset in ['UCI', 'GERMAN', 'AUSTRALIAN']:
            dataset_results = master_df[master_df['Dataset'] == dataset]
            if not dataset_results.empty:
                best_result = dataset_results.loc[dataset_results['Accuracy'].idxmax()]
                best_by_dataset[dataset] = best_result
        
        summary_text = f"""
信用评分模型优化系统 - 实验总结报告
Credit Scoring Model Optimization System - Experiment Summary Report
==================================================================================

实验时间 | Experiment Time: {timestamp}

📊 实验概览 | Experiment Overview
-----------------------------------------
• 总模型数量 | Total Models: {total_experiments}
• 教师模型数量 | Teacher Models: {total_teacher_models}
• 数据集数量 | Number of Datasets: 3 (UCI, German, Australian)
• 模型类型 | Model Types: 教师模型, 基础决策树, 全特征蒸馏, Top-k蒸馏
• 特征选择方法 | Feature Selection: SHAP Top-k (k=5-10) + All Features

🏆 最佳性能 | Best Performance
-----------------------------------------
• 最高准确率 | Highest Accuracy: {best_overall_accuracy:.4f}
• 最佳模型 | Best Model: {best_model_info['Model_Type']}
• 数据集 | Dataset: {best_model_info['Dataset']}
• 特征选择 | Feature Selection: {best_model_info['Feature_Selection']}
• 配置 | Configuration: T={best_model_info['Temperature']}, α={best_model_info['Alpha']}

📈 各数据集最佳结果 | Best Results by Dataset
-----------------------------------------"""

        for dataset, best_result in best_by_dataset.items():
            summary_text += f"""
{dataset} Dataset:
  • 最佳准确率 | Best Accuracy: {best_result['Accuracy']:.4f}
  • 模型类型 | Model Type: {best_result['Model_Type']}
  • 特征选择 | Feature Selection: {best_result['Feature_Selection']}
  • 配置 | Config: T={best_result['Temperature']}, α={best_result['Alpha']}"""

        # 添加SHAP特征重要性摘要
        summary_text += f"""

🔍 SHAP特征重要性分析 | SHAP Feature Importance Analysis
-----------------------------------------"""
        
        for dataset_name, shap_result in all_shap_results.items():
            top_3_features = shap_result['sorted_features'][:3]
            summary_text += f"""
{dataset_name.upper()} Dataset Top-3 Features:
  1. {top_3_features[0][0]}: {float(top_3_features[0][1]):.4f}
  2. {top_3_features[1][0]}: {float(top_3_features[1][1]):.4f}
  3. {top_3_features[2][0]}: {float(top_3_features[2][1]):.4f}"""

        # 模型类型对比
        summary_text += f"""

🧠 模型性能对比 | Model Performance Comparison
-----------------------------------------"""
        
        model_types = master_df['Model_Type'].unique()
        for model_type in model_types:
            model_data = master_df[master_df['Model_Type'] == model_type]
            avg_accuracy = model_data['Accuracy'].mean()
            best_accuracy = model_data['Accuracy'].max()
            summary_text += f"""
• {model_type}:
  - 平均准确率 | Average Accuracy: {avg_accuracy:.4f}
  - 最佳准确率 | Best Accuracy: {best_accuracy:.4f}"""

        summary_text += f"""

📁 输出文件 | Output Files
-----------------------------------------
• 主要结果表格 | Master Results Table: master_results_table_*.xlsx
• SHAP可视化 | SHAP Visualization: combined_shap_analysis.png
• 性能分析 | Performance Analysis: master_performance_analysis.png
• 模型文件 | Model Files: teacher_model_*.pkl, teacher_model_*.pth
• 实验数据 | Experiment Data: processed_data.pkl, shap_results.pkl

==================================================================================
实验完成 | Experiment Completed Successfully
"""
        
        # 保存总结报告
        summary_path = f"{self.results_dir}/experiment_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        print(f"   ✅ Experiment summary saved: {summary_path}")
        print(summary_text)
        
        return summary_path
        """生成实验总结报告"""
        print(f"\n📄 Generating experiment summary report...")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 计算统计信息
        total_teacher_models = len(teacher_models)
        total_student_experiments = len(results_df[results_df['Model_Type'] == 'Student'])
        best_overall_accuracy = results_df['Accuracy'].max()
        best_model_info = results_df[results_df['Accuracy'] == best_overall_accuracy].iloc[0]
        
        # 各数据集最佳结果
        best_by_dataset = {}
        for dataset in ['UCI', 'GERMAN', 'AUSTRALIAN']:
            dataset_results = results_df[results_df['Dataset'] == dataset]
            if not dataset_results.empty:
                best_result = dataset_results.loc[dataset_results['Accuracy'].idxmax()]
                best_by_dataset[dataset] = best_result
        
        summary_text = f"""
信用评分模型优化系统 - 实验总结报告
Credit Scoring Model Optimization System - Experiment Summary Report
==================================================================================

实验时间 | Experiment Time: {timestamp}

📊 实验概览 | Experiment Overview
-----------------------------------------
• 总教师模型数量 | Total Teacher Models: {total_teacher_models}
• 总学生实验数量 | Total Student Experiments: {total_student_experiments}
• 数据集数量 | Number of Datasets: 3 (UCI, German, Australian)
• 特征选择方法 | Feature Selection: SHAP Top-k (k=5-10)
• 知识蒸馏方法 | Knowledge Distillation: Temperature Scaling + Soft/Hard Label Mixing

🏆 最佳性能 | Best Performance
-----------------------------------------
• 最高准确率 | Highest Accuracy: {best_overall_accuracy:.4f}
• 最佳模型 | Best Model: {best_model_info['Model_Type']} - {best_model_info['Architecture']}
• 数据集 | Dataset: {best_model_info['Dataset']}
• 参数配置 | Configuration: k={best_model_info['k_features']}, T={best_model_info['Temperature']}, α={best_model_info['Alpha']}

📈 各数据集最佳结果 | Best Results by Dataset
-----------------------------------------"""

        for dataset, best_result in best_by_dataset.items():
            summary_text += f"""
{dataset} Dataset:
  • 最佳准确率 | Best Accuracy: {best_result['Accuracy']:.4f}
  • 模型类型 | Model Type: {best_result['Model_Type']} - {best_result['Architecture']}
  • 配置 | Config: k={best_result['k_features']}, T={best_result['Temperature']}, α={best_result['Alpha']}"""

        # 添加SHAP特征重要性摘要
        summary_text += f"""

🔍 SHAP特征重要性分析 | SHAP Feature Importance Analysis
-----------------------------------------"""
        
        for dataset_name, shap_result in all_shap_results.items():
            top_3_features = shap_result['sorted_features'][:3]
            summary_text += f"""
{dataset_name.upper()} Dataset Top-3 Features:
  1. {top_3_features[0][0]}: {float(top_3_features[0][1]):.4f}
  2. {top_3_features[1][0]}: {float(top_3_features[1][1]):.4f}
  3. {top_3_features[2][0]}: {float(top_3_features[2][1]):.4f}"""

        summary_text += f"""

🧠 知识蒸馏效果分析 | Knowledge Distillation Analysis
-----------------------------------------
• 教师模型平均准确率 | Teacher Model Avg Accuracy: {results_df[results_df['Model_Type'] == 'Teacher']['Accuracy'].mean():.4f}
• 学生模型平均准确率 | Student Model Avg Accuracy: {results_df[results_df['Model_Type'] == 'Student']['Accuracy'].mean():.4f}
• 最佳学生模型准确率 | Best Student Model Accuracy: {results_df[results_df['Model_Type'] == 'Student']['Accuracy'].max():.4f}

📁 输出文件 | Output Files
-----------------------------------------
• 结果表格 | Results Table: comprehensive_results_*.xlsx
• SHAP可视化 | SHAP Visualization: combined_shap_analysis.png
• 性能分析 | Performance Analysis: performance_analysis.png
• 模型文件 | Model Files: teacher_model_*.pkl, teacher_model_*.pth
• 实验数据 | Experiment Data: processed_data.pkl, shap_results.pkl, distillation_results.pkl

==================================================================================
实验完成 | Experiment Completed Successfully
"""
        
        # 保存总结报告
        summary_path = f"{self.results_dir}/experiment_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        print(f"   ✅ Experiment summary saved: {summary_path}")
        print(summary_text)
        
        return summary_path
