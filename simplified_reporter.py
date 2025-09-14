"""
简化Excel报告生成器
Simplified Excel Report Generator
"""

import pandas as pd
import os
from datetime import datetime

class SimplifiedReporter:
    """简化Excel报告生成器"""
    
    def __init__(self, results_dir='results'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    def extract_best_model(self, results):
        """从复杂结果结构中提取最佳模型"""
        if not results:
            return None
            
        print(f"   🔍 Debug: Analyzing results structure...")
        print(f"   🔍 Debug: Results type: {type(results)}")
        
        # 处理嵌套字典结构
        if isinstance(results, dict):
            print(f"   🔍 Debug: Results keys: {list(results.keys())}")
            
            # 检查是否是简单的 best 格式: {'best': {...}}
            if 'best' in results and isinstance(results['best'], dict):
                print(f"   🔍 Debug: Detected simple 'best' format")
                best_result = results['best']
                
                # 检查best结果是否包含评估指标
                if 'f1' in best_result or 'accuracy' in best_result:
                    print(f"   🔍 Debug: Found metrics in best result")
                    best_model = best_result.copy()
                    best_f1 = best_result.get('f1', best_result.get('accuracy', 0))
                    print(f"   ✅ Debug: Best model found from simple format - F1/Acc={best_f1:.4f}")
                    return best_model
                    
                # 检查best结果是否有嵌套的model
                elif 'model' in best_result and isinstance(best_result['model'], dict):
                    print(f"   🔍 Debug: Found nested model in best result")
                    model_result = best_result['model']
                    if 'f1' in model_result or 'accuracy' in model_result:
                        best_model = model_result.copy()
                        best_f1 = model_result.get('f1', model_result.get('accuracy', 0))
                        print(f"   ✅ Debug: Best model found from nested format - F1/Acc={best_f1:.4f}")
                        return best_model
            
            # 原有的复杂格式解析逻辑
            first_key = next(iter(results.keys())) if results else None
            if first_key is not None and isinstance(results[first_key], dict):
                first_sub_key = next(iter(results[first_key].keys())) if results[first_key] else None
                
                print(f"   🔍 Debug: First key: {first_key}, First sub key: {first_sub_key}")
                
                # Top-k格式 (有k层)
                if first_sub_key is not None and isinstance(results[first_key][first_sub_key], dict):
                    print(f"   🔍 Debug: Detected Top-k format")
                    best_f1 = -1
                    best_model = None
                    
                    for k_value, temp_results in results.items():
                        for temp, alpha_results in temp_results.items():
                            for alpha, depth_results in alpha_results.items():
                                for depth, result in depth_results.items():
                                    if isinstance(result, dict) and ('f1' in result or 'accuracy' in result):
                                        current_f1 = result.get('f1', result.get('accuracy', 0))
                                        if current_f1 > best_f1:
                                            best_f1 = current_f1
                                            best_model = result.copy()
                                            best_model.update({
                                                'k_features': k_value,
                                                'temperature': temp,
                                                'alpha': alpha,
                                                'max_depth': depth
                                            })
                    
                    if best_model:
                        print(f"   ✅ Debug: Best Top-k model found - F1={best_f1:.4f}")
                        return best_model
                
                # 标准格式 (无k层)
                elif first_sub_key is not None:
                    print(f"   🔍 Debug: Detected standard format")
                    best_f1 = -1
                    best_model = None
                    
                    for temp, alpha_results in results.items():
                        for alpha, depth_results in alpha_results.items():
                            for depth, result in depth_results.items():
                                if isinstance(result, dict) and ('f1' in result or 'accuracy' in result):
                                    current_f1 = result.get('f1', result.get('accuracy', 0))
                                    if current_f1 > best_f1:
                                        best_f1 = current_f1
                                        best_model = result.copy()
                                        best_model.update({
                                            'temperature': temp,
                                            'alpha': alpha,
                                            'max_depth': depth
                                        })
                    
                    if best_model:
                        print(f"   ✅ Debug: Best standard model found - F1={best_f1:.4f}")
                        return best_model
        
        # 处理直接的结果格式
        elif isinstance(results, dict) and ('f1' in results or 'accuracy' in results):
            print(f"   ✅ Debug: Direct result format found")
            return results
        
        print(f"   ❌ Debug: No valid model found")
        return None
    
    def generate_simplified_excel_report(self, results, output_file=None):
        """生成简化的Excel报告，每个数据集包含4种核心模型"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{self.results_dir}/simplified_results_{timestamp}.xlsx"
        
        print("\n📊 生成简化Excel报告...")
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # 为每个数据集创建一个工作表
            for dataset_name in ['german_credit', 'uci_credit', 'australian_credit']:
                dataset_results = []
                
                # 1. 教师模型 (Teacher Model)
                if dataset_name in results.get('teacher_models', {}):
                    teacher_data = results['teacher_models'][dataset_name]
                    dataset_results.append({
                        'Model Type': 'Teacher Model',
                        'Model Name': teacher_data.get('model_type', 'N/A'),
                        'Accuracy': teacher_data.get('accuracy', 0),
                        'F1 Score': teacher_data.get('f1', 0),
                        'Precision': teacher_data.get('precision', 0),
                        'Recall': teacher_data.get('recall', 0),
                        'Training Time (s)': teacher_data.get('training_time', 0),
                        'Model Size (KB)': teacher_data.get('model_size', 0),
                        'Notes': 'Optimal model from TabSurvey research'
                    })
                
                # 2. 基线决策树 (Baseline Decision Tree)
                baseline_key = f"{dataset_name}_baseline"
                if baseline_key in results.get('baseline_models', {}):
                    baseline_data = results['baseline_models'][baseline_key]
                    dataset_results.append({
                        'Model Type': 'Student (Decision Tree)',
                        'Model Name': 'Decision Tree',
                        'Accuracy': baseline_data.get('accuracy', 0),
                        'F1 Score': baseline_data.get('f1', 0),
                        'Precision': baseline_data.get('precision', 0),
                        'Recall': baseline_data.get('recall', 0),
                        'Training Time (s)': baseline_data.get('training_time', 0),
                        'Model Size (KB)': baseline_data.get('model_size', 0),
                        'Notes': 'Baseline student model (max_depth=5)'
                    })
                
                # 3. 完整知识蒸馏 (Full Distillation)
                distill_key = f"{dataset_name}_distillation"
                if distill_key in results.get('distillation_results', {}):
                    distill_data = results['distillation_results'][distill_key]
                    # 提取最佳结果
                    best_distill = self.extract_best_model(distill_data)
                    if best_distill:
                        dataset_results.append({
                            'Model Type': 'Full Distillation',
                            'Model Name': 'Distilled Decision Tree',
                            'Accuracy': best_distill.get('accuracy', 0),
                            'F1 Score': best_distill.get('f1', 0),
                            'Precision': best_distill.get('precision', 0),
                            'Recall': best_distill.get('recall', 0),
                            'Training Time (s)': best_distill.get('training_time', 0),
                            'Model Size (KB)': best_distill.get('model_size', 0),
                            'Notes': 'Full knowledge distillation from teacher'
                        })
                
                # 4. Top-k蒸馏 (Top-k Distillation)
                topk_key = f"{dataset_name}_top_k"
                if topk_key in results.get('top_k_results', {}):
                    topk_data = results['top_k_results'][topk_key]
                    # 提取最佳结果
                    best_topk = self.extract_best_model(topk_data)
                    if best_topk:
                        dataset_results.append({
                            'Model Type': 'Top-k Distillation',
                            'Model Name': 'Top-k Distilled Tree',
                            'Accuracy': best_topk.get('accuracy', 0),
                            'F1 Score': best_topk.get('f1', 0),
                            'Precision': best_topk.get('precision', 0),
                            'Recall': best_topk.get('recall', 0),
                            'Training Time (s)': best_topk.get('training_time', 0),
                            'Model Size (KB)': best_topk.get('model_size', 0),
                            'Notes': 'Top-k feature distillation'
                        })
                
                # 创建DataFrame并保存到工作表
                if dataset_results:
                    df = pd.DataFrame(dataset_results)
                    # 确保数值列的格式
                    numeric_cols = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'Training Time (s)', 'Model Size (KB)']
                    for col in numeric_cols:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                        if col in ['Accuracy', 'F1 Score', 'Precision', 'Recall']:
                            df[col] = df[col].round(4)
                        else:
                            df[col] = df[col].round(2)
                    
                    sheet_name = dataset_name.replace('_', ' ').title()
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"   ✅ {sheet_name}: {len(dataset_results)} models")
                else:
                    print(f"   ⚠️ {dataset_name}: No results found")
            
            # 创建总结工作表
            self._create_summary_sheet(writer, results)
        
        print(f"✅ 简化Excel报告已保存到: {output_file}")
        return output_file
    
    def _create_summary_sheet(self, writer, results):
        """创建总结工作表"""
        summary_data = []
        
        for dataset_name in ['german_credit', 'uci_credit', 'australian_credit']:
            # 获取教师模型结果
            teacher_acc = 0
            if dataset_name in results.get('teacher_models', {}):
                teacher_acc = results['teacher_models'][dataset_name].get('accuracy', 0)
            
            # 获取基线模型结果
            baseline_acc = 0
            baseline_key = f"{dataset_name}_baseline"
            if baseline_key in results.get('baseline_models', {}):
                baseline_acc = results['baseline_models'][baseline_key].get('accuracy', 0)
            
            # 获取蒸馏模型结果
            distill_acc = 0
            distill_key = f"{dataset_name}_distillation"
            if distill_key in results.get('distillation_results', {}):
                best_distill = self.extract_best_model(results['distillation_results'][distill_key])
                if best_distill:
                    distill_acc = best_distill.get('accuracy', 0)
            
            # 获取Top-k蒸馏结果
            topk_acc = 0
            topk_key = f"{dataset_name}_top_k"
            if topk_key in results.get('top_k_results', {}):
                best_topk = self.extract_best_model(results['top_k_results'][topk_key])
                if best_topk:
                    topk_acc = best_topk.get('accuracy', 0)
            
            summary_data.append({
                'Dataset': dataset_name.replace('_', ' ').title(),
                'Teacher Model Accuracy': round(teacher_acc, 4),
                'Student Model Accuracy': round(baseline_acc, 4),
                'Full Distillation Accuracy': round(distill_acc, 4),
                'Top-k Distillation Accuracy': round(topk_acc, 4),
                'Improvement (Full)': round(distill_acc - baseline_acc, 4),
                'Improvement (Top-k)': round(topk_acc - baseline_acc, 4),
                'Knowledge Transfer Rate': round((distill_acc / teacher_acc * 100) if teacher_acc > 0 else 0, 2)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        print(f"   ✅ Summary: Performance comparison created")
