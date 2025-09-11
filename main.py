"""
信用评分模型优化系统 - 主程序
Credit Scoring Model Optimization System - Main Program

基于SHAP特征重要性分析和知识蒸馏的信用评分模型优化系统
模块化架构，支持扩展参数组合和改进的用户体验
"""

import os
import warnings
import torch
import numpy as np
from tqdm import tqdm

# 解决中文路径编码问题
import locale
import tempfile
import multiprocessing
try:
    locale.setlocale(locale.LC_ALL, 'C')
except:
    pass

# 设置临时目录为英文路径避免编码问题
temp_dir = "C:\\temp_ml"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
os.environ['TEMP'] = temp_dir
os.environ['TMP'] = temp_dir

# 导入自定义模块
from data_preprocessing import DataPreprocessor
from neural_models import TeacherModelTrainer
from shap_analysis import SHAPAnalyzer
from distillation_module import KnowledgeDistillator
from experiment_manager import ExperimentManager
from tree_rules_analyzer import DecisionTreeRulesAnalyzer

warnings.filterwarnings('ignore')

def main():
    """主函数 - 运行完整的信用评分模型优化系统"""
    
    print("="*80)
    print("🎯 信用评分模型优化系统 | Credit Scoring Model Optimization System")
    print("   基于SHAP特征重要性分析和知识蒸馏 | SHAP + Knowledge Distillation")
    print("   增强版 - 支持决策树深度参数和消融实验分析")
    print("="*80)
    print("📊 实验参数配置:")
    print("   • Top-k特征: k=5,6,7,8")
    print("   • 温度参数: T=1,2,3,4,5") 
    print("   • 混合权重: α=0.0,0.2,0.4,0.6,0.8,1.0")
    print("   • 决策树深度: D=4,5,6,7,8")
    print("   • 决策树学生模型知识蒸馏")
    print("="*80)
    
    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    
    try:
        # ========================
        # 1. 数据预处理阶段
        # ========================
        print(f"\n🔄 Phase 1: Data Preprocessing")
        print(f"   Loading and preprocessing datasets...")
        
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.process_all_datasets()
        
        print(f"   ✅ Data preprocessing completed")
        for dataset_name, data_dict in processed_data.items():
            print(f"     • {dataset_name.upper()}: {data_dict['X_train'].shape[0]} train, {data_dict['X_test'].shape[0]} test samples")
        
        # ========================
        # 2. 教师模型训练阶段
        # ========================
        print(f"\n🧠 Phase 2: Teacher Model Training")
        print(f"   Training neural network teacher models...")
        
        trainer = TeacherModelTrainer()
        teacher_models = trainer.train_all_teacher_models(processed_data)
        
        print(f"   ✅ Teacher model training completed")
        for dataset_name, model_info in teacher_models.items():
            accuracy = model_info['test_metrics']['accuracy']
            print(f"     • {dataset_name.upper()}: {model_info['model_type']} - Accuracy: {accuracy:.4f}")
        
        # ========================
        # 3. SHAP特征重要性分析
        # ========================
        print(f"\n🔍 Phase 3: SHAP Feature Importance Analysis")
        print(f"   Training decision trees for SHAP analysis...")
        
        shap_analyzer = SHAPAnalyzer(processed_data)
        
        # 先训练决策树模型用于SHAP分析
        shap_analyzer.train_decision_trees()
        
        # 计算SHAP值
        all_shap_results = {}
        for dataset_name in ['uci', 'german', 'australian']:
            all_shap_results[dataset_name] = shap_analyzer.compute_shap_values(dataset_name, top_k_range=(5, 8))
        
        # 创建组合SHAP可视化
        shap_viz_path = shap_analyzer.create_combined_shap_visualization(all_shap_results)
        
        print(f"   ✅ SHAP analysis completed")
        print(f"     • Combined visualization: {shap_viz_path}")
        
        # ========================
        # 初始化知识蒸馏器
        # ========================
        distillator = KnowledgeDistillator(teacher_models, processed_data, all_shap_results)
        
        # ========================
        # 4. 基础决策树训练（对比基准）
        # ========================
        print(f"\n🌳 Phase 4: Baseline Decision Tree Training")
        print(f"   Training baseline decision trees for comparison...")
        
        baseline_results = {}
        for dataset_name in ['uci', 'german', 'australian']:
            baseline_results[dataset_name] = distillator.train_baseline_decision_tree(dataset_name)
        
        print(f"   ✅ Baseline decision tree training completed")
        for dataset_name, result in baseline_results.items():
            print(f"     • {dataset_name.upper()}: Accuracy: {result['accuracy']:.4f}")
        
        # ========================
        # 5. 全特征知识蒸馏实验
        # ========================
        print(f"\n🌟 Phase 5: All-Feature Knowledge Distillation")
        print(f"   Running all-feature distillation experiments without Optuna...")
        
        all_feature_distillation_results = distillator.run_all_feature_distillation(
            dataset_names=['uci', 'german', 'australian'],
            temperature_range=[1, 2, 3, 4, 5],   # Temperature: 1-5  
            alpha_range=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # Alpha: 0.1-0.9 (步长0.1)
            max_depth_range=[4, 5, 6, 7, 8]  # Depth: 4-8
        )
        
        print(f"   ✅ All-feature knowledge distillation completed")
        
        # ========================
        # 6. Top-k知识蒸馏实验
        # ========================
        print(f"\n🧪 Phase 6: Top-k Knowledge Distillation Experiments")
        print(f"   Running comprehensive distillation experiments without Optuna...")
        
        # Top-k特征蒸馏实验
        top_k_distillation_results = distillator.run_comprehensive_distillation(
            dataset_names=['uci', 'german', 'australian'],
            k_range=(5, 8),            # k: 5-8
            temperature_range=[1, 2, 3, 4, 5],   # Temperature: 1-5  
            alpha_range=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # Alpha: 0.1-0.9 (步长0.1)
            max_depth_range=[4, 5, 6, 7, 8]        # Depth: 4-8
        )
        
        print(f"   ✅ Top-k knowledge distillation experiments completed")
        
        # ========================
        # 7. 结果汇总和导出
        # ========================
        print(f"\n📊 Phase 7: Results Analysis and Export")
        print(f"   Generating comprehensive results and visualizations...")
        
        experiment_manager = ExperimentManager()
        
        # 创建综合对比表格 - 四种模型的性能对比
        comparison_excel_path, comparison_df = experiment_manager.create_comprehensive_comparison_table(
            teacher_models, baseline_results, all_feature_distillation_results, top_k_distillation_results
        )
        
        # 创建主要结果表格 - 详细的实验记录
        master_excel_path, master_df = experiment_manager.create_master_results_table(
            teacher_models, baseline_results, all_feature_distillation_results, top_k_distillation_results
        )
        
        # 保存所有模型和数据
        saved_paths = experiment_manager.save_models_and_data(
            teacher_models, processed_data, all_shap_results, top_k_distillation_results
        )
        
        # 创建性能可视化
        viz_path = experiment_manager.create_performance_visualization(master_df)
        
        # 创建Top-K参数分析图 (2×2布局)
        topk_param_viz_path = experiment_manager.create_topk_parameter_analysis(top_k_distillation_results)
        
        # 提取最优蒸馏树的决策规则
        rules_extractor = DecisionTreeRulesAnalyzer()
        rules_excel_path, best_trees_info = rules_extractor.extract_best_distillation_tree_rules(
            top_k_distillation_results, processed_data
        )
        
        # 生成决策树文本表示
        tree_text_representations = rules_extractor.generate_tree_text_representation(
            best_trees_info, processed_data
        )
        
        # 生成实验总结 - 使用主要结果
        summary_path = experiment_manager.generate_experiment_summary(
            teacher_models, all_shap_results, top_k_distillation_results, master_df
        )
        
        print(f"\n🎉 System Execution Completed Successfully!")
        print(f"   📁 All results saved to: ./results/")
        print(f"   📊 Model Comparison Excel: {comparison_excel_path}")
        print(f"   📋 Master Excel report: {master_excel_path}")
        print(f"   📈 Performance charts: {viz_path}")
        print(f"   🌳 Decision tree rules: {rules_excel_path}")
        print(f"   📄 Summary report: {summary_path}")
        
        # 显示四种模型对比摘要
        print(f"\n📋 Model Comparison Summary:")
        for dataset in ['UCI', 'GERMAN', 'AUSTRALIAN']:
            dataset_results = comparison_df[comparison_df['Dataset'] == dataset]
            print(f"\n   {dataset} Dataset:")
            for _, row in dataset_results.iterrows():
                print(f"     • {row['Model_Type']}: F1={row['F1_Score']}, Acc={row['Accuracy']}")
        
        # 显示最佳结果摘要
        best_result = master_df.loc[master_df['Accuracy'].idxmax()]
        print(f"\n🏆 Overall Best Model Performance:")
        print(f"   • Dataset: {best_result['Dataset']}")
        print(f"   • Model: {best_result['Model_Type']} - {best_result['Architecture']}")
        print(f"   • Accuracy: {best_result['Accuracy']:.4f}")
        print(f"   • Feature Selection: {best_result['Feature_Selection']}")
        if best_result['Temperature'] != 'N/A':
            print(f"   • Configuration: T={best_result['Temperature']}, α={best_result['Alpha']}")
        
        # 显示最优蒸馏树信息
        print(f"\n🌳 Best Distillation Trees by Dataset:")
        for dataset_name, tree_info in best_trees_info.items():
            print(f"   • {dataset_name.upper()}:")
            print(f"     - Configuration: k={tree_info['k']}, T={tree_info['temperature']}, α={tree_info['alpha']}, D={tree_info['max_depth']}")
            print(f"     - Performance: Accuracy={tree_info['accuracy']:.4f}, F1={tree_info['f1']:.4f}")
            print(f"     - Features: {len(tree_info['feature_names'])} selected")
        
    except Exception as e:
        print(f"\n❌ Error during system execution: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    main()
