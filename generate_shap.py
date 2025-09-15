#!/usr/bin/env python3
"""
生成改进的SHAP图
"""

from data_preprocessing import DataPreprocessor
from shap_analysis import SHAPAnalyzer

def main():
    print("🚀 Generating improved SHAP visualization...")
    
    # 1. 处理数据
    print("📊 Processing datasets...")
    dp = DataPreprocessor()
    processed_data = dp.process_all_datasets()
    
    # 2. 创建SHAP分析器
    print("🔧 Creating SHAP analyzer...")
    analyzer = SHAPAnalyzer(processed_data)
    
    # 3. 训练决策树
    print("🌳 Training decision trees...")
    analyzer.train_decision_trees()
    
    # 4. 计算每个数据集的SHAP值
    print("🔍 Computing SHAP values...")
    all_results = {}
    for dataset_name in ['german', 'australian', 'uci']:
        print(f"   Computing for {dataset_name}...")
        all_results[dataset_name] = analyzer.compute_shap_values(dataset_name)
    
    # 5. 生成改进的可视化图
    print("📊 Creating improved visualization...")
    viz_path = analyzer.create_combined_shap_visualization(all_results)
    
    print(f"✅ Improved SHAP visualization saved to: {viz_path}")
    print("   Features:")
    print("   • Top 20 features for each dataset")
    print("   • Beautiful gradient colors")
    print("   • Enhanced layout and readability")

if __name__ == "__main__":
    main()