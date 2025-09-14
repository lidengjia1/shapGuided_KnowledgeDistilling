"""
测试SHAP修复 - 生成英文特征名的SHAP图像
Test SHAP fixes - Generate SHAP visualization with English feature names
"""

from data_preprocessing import DataPreprocessor
from shap_analysis import SHAPAnalyzer

def test_shap_fixes():
    """测试SHAP修复效果"""
    print("🔧 Testing SHAP fixes...")
    
    # 加载数据
    print("📊 Loading datasets...")
    data_preprocessor = DataPreprocessor()
    
    # 加载三个数据集
    german_data = data_preprocessor.load_german_credit()
    australian_data = data_preprocessor.load_australian_credit()
    uci_data = data_preprocessor.load_uci_credit()
    
    # 组织数据
    processed_data = {
        'german': {
            'X_train': german_data[0],
            'X_val': german_data[1], 
            'X_test': german_data[2],
            'y_train': german_data[3],
            'y_val': german_data[4],
            'y_test': german_data[5],
            'feature_names': data_preprocessor.feature_names['german']
        },
        'australian': {
            'X_train': australian_data[0],
            'X_val': australian_data[1],
            'X_test': australian_data[2], 
            'y_train': australian_data[3],
            'y_val': australian_data[4],
            'y_test': australian_data[5],
            'feature_names': data_preprocessor.feature_names['australian']
        },
        'uci': {
            'X_train': uci_data[0],
            'X_val': uci_data[1],
            'X_test': uci_data[2],
            'y_train': uci_data[3], 
            'y_val': uci_data[4],
            'y_test': uci_data[5],
            'feature_names': data_preprocessor.feature_names['uci']
        }
    }
    
    # 运行SHAP分析
    print("🎯 Running SHAP analysis with English features...")
    shap_analyzer = SHAPAnalyzer(processed_data)
    
    # 训练决策树模型
    print("🌳 Training decision trees...")
    shap_analyzer.train_decision_trees()
    
    # 计算SHAP值（只测试一个小数据集）
    print("🔍 Computing SHAP values for German dataset...")
    german_shap = shap_analyzer.compute_shap_values('german')
    
    print("🔍 Computing SHAP values for Australian dataset...")
    australian_shap = shap_analyzer.compute_shap_values('australian')
    
    print("🔍 Computing SHAP values for UCI dataset (this may take a while)...")
    uci_shap = shap_analyzer.compute_shap_values('uci')
    
    # 组合结果
    all_shap_results = {
        'german': german_shap,
        'australian': australian_shap,
        'uci': uci_shap
    }
    
    # 创建可视化
    print("📊 Creating English SHAP visualization...")
    visualization_path = shap_analyzer.create_combined_shap_visualization(all_shap_results)
    
    print("✅ SHAP fixes test completed!")
    print(f"   📊 Visualization saved: {visualization_path}")
    print("   📋 Check the image for:")
    print("      - English titles and labels")
    print("      - German, Australian, UCI order")
    print("      - Proper feature names without duplicates")

if __name__ == "__main__":
    test_shap_fixes()