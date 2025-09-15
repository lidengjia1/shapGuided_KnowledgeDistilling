"""
测试并发配置
"""
from multiprocessing import cpu_count

def test_concurrent_config():
    """测试并发配置"""
    n_cores = cpu_count()
    n_jobs = max(1, min(n_cores - 1, n_cores))
    
    print(f"🔧 并发配置测试:")
    print(f"   CPU核心数: {n_cores}")
    print(f"   建议并发数: {n_jobs}")
    print(f"   系统保留核心数: 1")
    print(f"   实际使用核心数: {n_jobs}")
    
    # 测试SHAP分析器
    from data_preprocessing import DataPreprocessor
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.process_all_datasets()
    
    from shap_analysis import SHAPAnalyzer
    shap_analyzer = SHAPAnalyzer(processed_data)
    print(f"   SHAP分析器并发数: {shap_analyzer.n_jobs}")
    
    print("✅ 并发配置测试完成")

if __name__ == "__main__":
    test_concurrent_config()