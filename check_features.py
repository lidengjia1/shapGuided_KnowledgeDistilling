"""
检查每个数据集的特征数量
"""
from data_preprocessing import DataPreprocessor

def check_dataset_features():
    """检查每个数据集的特征数量"""
    preprocessor = DataPreprocessor()
    all_data = preprocessor.process_all_datasets()
    
    for dataset_name, data_dict in all_data.items():
        n_features = len(data_dict['feature_names'])
        print(f"{dataset_name.upper()} Dataset:")
        print(f"  特征数量: {n_features}")
        print(f"  建议 k 范围: 5 到 {n_features}")
        print()

if __name__ == "__main__":
    check_dataset_features()