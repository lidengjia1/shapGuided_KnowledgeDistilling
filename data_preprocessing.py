"""
数据预处理模块
Data Preprocessing Module
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# 设置随机种子
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

class DataPreprocessor:
    """数据预处理类"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_mappings = self._create_feature_mappings()
    
    def _create_feature_mappings(self):
        """创建真实特征名映射"""
        return {
            'german': {
                'Status': 'Account_Status',
                'Duration': 'Loan_Duration_Months', 
                'Credit_history': 'Credit_History',
                'Purpose': 'Loan_Purpose',
                'Credit_amount': 'Credit_Amount',
                'Savings': 'Savings_Account',
                'Employment': 'Employment_Duration',
                'Installment_rate': 'Installment_Rate',
                'Personal_status': 'Personal_Status_Sex',
                'Other_parties': 'Other_Debtors',
                'Residence_since': 'Residence_Duration',
                'Property_magnitude': 'Property_Type',
                'Age': 'Age_Years',
                'Other_payment_plans': 'Other_Installment_Plans',
                'Housing': 'Housing_Type',
                'Existing_credits': 'Existing_Credits_Count',
                'Job': 'Job_Type',
                'Num_dependents': 'Dependents_Count',
                'Own_telephone': 'Telephone_Owner',
                'Foreign_worker': 'Foreign_Worker'
            },
            'australian': {f'feature_{i}': f'Feature_{i+1}' for i in range(14)},
            'uci': {
                'X1': 'Credit_Limit',
                'X2': 'Gender', 
                'X3': 'Education',
                'X4': 'Marriage',
                'X5': 'Age',
                'X6': 'Payment_Status_Sep',
                'X7': 'Payment_Status_Aug',
                'X8': 'Payment_Status_Jul',
                'X9': 'Payment_Status_Jun',
                'X10': 'Payment_Status_May',
                'X11': 'Payment_Status_Apr',
                'X12': 'Bill_Amount_Sep',
                'X13': 'Bill_Amount_Aug',
                'X14': 'Bill_Amount_Jul',
                'X15': 'Bill_Amount_Jun',
                'X16': 'Bill_Amount_May',
                'X17': 'Bill_Amount_Apr',
                'X18': 'Payment_Amount_Sep',
                'X19': 'Payment_Amount_Aug',
                'X20': 'Payment_Amount_Jul',
                'X21': 'Payment_Amount_Jun',
                'X22': 'Payment_Amount_May',
                'X23': 'Payment_Amount_Apr'
            }
        }
    
    def load_and_preprocess_data(self):
        """加载并预处理三个数据集"""
        print("🔄 Loading and preprocessing datasets...")
        
        # 1. German Credit Dataset
        german_df = pd.read_csv('data/german_credit.csv')
        print(f"German Credit original shape: {german_df.shape}")
        
        # 处理分类变量
        categorical_cols = ['Status', 'Credit_history', 'Purpose', 'Savings', 'Employment', 
                          'Personal_status', 'Other_parties', 'Property_magnitude', 
                          'Other_payment_plans', 'Housing', 'Job', 'Own_telephone', 'Foreign_worker']
        
        german_processed = german_df.copy()
        for col in categorical_cols:
            if col in german_processed.columns:
                le = LabelEncoder()
                german_processed[col] = le.fit_transform(german_processed[col].astype(str))
                self.encoders[f'german_{col}'] = le
        
        # 转换标签：1->0, 2->1
        german_processed['Class'] = german_processed['Class'] - 1
        
        # 2. Australian Credit Dataset
        australian_df = pd.read_csv('data/australian_credit.csv')
        print(f"Australian Credit original shape: {australian_df.shape}")
        australian_processed = australian_df.copy()
        
        # 3. UCI Credit Dataset
        uci_df = pd.read_excel('data/uci_credit.xls')
        print(f"UCI Credit original shape: {uci_df.shape}")
        
        # 去除第一行（标题行）和ID列
        uci_df = uci_df.iloc[1:].reset_index(drop=True)
        uci_df = uci_df.drop('Unnamed: 0', axis=1, errors='ignore')
        
        # 重命名列名
        feature_cols = [f'X{i}' for i in range(1, len(uci_df.columns))]
        uci_df.columns = feature_cols + ['Y']
        
        # 确保数据类型正确
        for col in uci_df.columns:
            uci_df[col] = pd.to_numeric(uci_df[col], errors='coerce')
        
        # 删除包含NaN的行
        uci_df = uci_df.dropna().reset_index(drop=True)
        
        # 标签列重命名为Class
        uci_processed = uci_df.rename(columns={'Y': 'Class'})
        
        print(f"German Credit processed shape: {german_processed.shape}")
        print(f"Australian Credit processed shape: {australian_processed.shape}")
        print(f"UCI Credit processed shape: {uci_processed.shape}")
        
        return {
            'german': german_processed,
            'australian': australian_processed,
            'uci': uci_processed
        }
    
    def split_and_scale_data(self, df, dataset_name):
        """划分并标准化数据 (6:2:2)"""
        X = df.drop('Class', axis=1).values
        y = df['Class'].values
        
        # 第一次划分：训练集(60%) vs 临时集(40%)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=RANDOM_SEED, stratify=y
        )
        
        # 第二次划分：验证集(20%) vs 测试集(20%)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=y_temp
        )
        
        # 标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers[dataset_name] = scaler
        
        # 获取原始特征名
        original_feature_names = list(df.drop('Class', axis=1).columns)
        
        # 映射到真实特征名
        feature_mapping = self.feature_mappings.get(dataset_name, {})
        real_feature_names = [feature_mapping.get(name, name) for name in original_feature_names]
        
        print(f"{dataset_name} dataset split:")
        print(f"  Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")
        print(f"  Class distribution - Train: {np.bincount(y_train)}, Val: {np.bincount(y_val)}, Test: {np.bincount(y_test)}")
        
        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': real_feature_names,
            'original_feature_names': original_feature_names
        }
    
    def process_all_datasets(self):
        """处理所有数据集并返回处理后的数据"""
        print("📊 Processing all datasets...")
        
        # 加载和预处理原始数据
        raw_datasets = self.load_and_preprocess_data()
        
        # 处理每个数据集
        processed_datasets = {}
        for dataset_name, df in raw_datasets.items():
            print(f"\n🔧 Processing {dataset_name} dataset...")
            processed_datasets[dataset_name] = self.split_and_scale_data(df, dataset_name)
        
        print("\n✅ All datasets processed successfully!")
        return processed_datasets
