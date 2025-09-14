"""
神经网络模型模块 - 更新版
Neural Network Models Module - Updated Version
使用PyTorch实现的信用评分神经网络
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import time
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CreditNet(nn.Module):
    """信用评分神经网络模型"""
    
    def __init__(self, input_dim, dataset_type='german'):
        super(CreditNet, self).__init__()
        
        # 根据不同数据集调整网络结构
        if dataset_type == 'german':
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
        elif dataset_type == 'australian':
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        elif dataset_type == 'uci':
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.4),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
    
    def forward(self, x):
        return self.layers(x)

class NeuralNetworkTrainer:
    """神经网络训练器"""
    
    def __init__(self, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train_model(self, model, train_loader, val_loader, criterion, optimizer, num_epochs=100, patience=10):
        """训练模型"""
        best_acc = 0.0
        patience_counter = 0
        train_losses = []
        val_accuracies = []
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1).float())
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
            
            # 验证
            model.eval()
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    preds = (outputs > 0.5).float()
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            
            epoch_loss = running_loss / len(train_loader.dataset)
            val_acc = accuracy_score(val_labels, val_preds)
            
            train_losses.append(epoch_loss)
            val_accuracies.append(val_acc)
            
            # 早停机制
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                # 保存最佳模型状态
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f'     Early stopping at epoch {epoch+1}')
                break
            
            if (epoch + 1) % 20 == 0:
                print(f'     Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # 恢复最佳模型状态
        model.load_state_dict(best_model_state)
        return model, train_losses, val_accuracies, best_acc
    
    def test_model(self, model, test_loader):
        """测试模型"""
        model.eval()
        test_preds = []
        test_labels = []
        test_probs = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                preds = (outputs > 0.5).float()
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
                test_probs.extend(outputs.cpu().numpy())
        
        # 计算评估指标
        accuracy = accuracy_score(test_labels, test_preds)
        precision = precision_score(test_labels, test_preds, average='weighted', zero_division=0)
        recall = recall_score(test_labels, test_preds, average='weighted', zero_division=0)
        f1 = f1_score(test_labels, test_preds, average='weighted', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': test_preds,
            'probabilities': test_probs,
            'true_labels': test_labels
        }
    
    def create_data_loaders(self, data_dict, batch_size=32):
        """创建数据加载器"""
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(data_dict['X_train'])
        y_train_tensor = torch.LongTensor(data_dict['y_train'])
        X_val_tensor = torch.FloatTensor(data_dict['X_val'])
        y_val_tensor = torch.LongTensor(data_dict['y_val'])
        X_test_tensor = torch.FloatTensor(data_dict['X_test'])
        y_test_tensor = torch.LongTensor(data_dict['y_test'])
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader

def create_teacher_model(dataset_name, processed_data):
    """创建并训练教师模型"""
    print(f"   📚 Training {dataset_name.upper()} teacher model...")
    
    # 获取数据
    data_dict = processed_data[dataset_name]
    input_dim = data_dict['X_train'].shape[1]
    
    # 创建训练器
    trainer = NeuralNetworkTrainer(device)
    
    # 创建数据加载器
    if dataset_name == 'uci':
        batch_size = 64
        learning_rate = 0.0005
        num_epochs = 200
        patience = 20
    elif dataset_name == 'australian':
        batch_size = 32
        learning_rate = 0.001
        num_epochs = 150
        patience = 15
    else:  # german
        batch_size = 32
        learning_rate = 0.001
        num_epochs = 100
        patience = 10
    
    train_loader, val_loader, test_loader = trainer.create_data_loaders(data_dict, batch_size)
    
    # 创建模型
    model = CreditNet(input_dim, dataset_name).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练模型
    start_time = time.time()
    trained_model, train_losses, val_accuracies, best_val_acc = trainer.train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs, patience
    )
    training_time = time.time() - start_time
    
    # 测试模型
    test_results = trainer.test_model(trained_model, test_loader)
    
    # 计算模型大小
    model_size = sum(p.numel() * p.element_size() for p in trained_model.parameters()) / 1024  # KB
    
    print(f"     ✅ {dataset_name.upper()}: PyTorch Neural Network - Accuracy: {test_results['accuracy']:.4f}, F1: {test_results['f1']:.4f}")
    
    return {
        'model': trained_model,
        'model_type': 'PyTorch Neural Network',
        'accuracy': test_results['accuracy'],
        'precision': test_results['precision'],
        'recall': test_results['recall'],
        'f1': test_results['f1'],
        'predictions': test_results['predictions'],
        'probabilities': test_results['probabilities'],
        'true_labels': test_results['true_labels'],
        'training_time': training_time,
        'model_size': model_size,
        'best_val_accuracy': best_val_acc,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'feature_names': data_dict['feature_names']
    }

def train_all_teacher_models(processed_data):
    """训练所有教师模型"""
    print("🧠 Phase 2: Teacher Model Training")
    print("   Training neural network teacher models...")
    print(f"   🔧 Using device: {device}")
    
    teacher_models = {}
    
    for dataset_name in ['uci', 'german', 'australian']:
        if dataset_name in processed_data:
            teacher_models[dataset_name] = create_teacher_model(dataset_name, processed_data)
        else:
            print(f"   ⚠️ {dataset_name.upper()} dataset not found in processed data")
    
    print("   ✅ Teacher model training completed")
    for dataset_name, model_info in teacher_models.items():
        print(f"     • {dataset_name.upper()}: {model_info['model_type']} - Accuracy: {model_info['accuracy']:.4f}")
    
    return teacher_models

if __name__ == "__main__":
    # 测试神经网络训练
    from data_preprocessing import DataPreprocessor
    
    print("Testing neural network training...")
    
    # 加载和预处理数据
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.process_all_datasets()
    
    # 训练教师模型
    teacher_models = train_all_teacher_models(processed_data)
    
    print("\nTraining completed!")
    for dataset_name, model_info in teacher_models.items():
        print(f"{dataset_name.upper()} Dataset Results:")
        print(f"  Accuracy: {model_info['accuracy']:.4f}")
        print(f"  F1 Score: {model_info['f1']:.4f}")
        print(f"  Training Time: {model_info['training_time']:.2f}s")
        print(f"  Model Size: {model_info['model_size']:.2f}KB")
