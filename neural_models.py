"""
神经网络模型定义模块
Neural Network Models Module
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# 设置随机种子
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

class MLP_UCI(nn.Module):
    """UCI Credit Dataset - MLP多层感知机"""
    def __init__(self, input_dim):
        super(MLP_UCI, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 输出raw logits，不用Sigmoid
        )

    def forward(self, x):
        return self.net(x)
    
    def predict_proba(self, x):
        """用于推理的概率输出"""
        with torch.no_grad():
            # 确保输入是Tensor
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            elif not isinstance(x, torch.Tensor):
                x = torch.FloatTensor(x)
            
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            
            # 对于二分类，返回两个类别的概率
            prob_positive = probs.numpy().flatten()
            prob_negative = 1 - prob_positive
            
            # 返回形状为 (n_samples, 2) 的概率数组
            return np.column_stack([prob_negative, prob_positive])

class RBF_German(nn.Module):
    """German Credit Dataset - RBF径向基函数网络"""
    def __init__(self, input_dim, num_rbf=50):
        super(RBF_German, self).__init__()
        self.num_rbf = num_rbf
        self.centers = nn.Parameter(torch.randn(num_rbf, input_dim))
        self.beta = nn.Parameter(torch.ones(1)*1.0)
        self.linear = nn.Linear(num_rbf, 1)  # 输出raw logits

    def rbf_layer(self, x):
        x_expanded = x.unsqueeze(1)
        c_expanded = self.centers.unsqueeze(0)
        dist_sq = ((x_expanded - c_expanded)**2).sum(-1)
        return torch.exp(-self.beta * dist_sq)

    def forward(self, x):
        phi = self.rbf_layer(x)
        out = self.linear(phi)
        return out  # raw logits
    
    def predict_proba(self, x):
        """用于推理的概率输出"""
        with torch.no_grad():
            # 确保输入是Tensor
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            elif not isinstance(x, torch.Tensor):
                x = torch.FloatTensor(x)
            
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            
            # 对于二分类，返回两个类别的概率
            prob_positive = probs.numpy().flatten()
            prob_negative = 1 - prob_positive
            
            # 返回形状为 (n_samples, 2) 的概率数组
            return np.column_stack([prob_negative, prob_positive])

class Autoencoder(nn.Module):
    """自编码器"""
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

class AE_MLP_Australian(nn.Module):
    """Australian Credit - 自编码器增强MLP"""
    def __init__(self, input_dim, latent_dim=8):
        super(AE_MLP_Australian, self).__init__()
        self.ae = Autoencoder(input_dim, latent_dim)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 输出raw logits
        )

    def forward(self, x):
        _, z = self.ae(x)
        out = self.classifier(z)
        return out  # raw logits
    
    def predict_proba(self, x):
        """用于推理的概率输出"""
        with torch.no_grad():
            # 确保输入是Tensor
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            elif not isinstance(x, torch.Tensor):
                x = torch.FloatTensor(x)
            
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            
            # 对于二分类，返回两个类别的概率
            prob_positive = probs.numpy().flatten()
            prob_negative = 1 - prob_positive
            
            # 返回形状为 (n_samples, 2) 的概率数组
            return np.column_stack([prob_negative, prob_positive])

class TeacherModelTrainer:
    """教师模型训练器"""
    
    def __init__(self, device='cpu'):
        self.device = device
        
    def create_model(self, dataset_name, input_dim):
        """根据数据集创建对应的模型"""
        if dataset_name == 'uci':
            return MLP_UCI(input_dim)
        elif dataset_name == 'german':
            return RBF_German(input_dim)
        elif dataset_name == 'australian':
            return AE_MLP_Australian(input_dim)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def train_model(self, model, X_train, y_train, X_val, y_val, 
                   epochs=200, lr=1e-3, weight_decay=1e-4):
        """训练教师模型"""
        model = model.to(self.device)
        
        # 计算类别权重以处理不平衡数据
        pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        pos_weight_tensor = torch.FloatTensor([pos_weight]).to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).view(-1, 1).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).view(-1, 1).to(self.device)
        
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        patience = 15
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            optimizer.zero_grad()
            logits = model(X_train_tensor)
            loss = criterion(logits, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            # 验证阶段
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_tensor)
                val_loss = criterion(val_logits, y_val_tensor).item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        # 恢复最佳模型
        model.load_state_dict(best_state)
        return model
    
    def train_teacher_model(self, dataset_name, data_dict):
        """训练教师模型（使用固定架构）"""
        print(f"🎯 Training {dataset_name.upper()} teacher model...")
        
        # 创建模型
        model = self.create_model(dataset_name, data_dict['X_train'].shape[1])
        input_features = data_dict['X_train'].shape[1]
        
        # 获取模型架构名称和参数数量
        architecture_name = self.get_architecture_name(dataset_name)
        model_parameters = sum(p.numel() for p in model.parameters())
        
        import time
        start_time = time.time()
        
        # 训练模型
        model = self.train_model(
            model, data_dict['X_train'], data_dict['y_train'],
            data_dict['X_val'], data_dict['y_val'],
            lr=1e-3, weight_decay=1e-4
        )
        
        training_time = time.time() - start_time
        
        # 评估性能 - 使用最优阈值而不是固定的0.5
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(data_dict['X_test']).to(self.device)
            logits = model(X_test_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            
            # 在验证集上寻找最优阈值
            X_val_tensor = torch.FloatTensor(data_dict['X_val']).to(self.device)
            val_logits = model(X_val_tensor)
            val_probs = torch.sigmoid(val_logits).cpu().numpy().flatten()
            
            # 寻找最优阈值（最大化F1分数）
            best_threshold = 0.5
            best_f1 = 0
            for threshold in np.arange(0.1, 0.9, 0.05):
                val_preds = (val_probs > threshold).astype(int)
                f1 = f1_score(data_dict['y_val'], val_preds, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            # 使用最优阈值进行最终预测
            preds = (probs > best_threshold).astype(int)
            
            accuracy = accuracy_score(data_dict['y_test'], preds)
            f1 = f1_score(data_dict['y_test'], preds, zero_division=0)
            precision = precision_score(data_dict['y_test'], preds, zero_division=0)
            recall = recall_score(data_dict['y_test'], preds, zero_division=0)
        
        print(f"  Test Accuracy: {accuracy:.4f}, Test F1-Score: {f1:.4f}")
        
        return {
            'model': model,
            'model_type': architecture_name,
            'feature_names': [f'feature_{i}' for i in range(input_features)],
            'test_metrics': {
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall
            },
            'training_info': {
                'final_loss': 'N/A',  # BCEWithLogitsLoss最终值
                'epochs': 100,  # 默认epoch数
                'parameters': model_parameters,
                'training_time': f"{training_time:.2f}s"
            }
        }
    
    def get_architecture_name(self, dataset_name):
        """根据数据集名称返回架构名称"""
        if dataset_name == 'uci':
            return 'MLP_UCI'
        elif dataset_name == 'german':
            return 'RBF_German'
        elif dataset_name == 'australian':
            return 'AE_MLP_Australian'
        else:
            return f'Neural_Network_{dataset_name.upper()}'
    
    def train_all_teacher_models(self, processed_data):
        """训练所有数据集的教师模型"""
        print("🧠 Training teacher models for all datasets...")
        
        teacher_models = {}
        
        for dataset_name, data_dict in processed_data.items():
            print(f"\n🔬 Training teacher model for {dataset_name} dataset...")
            teacher_models[dataset_name] = self.train_teacher_model(dataset_name, data_dict)
        
        print("\n✅ All teacher models trained successfully!")
        return teacher_models
