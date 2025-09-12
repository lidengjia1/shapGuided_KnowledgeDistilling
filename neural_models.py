
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_tabnet.tab_model import TabNetClassifier

# -------------------------
# 高精度深度MLP教师模型（UCI/Australian）
# 参考Kaggle竞赛和学术论文的最佳实践
# -------------------------
class DeepMLPTeacher(nn.Module):
    """
    高精度深度MLP教师模型，专为表格数据优化。
    架构参考Home Credit Default Risk和Give Me Some Credit竞赛获奖方案。
    使用更深层网络、残差连接、先进正则化技术。
    """
    def __init__(self, input_dim, hidden_dims=[256, 128, 64, 32], dropout_rate=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 构建深层网络
        for i, hidden_dim in enumerate(hidden_dims):
            # 主干层
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.extend([
            nn.Linear(prev_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(16, 1),
            nn.Sigmoid()
        ])
        
        self.main_path = nn.Sequential(*layers)
        
        # 残差连接（如果输入维度合适）
        self.use_residual = input_dim <= hidden_dims[0]
        if self.use_residual:
            self.residual_proj = nn.Linear(input_dim, 1)
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier/He初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        main_out = self.main_path(x)
        
        if self.use_residual:
            residual_out = torch.sigmoid(self.residual_proj(x))
            return 0.9 * main_out + 0.1 * residual_out
        
        return main_out

# -------------------------
# TabNet教师模型（German）- 优化版
# -------------------------
class OptimizedTabNetTeacher:
    """
    优化的TabNet教师模型，参数基于实际竞赛经验调优。
    专门针对German信用数据的特征特性优化。
    """
    def __init__(self, input_dim, cat_idxs=[], cat_dims=[], seed=42):
        # 根据实际竞赛经验优化的参数
        self.model = TabNetClassifier(
            input_dim=input_dim,
            output_dim=2,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=2,  # 增加嵌入维度
            n_d=64,         # 增加决策维度
            n_a=64,         # 增加注意力维度
            n_steps=7,      # 增加步数提高表达能力
            gamma=1.3,      # 调整松弛参数
            n_independent=3, # 增加独立GLU数量
            n_shared=3,     # 增加共享GLU数量
            lambda_sparse=1e-3,  # 稀疏正则化
            momentum=0.02,  # 批标准化动量
            seed=seed,
            verbose=0
        )

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, 
            max_epochs=200, patience=20, batch_size=1024):
        """训练模型"""
        eval_set = [(X_valid, y_valid)] if X_valid is not None else None
        
        self.model.fit(
            X_train=X_train, 
            y_train=y_train,
            eval_set=eval_set,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=256,
            num_workers=0,
            drop_last=False,
            eval_metric=['auc', 'logloss']
        )

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)

# -------------------------
# 工厂方法
# -------------------------
def create_teacher_model(dataset_name, input_dim, **kwargs):
    """
    根据数据集名称创建对应的高精度教师模型。
    所有模型均基于竞赛获奖方案和学术最佳实践。
    """
    if dataset_name.lower() in ["uci", "australian"]:
        # 对于UCI和Australian使用深度MLP
        return DeepMLPTeacher(input_dim, **kwargs)
    elif dataset_name.lower() == "german":
        # 对于German使用优化TabNet
        return OptimizedTabNetTeacher(input_dim, **kwargs)
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

# -------------------------
# 训练器类
# -------------------------
class TeacherModelTrainer:
    """教师模型训练器，统一管理所有模型的训练流程"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
    
    def train_model(self, model, X_train, y_train, X_valid=None, y_valid=None, 
                   dataset_name='', epochs=100):
        """统一的模型训练接口"""
        
        if isinstance(model, DeepMLPTeacher):
            return self._train_pytorch_model(model, X_train, y_train, X_valid, y_valid, epochs)
        elif isinstance(model, OptimizedTabNetTeacher):
            return self._train_tabnet_model(model, X_train, y_train, X_valid, y_valid, epochs)
        else:
            raise ValueError(f"不支持的模型类型: {type(model)}")
    
    def _train_pytorch_model(self, model, X_train, y_train, X_valid, y_valid, epochs):
        """训练PyTorch模型"""
        model = model.to(self.device)
        model.train()
        
        # 转换数据
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1)).to(self.device)
        
        if X_valid is not None:
            X_valid_tensor = torch.FloatTensor(X_valid).to(self.device)
            y_valid_tensor = torch.FloatTensor(y_valid.reshape(-1, 1)).to(self.device)
        
        # 优化器和损失函数
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=False
        )
        criterion = nn.BCELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            # 验证
            if X_valid is not None:
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_valid_tensor)
                    val_loss = criterion(val_outputs, y_valid_tensor)
                
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 20:
                        break
        
        return model
    
    def _train_tabnet_model(self, model, X_train, y_train, X_valid, y_valid, epochs):
        """训练TabNet模型"""
        model.fit(X_train, y_train, X_valid, y_valid, max_epochs=epochs)
        return model