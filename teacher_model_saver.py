"""
精简版教师模型保存器
Simplified Teacher Model Saver
"""

import os
import torch
import pickle
import json
from datetime import datetime


class TeacherModelSaver:
    """教师模型保存器"""
    
    def __init__(self, models_dir='trained_models'):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
    
    def save_teacher_models(self, teacher_models):
        """
        保存教师模型到trained_models文件夹
        
        Args:
            teacher_models: 教师模型字典
        """
        print(f"💾 保存教师模型到 {self.models_dir}...")
        
        for dataset_name, model_info in teacher_models.items():
            # 保存模型性能信息到JSON
            json_path = f"{self.models_dir}/teacher_model_{dataset_name}.json"
            model_metrics = {
                'dataset': dataset_name,
                'model_type': model_info.get('model_type', 'PyTorch_DNN'),
                'accuracy': model_info.get('accuracy', 0),
                'precision': model_info.get('precision', 0),
                'recall': model_info.get('recall', 0),
                'f1': model_info.get('f1', 0),
                'auc': model_info.get('auc', 0),
                'training_time': model_info.get('training_time', 0),
                'saved_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(model_metrics, f, indent=2, ensure_ascii=False)
            
            # 保存模型对象
            pkl_path = f"{self.models_dir}/teacher_model_{dataset_name}.pkl"
            with open(pkl_path, 'wb') as f:
                pickle.dump({
                    'model': model_info['model'],
                    'model_type': model_info.get('model_type', 'PyTorch_DNN'),
                    'metrics': model_metrics
                }, f)
            
            # 如果是PyTorch模型，额外保存state_dict
            model = model_info['model']
            if hasattr(model, 'state_dict'):
                pth_path = f"{self.models_dir}/teacher_model_{dataset_name}.pth"
                torch.save(model.state_dict(), pth_path)
                print(f"   ✅ PyTorch模型已保存: {dataset_name}")
            else:
                print(f"   ✅ 模型已保存: {dataset_name}")
        
        print(f"   📦 所有教师模型已保存到: {self.models_dir}/")
        return True