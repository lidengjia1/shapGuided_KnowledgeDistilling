"""
Top-k消融实验可视化模块
Top-k Ablation Study Visualization Module
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')

class TopKAblationVisualizer:
    """Top-k消融实验可视化器"""
    
    def __init__(self, results_dir='results'):
        self.results_dir = results_dir
        
    def create_topk_ablation_plots(self, top_k_distillation_results):
        """创建Top-k消融实验的四张图表 (2×2布局)
        
        四张图分别展示：
        1. Top-k特征数量对准确率的影响
        2. 决策树深度对准确率的影响  
        3. 温度参数对准确率的影响
        4. 加权参数(Alpha)对准确率的影响
        
        每张图包含三条线，代表三个数据集
        """
        print(f"📊 Creating Top-k Ablation Study Plots...")
        
        # 提取数据
        ablation_data = self._extract_ablation_data(top_k_distillation_results)
        
        # 创建2×2子图布局
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Top-k Knowledge Distillation Ablation Study', fontsize=16, fontweight='bold')
        
        datasets = ['uci', 'german', 'australian']
        dataset_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        dataset_labels = ['UCI Credit', 'German Credit', 'Australian Credit']
        
        # 1. Top-k特征数量影响 (左上)
        ax1 = axes[0, 0]
        for i, dataset in enumerate(datasets):
            k_performance = ablation_data[dataset]['k_performance']
            k_values = sorted(k_performance.keys())
            accuracies = [k_performance[k] for k in k_values]
            
            ax1.plot(k_values, accuracies, marker='o', linewidth=2.5, markersize=8,
                    color=dataset_colors[i], label=dataset_labels[i])
        
        ax1.set_title('Impact of Top-k Feature Count', fontweight='bold', fontsize=12)
        ax1.set_xlabel('Number of Top-k Features')
        ax1.set_ylabel('Average Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(5, 11))
        
        # 2. 决策树深度影响 (右上)
        ax2 = axes[0, 1]
        for i, dataset in enumerate(datasets):
            depth_performance = ablation_data[dataset]['depth_performance']
            depth_values = sorted(depth_performance.keys())
            accuracies = [depth_performance[d] for d in depth_values]
            
            ax2.plot(depth_values, accuracies, marker='s', linewidth=2.5, markersize=8,
                    color=dataset_colors[i], label=dataset_labels[i])
        
        ax2.set_title('Impact of Decision Tree Max Depth', fontweight='bold', fontsize=12)
        ax2.set_xlabel('Max Depth')
        ax2.set_ylabel('Average Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(4, 9))
        
        # 3. 温度参数影响 (左下)
        ax3 = axes[1, 0]
        for i, dataset in enumerate(datasets):
            temp_performance = ablation_data[dataset]['temp_performance']
            temp_values = sorted(temp_performance.keys())
            accuracies = [temp_performance[t] for t in temp_values]
            
            ax3.plot(temp_values, accuracies, marker='^', linewidth=2.5, markersize=8,
                    color=dataset_colors[i], label=dataset_labels[i])
        
        ax3.set_title('Impact of Temperature Parameter', fontweight='bold', fontsize=12)
        ax3.set_xlabel('Temperature (T)')
        ax3.set_ylabel('Average Accuracy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(range(1, 6))
        
        # 4. Alpha参数影响 (右下)
        ax4 = axes[1, 1]
        for i, dataset in enumerate(datasets):
            alpha_performance = ablation_data[dataset]['alpha_performance']
            alpha_values = sorted(alpha_performance.keys())
            accuracies = [alpha_performance[a] for a in alpha_values]
            
            ax4.plot(alpha_values, accuracies, marker='D', linewidth=2.5, markersize=8,
                    color=dataset_colors[i], label=dataset_labels[i])
        
        ax4.set_title('Impact of Alpha Parameter (Soft/Hard Label Mixing)', fontweight='bold', fontsize=12)
        ax4.set_xlabel('Alpha (α)')
        ax4.set_ylabel('Average Accuracy')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        
        # 调整布局并保存
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        viz_path = f"{self.results_dir}/topk_ablation_study.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Top-k ablation study plots saved: {viz_path}")
        
        # 打印关键发现
        self._print_ablation_insights(ablation_data)
        
        return viz_path
    
    def _extract_ablation_data(self, top_k_distillation_results):
        """从实验结果中提取消融实验数据"""
        ablation_data = {}
        
        for dataset_name in ['uci', 'german', 'australian']:
            ablation_data[dataset_name] = {
                'k_performance': defaultdict(list),
                'depth_performance': defaultdict(list),
                'temp_performance': defaultdict(list),
                'alpha_performance': defaultdict(list)
            }
            
            if dataset_name not in top_k_distillation_results:
                continue
                
            dataset_results = top_k_distillation_results[dataset_name]
            
            # 遍历所有实验结果
            for k in dataset_results:
                for temp in dataset_results[k]:
                    for alpha in dataset_results[k][temp]:
                        for depth in dataset_results[k][temp][alpha]:
                            result = dataset_results[k][temp][alpha][depth]
                            if result is not None and 'accuracy' in result:
                                accuracy = result['accuracy']
                                
                                # 按不同维度收集数据
                                ablation_data[dataset_name]['k_performance'][k].append(accuracy)
                                ablation_data[dataset_name]['depth_performance'][depth].append(accuracy)
                                ablation_data[dataset_name]['temp_performance'][temp].append(accuracy)
                                ablation_data[dataset_name]['alpha_performance'][alpha].append(accuracy)
            
            # 计算平均值
            for perf_type in ['k_performance', 'depth_performance', 'temp_performance', 'alpha_performance']:
                for param_value in ablation_data[dataset_name][perf_type]:
                    accuracy_list = ablation_data[dataset_name][perf_type][param_value]
                    ablation_data[dataset_name][perf_type][param_value] = np.mean(accuracy_list)
        
        return ablation_data
    
    def _print_ablation_insights(self, ablation_data):
        """打印消融实验的关键发现"""
        print(f"\n🔍 Top-k Ablation Study Key Insights:")
        
        for dataset_name in ['uci', 'german', 'australian']:
            print(f"\n📈 {dataset_name.upper()} Dataset:")
            
            # 最佳k值
            k_perf = ablation_data[dataset_name]['k_performance']
            if k_perf:
                best_k = max(k_perf.keys(), key=lambda x: k_perf[x])
                print(f"   • Best k (features): {best_k} (Acc: {k_perf[best_k]:.4f})")
            
            # 最佳深度
            depth_perf = ablation_data[dataset_name]['depth_performance']
            if depth_perf:
                best_depth = max(depth_perf.keys(), key=lambda x: depth_perf[x])
                print(f"   • Best depth: {best_depth} (Acc: {depth_perf[best_depth]:.4f})")
            
            # 最佳温度
            temp_perf = ablation_data[dataset_name]['temp_performance']
            if temp_perf:
                best_temp = max(temp_perf.keys(), key=lambda x: temp_perf[x])
                print(f"   • Best temperature: {best_temp} (Acc: {temp_perf[best_temp]:.4f})")
            
            # 最佳alpha
            alpha_perf = ablation_data[dataset_name]['alpha_performance']
            if alpha_perf:
                best_alpha = max(alpha_perf.keys(), key=lambda x: alpha_perf[x])
                print(f"   • Best alpha: {best_alpha} (Acc: {alpha_perf[best_alpha]:.4f})")
        
        return ablation_data
