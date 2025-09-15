"""
消融实验分析器 - Ablation Study Analyzer
记录和可视化Top-k知识蒸馏中各参数的消融实验结果
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os

# 设置matplotlib为非交互式模式和字体
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.use('Agg')
plt.style.use('default')
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
sns.set_palette("husl")

class AblationStudyAnalyzer:
    """消融实验分析器"""
    
    def __init__(self):
        self.ablation_results = []
        self.experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def record_experiment_result(self, dataset_name, k, temperature, alpha, max_depth, accuracy, f1_score, precision, recall):
        """记录每次实验的结果"""
        result = {
            'dataset': dataset_name,
            'k': k,
            'temperature': temperature,
            'alpha': alpha,
            'max_depth': max_depth,
            'accuracy': accuracy,
            'f1_score': f1_score,
            'precision': precision,
            'recall': recall,
            'timestamp': datetime.now().isoformat()
        }
        self.ablation_results.append(result)
        
    def save_ablation_data(self):
        """保存消融实验数据"""
        if not self.ablation_results:
            print("❌ No ablation results to save")
            return None
            
        # 保存为JSON
        json_path = f'results/ablation_study_{self.experiment_timestamp}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.ablation_results, f, indent=2)
            
        # 保存为CSV
        df = pd.DataFrame(self.ablation_results)
        csv_path = f'results/ablation_study_{self.experiment_timestamp}.csv'
        df.to_csv(csv_path, index=False)
        
        print(f"✅ Ablation study data saved:")
        print(f"   📊 JSON: {json_path}")
        print(f"   📊 CSV: {csv_path}")
        
        return csv_path
        
    def create_ablation_visualizations(self):
        """创建1x2消融实验可视化图 - 只关注α和深度参数"""
        if not self.ablation_results:
            print("❌ No ablation results to visualize")
            return None
            
        df = pd.DataFrame(self.ablation_results)
        
        # 创建1x2子图
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 数据集颜色映射 - 使用简单的颜色区分
        datasets = df['dataset'].unique()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝色、橙色、绿色
        dataset_colors = dict(zip(datasets, colors[:len(datasets)]))
        
        # 1. 加权参数α分析
        self._plot_alpha_ablation(df, axes[0], dataset_colors)
        
        # 2. 决策树深度分析
        self._plot_depth_ablation(df, axes[1], dataset_colors)
        
        plt.tight_layout()
        
        # 保存图像
        plot_path = f'results/ablation_study_analysis_{self.experiment_timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Ablation study visualization saved: {plot_path}")
        return plot_path
        
    def _plot_topk_ablation(self, df, ax, dataset_colors):
        """绘制Top-k特征数量的消融分析"""
        for dataset in df['dataset'].unique():
            dataset_data = df[df['dataset'] == dataset]
            # 按k值分组，计算平均准确率
            k_grouped = dataset_data.groupby('k')['accuracy'].mean().reset_index()
            
            ax.plot(k_grouped['k'], k_grouped['accuracy'], 
                   label=dataset.upper(), marker='o', linewidth=2, markersize=6,
                   color=dataset_colors[dataset])
                       
        ax.set_xlabel('Number of Top-k Features', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xticks(sorted(df['k'].unique()))
        
    def _plot_temperature_ablation(self, df, ax, dataset_colors):
        """绘制温度参数的消融分析"""
        for dataset in df['dataset'].unique():
            dataset_data = df[df['dataset'] == dataset]
            # 按温度分组，计算平均准确率
            temp_grouped = dataset_data.groupby('temperature')['accuracy'].mean().reset_index()
            
            ax.plot(temp_grouped['temperature'], temp_grouped['accuracy'],
                   label=dataset.upper(), marker='s', linewidth=2, markersize=6,
                   color=dataset_colors[dataset])
                       
        ax.set_xlabel('Temperature Parameter (T)', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xticks(sorted(df['temperature'].unique()))
        
    def _plot_alpha_ablation(self, df, ax, dataset_colors):
        """绘制加权参数α的消融分析"""
        for dataset in df['dataset'].unique():
            dataset_data = df[df['dataset'] == dataset]
            # 按α值分组，计算平均准确率
            alpha_grouped = dataset_data.groupby('alpha')['accuracy'].mean().reset_index()
            
            ax.plot(alpha_grouped['alpha'], alpha_grouped['accuracy'],
                   label=dataset.upper(), marker='^', linewidth=2, markersize=6,
                   color=dataset_colors[dataset])
                       
        ax.set_xlabel('Weight Parameter (α)', fontsize=12, fontfamily='sans-serif')
        ax.set_ylabel('Accuracy', fontsize=12, fontfamily='sans-serif')
        ax.set_title('Alpha Parameter Ablation Study', fontsize=14, fontfamily='sans-serif')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
    def _plot_depth_ablation(self, df, ax, dataset_colors):
        """绘制决策树深度的消融分析"""
        for dataset in df['dataset'].unique():
            dataset_data = df[df['dataset'] == dataset]
            # 按深度分组，计算平均准确率
            depth_grouped = dataset_data.groupby('max_depth')['accuracy'].mean().reset_index()
            
            ax.plot(depth_grouped['max_depth'], depth_grouped['accuracy'],
                   label=dataset.upper(), marker='d', linewidth=2, markersize=6,
                   color=dataset_colors[dataset])
                       
        ax.set_xlabel('Decision Tree Max Depth', fontsize=12, fontfamily='sans-serif')
        ax.set_ylabel('Accuracy', fontsize=12, fontfamily='sans-serif')
        ax.set_title('Tree Depth Ablation Study', fontsize=14, fontfamily='sans-serif')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_xticks(sorted(df['max_depth'].unique()))
        
    def load_and_visualize_existing_data(self, data_path):
        """从已有数据文件加载并可视化"""
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                self.ablation_results = json.load(f)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            self.ablation_results = df.to_dict('records')
        else:
            raise ValueError("Data file must be JSON or CSV format")
            
        return self.create_ablation_visualizations()
        
    def generate_summary_report(self):
        """生成消融实验总结报告"""
        if not self.ablation_results:
            return None
            
        df = pd.DataFrame(self.ablation_results)
        
        report = []
        report.append("=" * 80)
        report.append("ABLATION STUDY SUMMARY REPORT")
        report.append("=" * 80)
        report.append(f"Experiment Timestamp: {self.experiment_timestamp}")
        report.append(f"Total Experiments: {len(self.ablation_results)}")
        report.append(f"Datasets: {', '.join(df['dataset'].unique())}")
        report.append("")
        
        # 最佳配置分析
        for dataset in df['dataset'].unique():
            dataset_data = df[df['dataset'] == dataset]
            best_idx = dataset_data['accuracy'].idxmax()
            best_config = dataset_data.loc[best_idx]
            
            report.append(f"📊 {dataset.upper()} Dataset Best Configuration:")
            report.append(f"   • Accuracy: {best_config['accuracy']:.4f}")
            report.append(f"   • Top-k: {best_config['k']}")
            report.append(f"   • Temperature: {best_config['temperature']}")
            report.append(f"   • Alpha: {best_config['alpha']}")
            report.append(f"   • Max Depth: {best_config['max_depth']}")
            report.append("")
            
        # 参数影响分析
        report.append("🔍 Parameter Impact Analysis:")
        for param in ['k', 'temperature', 'alpha', 'max_depth']:
            correlation = df.groupby(param)['accuracy'].mean().corr(df.groupby(param).size())
            report.append(f"   • {param.upper()}: {correlation:.3f} correlation with accuracy")
            
        report_text = "\n".join(report)
        
        # 保存报告
        report_path = f'results/ablation_study_report_{self.experiment_timestamp}.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
            
        # 生成Excel报告
        excel_path = f'results/ablation_study_results_{self.experiment_timestamp}.xlsx'
        df.to_excel(excel_path, index=False)
        
        print(f"✅ Ablation study report saved: {report_path}")
        print(f"✅ Ablation study Excel saved: {excel_path}")
        print("\n" + report_text)
        
        return report_path

    def create_topk_ablation_visualizations(self):
        """创建2x2 Top-k消融实验可视化图 - 关注k, α, 深度, 温度参数"""
        if not self.ablation_results:
            print("❌ No Top-k ablation results to visualize")
            return None
            
        df = pd.DataFrame(self.ablation_results)
        
        # 创建2x2子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 数据集颜色映射
        datasets = df['dataset'].unique()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝色、橙色、绿色
        dataset_colors = dict(zip(datasets, colors[:len(datasets)]))
        
        # 1. Top-k特征数量分析
        self._plot_topk_k_ablation(df, axes[0, 0], dataset_colors)
        
        # 2. 加权参数α分析
        self._plot_alpha_ablation(df, axes[0, 1], dataset_colors)
        
        # 3. 决策树深度分析
        self._plot_depth_ablation(df, axes[1, 0], dataset_colors)
        
        # 4. 温度参数分析
        self._plot_temperature_ablation(df, axes[1, 1], dataset_colors)
        
        plt.tight_layout()
        
        # 保存图像
        plot_path = f'results/topk_ablation_study_analysis_{self.experiment_timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Top-k ablation study visualization saved: {plot_path}")
        return plot_path

    def _plot_topk_k_ablation(self, df, ax, dataset_colors):
        """绘制Top-k特征数量消融分析"""
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            k_accuracy = dataset_df.groupby('k')['accuracy'].mean().reset_index()
            
            ax.plot(k_accuracy['k'], k_accuracy['accuracy'], 
                   marker='o', linewidth=2, markersize=6, 
                   color=dataset_colors[dataset], label=dataset.upper())
        
        ax.set_xlabel('Top-k Features', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_temperature_ablation(self, df, ax, dataset_colors):
        """绘制温度参数消融分析"""
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            temp_accuracy = dataset_df.groupby('temperature')['accuracy'].mean().reset_index()
            
            ax.plot(temp_accuracy['temperature'], temp_accuracy['accuracy'], 
                   marker='s', linewidth=2, markersize=6, 
                   color=dataset_colors[dataset], label=dataset.upper())
        
        ax.set_xlabel('Temperature', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def save_ablation_data(self, prefix='ablation_study'):
        """保存消融实验数据 - 支持自定义前缀"""
        if not self.ablation_results:
            print("❌ No ablation data to save")
            return
        
        df = pd.DataFrame(self.ablation_results)
        
        # 保存CSV
        csv_path = f'results/{prefix}_{self.experiment_timestamp}.csv'
        df.to_csv(csv_path, index=False)
        
        # 保存JSON  
        json_path = f'results/{prefix}_{self.experiment_timestamp}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.ablation_results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ {prefix} data saved: {csv_path}")
        print(f"✅ {prefix} data saved: {json_path}")

    def generate_summary_report(self, prefix='ablation_study'):
        """生成消融实验总结报告 - 支持自定义前缀"""
        if not self.ablation_results:
            print("❌ No ablation data to generate report")
            return None
            
        df = pd.DataFrame(self.ablation_results)
        
        report = []
        report.append("="*80)
        if 'topk' in prefix:
            report.append("Top-k Knowledge Distillation Ablation Study Report")
        else:
            report.append("All-Feature Knowledge Distillation Ablation Study Report")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total experiments: {len(df)}")
        report.append("")
        
        # 数据集统计
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            best_result = dataset_df.loc[dataset_df['accuracy'].idxmax()]
            
            report.append(f"📊 {dataset.upper()} Dataset:")
            report.append(f"   Best Accuracy: {best_result['accuracy']:.4f}")
            report.append(f"   Best F1-Score: {best_result['f1_score']:.4f}")
            
            if 'k' in best_result and best_result['k'] is not None:
                report.append(f"   Optimal k: {best_result['k']}")
            report.append(f"   Optimal α: {best_result['alpha']}")
            report.append(f"   Optimal Temperature: {best_result['temperature']}")
            report.append(f"   Optimal Max Depth: {best_result['max_depth']}")
            report.append("")
            
        report_text = "\n".join(report)
        
        # 保存报告
        report_path = f'results/{prefix}_report_{self.experiment_timestamp}.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
            
        # 生成Excel报告
        excel_path = f'results/{prefix}_results_{self.experiment_timestamp}.xlsx'
        df.to_excel(excel_path, index=False)
        
        print(f"✅ {prefix} report saved: {report_path}")
        print(f"✅ {prefix} Excel saved: {excel_path}")
        print("\n" + report_text)
        
        return report_path

# 全局消融实验分析器实例
ablation_analyzer = AblationStudyAnalyzer()