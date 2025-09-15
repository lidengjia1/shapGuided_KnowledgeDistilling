"""
重新生成消融实验图的脚本
Re-generate ablation study visualizations script
"""

import pandas as pd
import os
from ablation_analyzer import AblationStudyAnalyzer

def regenerate_ablation_plots():
    """重新生成消融实验图"""
    print("🔄 重新生成消融实验图...")
    
    # 找到最新的消融实验数据文件
    results_dir = "results"
    ablation_files = []
    topk_ablation_files = []
    
    for filename in os.listdir(results_dir):
        if filename.startswith('ablation_study_') and filename.endswith('.csv'):
            ablation_files.append(filename)
        elif filename.startswith('topk_ablation_study_') and filename.endswith('.csv'):
            topk_ablation_files.append(filename)
    
    if not ablation_files and not topk_ablation_files:
        print("❌ 没有找到消融实验数据文件")
        return
    
    # 使用最新的文件
    latest_ablation = sorted(ablation_files)[-1] if ablation_files else None
    latest_topk_ablation = sorted(topk_ablation_files)[-1] if topk_ablation_files else None
    
    print(f"📂 找到消融实验数据文件:")
    if latest_ablation:
        print(f"   - 全特征消融: {latest_ablation}")
    if latest_topk_ablation:
        print(f"   - Top-k消融: {latest_topk_ablation}")
    
    # 创建消融分析器
    analyzer = AblationStudyAnalyzer()
    
    # 重新生成图表
    if latest_ablation:
        print("\n🎨 重新生成全特征消融图...")
        ablation_df = pd.read_csv(os.path.join(results_dir, latest_ablation))
        viz_path = analyzer.create_ablation_visualizations(ablation_df, "ablation_study")
        print(f"   ✅ 生成: {viz_path}")
    
    if latest_topk_ablation:
        print("\n🎨 重新生成Top-k消融图...")
        topk_ablation_df = pd.read_csv(os.path.join(results_dir, latest_topk_ablation))
        topk_viz_path = analyzer.create_ablation_visualizations(topk_ablation_df, "topk_ablation_study")
        print(f"   ✅ 生成: {topk_viz_path}")
    
    print("\n🎉 消融实验图重新生成完成!")

if __name__ == "__main__":
    regenerate_ablation_plots()