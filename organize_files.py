"""
文件管理和结构优化脚本
File Management and Structure Optimization Script
"""

import os
import shutil
import glob

def organize_project_structure():
    """优化项目文件结构"""
    print("🗂️  Organizing project structure...")
    
    # 1. 创建必要的目录
    directories = [
        'results',
        'models',
        'data',
        'visualization'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✅ Directory created/verified: {directory}")
    
    # 2. 移动模型文件
    model_files = glob.glob('results/teacher_model_*.pkl') + glob.glob('results/teacher_model_*.pth')
    for model_file in model_files:
        filename = os.path.basename(model_file)
        new_path = f"models/{filename}"
        if os.path.exists(model_file):
            shutil.move(model_file, new_path)
            print(f"   📦 Moved: {filename} -> models/")
    
    # 3. 移动可视化文件
    viz_files = glob.glob('results/*.png')
    for viz_file in viz_files:
        filename = os.path.basename(viz_file)
        new_path = f"visualization/{filename}"
        if os.path.exists(viz_file):
            shutil.move(viz_file, new_path)
            print(f"   🎨 Moved: {filename} -> visualization/")
    
    # 4. 删除临时和测试文件
    temp_patterns = [
        'results/*.pkl',  # 除了模型，其他pickle文件
        'results/tree_text_*.txt',
        'results/~$*.xlsx',  # Excel临时文件
    ]
    
    for pattern in temp_patterns:
        temp_files = glob.glob(pattern)
        for temp_file in temp_files:
            if 'distillation_results' not in temp_file and 'processed_data' not in temp_file:
                try:
                    os.remove(temp_file)
                    filename = os.path.basename(temp_file)
                    print(f"   🗑️  Removed: {filename}")
                except:
                    pass
    
    print("   ✅ Project structure optimization completed!")

def clean_unused_python_files():
    """清理无用的Python文件"""
    print("\n🧹 Cleaning unused Python files...")
    
    # 核心必要文件（不删除）
    essential_files = {
        'main.py',
        'data_preprocessing.py', 
        'neural_models.py',
        'shap_analysis.py',
        'distillation_module.py',
        'experiment_manager.py',
        'ablation_visualizer.py',
        'tree_rules_analyzer.py',
        'organize_files.py'
    }
    
    # 可删除的测试和调试文件
    files_to_remove = [
        'credit_scoring_system.py',
        'debug_features.py', 
        'knowledge_distillation.py',
        'main_optimized.py',
        'simple_debug.py',
        'simple_test.py',
        'student_model_trainer.py',
        'test_enhanced_models.py',
        'test_fix.py',
        'tree_rules_extractor.py'  # 已被tree_rules_analyzer.py替代
    ]
    
    removed_count = 0
    for file_to_remove in files_to_remove:
        if os.path.exists(file_to_remove):
            try:
                os.remove(file_to_remove)
                print(f"   🗑️  Removed: {file_to_remove}")
                removed_count += 1
            except Exception as e:
                print(f"   ⚠️  Error removing {file_to_remove}: {str(e)}")
    
    print(f"   ✅ Removed {removed_count} unused Python files")
    
    # 显示保留的核心文件
    print(f"\n📋 Essential files retained:")
    for essential_file in sorted(essential_files):
        if os.path.exists(essential_file):
            print(f"   ✅ {essential_file}")

def clean_pycache():
    """清理__pycache__目录"""
    print("\n🧹 Cleaning __pycache__ directories...")
    
    pycache_dirs = glob.glob('**/__pycache__', recursive=True)
    for pycache_dir in pycache_dirs:
        try:
            shutil.rmtree(pycache_dir)
            print(f"   🗑️  Removed: {pycache_dir}")
        except Exception as e:
            print(f"   ⚠️  Error removing {pycache_dir}: {str(e)}")
    
    print("   ✅ __pycache__ cleanup completed!")

if __name__ == "__main__":
    organize_project_structure()
    clean_unused_python_files()
    clean_pycache()
