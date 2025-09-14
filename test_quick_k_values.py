from experiment_manager import ExperimentManager
import pickle
import os

# 加载之前的实验结果
print("Loading saved results...")

with open('results/shap_results.pkl', 'rb') as f:
    all_shap_results = pickle.load(f)

with open('results/processed_data.pkl', 'rb') as f:
    processed_data = pickle.load(f)
    
with open('results/distillation_results.pkl', 'rb') as f:
    distillation_data = pickle.load(f)

# 提取各种结果
teacher_models = distillation_data.get('teacher_models', {})
baseline_results = distillation_data.get('baseline_results', {})
all_feature_distillation_results = distillation_data.get('all_feature_distillation_results', {})
top_k_distillation_results = distillation_data.get('top_k_distillation_results', {})

print("Results loaded. Testing k value extraction...")

# 测试k值提取
em = ExperimentManager()

for dataset in ['uci', 'german', 'australian']:
    if dataset in top_k_distillation_results:
        print(f"\n=== Testing {dataset} ===")
        results = top_k_distillation_results[dataset]
        print(f"Raw results keys: {list(results.keys())}")
        
        best = em._find_best_model_in_results(results)
        if best:
            print(f"Best result keys: {list(best.keys())}")
            print(f"k value: {best.get('k', 'NOT FOUND')}")
            print(f"temperature: {best.get('temperature', 'NOT FOUND')}")
            print(f"alpha: {best.get('alpha', 'NOT FOUND')}")
        else:
            print("No best result found!")

print("\nTesting comparison table generation...")
comparison_results = em.create_comprehensive_comparison_table(
    teacher_models, baseline_results, all_feature_distillation_results, top_k_distillation_results
)
