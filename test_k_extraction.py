from experiment_manager import ExperimentManager

# 测试k值提取逻辑
em = ExperimentManager()

# 模拟一个top-k结果
test_result = {
    'best': {'accuracy': 0.8406, 'f1': 0.8376, 'temperature': 1, 'alpha': 0.1, 'max_depth': 4},
    'best_k': 5
}

print("Testing k value extraction...")
best = em.extract_best_model(test_result)
print('Best result keys:', list(best.keys()))
print('k value:', best.get('k', 'NOT FOUND'))
print('temperature:', best.get('temperature', 'NOT FOUND'))
print('alpha:', best.get('alpha', 'NOT FOUND'))
