import pandas as pd

# 检查生成的Excel文件
df = pd.read_excel('results/comprehensive_model_comparison.xlsx')
print('Feature Selection values:')
print(df['Feature_Selection'].values)
print()
print('Top-k rows:')
topk_rows = df[df['Model_Type'] == 'Top-k Distilled Tree']
print(topk_rows[['Dataset', 'Feature_Selection', 'Notes']].to_string(index=False))
