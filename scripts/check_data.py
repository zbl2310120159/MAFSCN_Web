import pandas as pd
import os

raw = 'd:/aicode/MAFSCN/MAFSCN_Web/data/raw'
files = sorted(os.listdir(raw))
print(f'共{len(files)}个文件:')
for f in files:
    print(f'  {f}')

print()
total_rows = 0
for f in files:
    df = pd.read_excel(os.path.join(raw, f))
    total_rows += df.shape[0]
    print(f'{f}: {df.shape[0]}行, 列数={df.shape[1]}')

print(f'\n总行数: {total_rows}')

# 验证列名一致性
cols_set = set()
for f in files:
    df = pd.read_excel(os.path.join(raw, f))
    cols_set.add(tuple(df.columns.tolist()))
print(f'\n列名模式数: {len(cols_set)}')
if len(cols_set) == 1:
    print('所有文件列名一致!')
    print(f'列名: {list(cols_set)[0]}')
else:
    for i, c in enumerate(cols_set):
        print(f'模式{i+1}: {c}')