import pandas as pd

a = pd.DataFrame([[1, 4], [2, 5]])
m = dict(a.values)

b = pd.DataFrame([[1], [2], [2], [1]], columns=['a'])
b['a'] = b['a'].map(m)
print(b)