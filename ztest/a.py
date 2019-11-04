import numpy as np
import pandas as pd

frame = pd.DataFrame({'a': 1.,
					  'b': pd.Timestamp('20190101'),
					  'c': pd.Series(1, index=list(range(4)), dtype='float32'),
					  'd': np.array([3] * 4, dtype='int32'),
					  'e': pd.Categorical(['test', 'train', 'test', 'train']),
					  'f': 'foo'})
dates = pd.date_range('20160101', periods=6)

randn = np.random.randn(6, 4)
df = pd.DataFrame(randn, columns=['a', 'b', 'c', 'd'])
print(df)
print(df.loc['0','a'])
print('-------------')
print(df.ix[0,'a'])
