import pandas as pd
df=pd.read_csv("/data.txt")
df.mean()
df['one']  #第一列
df['one'].str.toUpper() #字符串函数
df2=df[['one',]]#多列
df[0:3] #行
df.loc[0:3,'one'] #location iloc为int&location
df.ix['a':'b',0]  #更广义切片
df.set_index('one')#设置索引
df.reset_index(inplace=True)#重置索引
df2=df[['one',]].astype('int')#改变列的类型
df.sort_index(axis=1,ascending=False)#多种排序
df.zzz.value_counts() #计算词频
df.drop(["A"],inplace=True)  #删除列
df.zzz=pd.Series(df.zzz).str.replace('%','').astype(float)#部分值去掉百分号
pd.unique(df["loan_status"]).values.ravel() #return an ndarray of the flattened values
for col in df.select_dtype(include=['object']).columns:         #查看字段有哪些值
    print("column [] has [] unique instance",format(col,len(df[col].unique())))



