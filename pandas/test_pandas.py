import pandas as pd


#Series 可以看做一个定长的有序字典。基本任意的一维数据都可以用来构造 Series 对象：
s = pd.Series([1,2,3.0,'abc'])
s1 = pd.Series(data=[1,3,5,7],index = ['a','b','x','y'])
#通过下标获取数据
s1['a']
#Series的name属性
s1.name='test_series'
#pandas 最重要的一个功能是，它可以对不同索引的对象进行算术运算。
# 在将对象相加时，结果的索引取索引对的并集。自动的数据对齐在不重叠的索引处引入空值，
# 默认为 NaN。
foo = pd.Series({'a':1,'b':2})
foo

bar = pd.Series({'b':3,'d':4})
bar

foo + bar






#DataFrame 是一个表格型的数据结构 ,类似于数据库的表
data = {'state':['Ohino','Ohino','Ohino','Nevada','Nevada'],'year':[2000,2001,2002,2001,2002],'pop':[1.5,1.7,3.6,2.4,2.9]}
df = pd.DataFrame(data)
#获取行名
df = pd.DataFrame(data,index=['one','two','three','four','five'],columns=['year','state','pop','debt'])
#获取index
df.index
#获取列名
df.columns
#删除,返回一个新对象，原对象不会被改变
df.drop('one')
#pandas 也支持通过 obj[::] 的方式进行索引和切片，以及通过布尔型数组进行过滤。
#换成 'c' 这样的字符串索引时，结果就包含了这个边界元素。
df
df[:2]
df[:'three']
#DataFrame 对象的标准切片语法为：.ix[::,::]。ix 对象可以接受两套切片，分别为行（axis=0）和列（axis=1）的方向：
#对于行的切片和列的切片
df.ix[:2,:2]


#排序操作
#Series 的 sort_index(ascending=True) 方法可以对 index 进行排序操作，ascending 参数用于控制升序或降序，默认为升序。
df.sort_index(by='year')
df.sort_index(by=['year','pop'])
df.sort_index(axis=1)
df.min()
#跳过NaN值
df.mean()
