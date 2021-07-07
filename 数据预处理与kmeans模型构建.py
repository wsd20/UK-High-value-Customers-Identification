import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px

'''
问题陈述背景：
英国一家在线零售店捕获了一年（2016 年 11 月至 2017 年 12 月）不同产品的销售数据。
该组织主要在在线平台上销售礼品。购买的顾客直接为自己消费。有一些小企业通过零售渠道批量购买并销售给其他客户。

项目目标：
为企业寻找大量购买他们喜欢的产品的重要客户。
该组织希望在识别细分市场后向高价值客户推出忠诚度计划。使用聚类方法将客户分组：

数据集描述
这是一个跨国数据集，包含英国在线零售店2016年11月至2017年12月之间发生的所有交易。
属性描述
InvoiceNo （为每个交易唯一分配的6位整数）
StockCode （项）库存代码产品代码
Description（项）名称
Quantity（项目）的数量
InvoiceDate生成每个事务的日期
UnitPrice单价（单位产品价格）
CustomerID 客户编号（分配给每个客户的唯一ID）
Country Country name（每个客户居住的国家的名称）
'''

'''
任务
1、确定正确数量的客户群。 
2、提供受高度重视的客户数量。
3、确定可提供最高准确度并解释稳健聚类的聚类算法。
4、如果在其中一个集群中加载了观察数量，请使用聚类算法进一步分解该集群。
[提示：这里加载意味着如果任何集群与其他集群相比具有更多数量的数据点，
则通过增加集群数量来拆分该集群并观察，将结果与之前的结果进行比较。]

'''

'''parse_dates参数：将csv中的时间字符串转换成日期格式'''
data = pd.read_csv('./项目/英国-高价值客户识别/Ecommerce.csv',parse_dates=['InvoiceDate'],encoding='unicode_escape')
data.info()


'''导入数据后存在一个全是空值的列，删除该列'''
data = data.drop(['Unnamed: 8'],axis=1)
data.dtypes

'''查看每列的空值数目'''
data.apply(lambda x: sum(x.isnull()))
'''可以看到Description有1454个空值，CustomerID存在135080个空值，
Description是对用户所购买物品的描述，CustomerID是每个用户的唯一标识'''

'''去除存在空值的数据'''
data = data.dropna()

'''查看物品描述'''
description = data['Description']
description.head(10)

'''查看用户数量'''
data['CustomerID'].value_counts()
'''用户数量为4372'''



'''将每笔账单的购买物品的单价乘以数量作为新的一列销售额加入data'''
data['total_price'] = data['Quantity']*data['UnitPrice']

'''转换InvoiceDate日期'''
data['Year'] = data['InvoiceDate'].dt.year
data['Month'] = data['InvoiceDate'].dt.month
data['Day'] = data['InvoiceDate'].dt.day
data.head()

'''去除购买数量和购买单价小于等于0的订单'''
discard = data[data['Quantity'] <= 0].index
data.drop(discard, inplace=True )
data.shape

discard = data[data['UnitPrice'] <= 0].index
data.drop(discard, inplace=True )
data.shape

'''使用散点图查看异常值'''
plt.scatter(data['Quantity'], data['UnitPrice'])
plt.show()

'''去除异常值'''
discard2 = data[data['UnitPrice'] > 5000].index
discard3 = data[data['Quantity'] > 10000].index
data.drop(discard2, inplace=True )
data.drop(discard3, inplace=True )

'''去除异常点后'''
plt.scatter(data['Quantity'], data['UnitPrice'])
plt.show()


data['CustomerID'] = data['CustomerID'].astype('int').astype('category')

# 将对象列转换为类别以减少使用的内存
categories = ['InvoiceNo', 'StockCode', 'Description', 'Country']
for c in categories:
    data[c] = data[c].astype('category')
print(data.info())

'''查看销售年份'''
years = data['Year'].value_counts()
years


'''查看各年销售占比'''
pie, ax = plt.subplots(figsize=[12,8])
labels = ['2017', '2016']
colors = ['goldenrod', 'teal']
plt.pie(x = years, autopct='%.1f%%', explode=[0.05]*2, labels=labels, pctdistance=0.5, colors = colors)
plt.title('% of sales by year')
plt.show()

# 按年度分开销售
sales_16 = data[data['Year'] == 2016]
sales_17 = data[data['Year'] == 2017]

monthly_16 = sales_16['Month'].value_counts()
monthly_16


'''2016年销售额'''
plt.figure(figsize=(8,4))
monthly_16.sort_index().plot(color='teal', kind='bar')
plt.title('# of sales in the months of 2016')
plt.xlabel('Month')
plt.ylabel('# of sales')
plt.grid()
plt.show()



cash_16 = sales_16.groupby('Month')['total_price'].sum()


'''2016年每月总收入（英镑）'''
plt.figure(figsize=(8,4))
cash_16.sort_index().plot(kind='bar', color='teal')
plt.title('Total income (pounds) by month in 2016')
plt.xlabel('Month')
plt.ylabel('Income (pounds)')
plt.grid()
plt.show()

'''2016年月度销售总数量'''
quantity_16 = sales_16.groupby('Month')['Quantity'].sum()
plt.figure(figsize=(8,4))
quantity_16.sort_index().plot(kind='bar', color='teal')
plt.title('Total quantity sold by month in 2016')
plt.xlabel('Month')
plt.ylabel('Quantity')
plt.grid()
plt.show()


'''2017年销售额'''
monthly_17 = sales_17['Month'].value_counts()

plt.figure(figsize=(8,4))
monthly_17.sort_index().plot(kind='bar', color='goldenrod')
plt.title('# of sales in the months of 2017')
plt.xlabel('Month')
plt.ylabel('# of sales')
plt.grid()
plt.show()


'''2017年每月总收入（英镑）'''
cash_17 = sales_17.groupby('Month')['total_price'].sum()

plt.figure(figsize=(8,4))
cash_17.sort_index().plot(kind='bar', color='goldenrod')
plt.title('Total income (pounds) by month in 2017')
plt.xlabel('Month')
plt.ylabel('Income (pounds)')
plt.grid()
plt.show()



'''2017年月度销售总数量'''
quantity_17 = sales_17.groupby('Month')['Quantity'].sum()

plt.figure(figsize=(8,4))
quantity_17.sort_index().plot(kind='bar', color='goldenrod')
plt.title('Total quantity sold by month in 2017')
plt.xlabel('Month')
plt.ylabel('Quantity')
plt.grid()
plt.show()



'''对比16和17年11月份12月份的销售额'''
months_comparison = data[(data['Month'] == 11) | (data['Month'] == 12)]

comparison = months_comparison.groupby(['Year', 'Month'])['total_price'].sum()
comparison = comparison.reset_index()

plt.figure(figsize=(8,4))
sns.barplot(data=comparison, x='Month', y='total_price', hue='Year')
plt.title('Comparison of months income by year')
plt.show()


'''统计购买数量最多的前十名用户'''
top_customers = data.groupby('CustomerID')['Quantity'].sum()
top_customers = top_customers.sort_values(ascending=False).head(10)

plt.figure(figsize=(8,4))
top_customers.plot(kind='bar', color='salmon')

plt.bar(top_customers)
plt.title('Top 10 cutomers by quantity bought')
plt.xlabel('Customer ID')
plt.ylabel('Quantity')
plt.grid()
plt.show()


data['Country'].value_counts().head()



'''展示除了英国本土的顾客外其他国家的顾客'''
countries = data['Country'].value_counts()[1:]
fig, ax = plt.subplots(figsize = (12,6))
ax.bar(countries.index, countries, color = 'salmon')
ax.set_xticklabels(countries.index, rotation = 90)
ax.set_title('Number of customers by country (excluding UK)')
ax.grid()
plt.show()


customer_df = data.groupby('CustomerID').agg({'total_price': ['mean','sum','max']})
data = pd.DataFrame(customer_df)
data.columns = ['Mean', 'Sum','Max']
data

features = customer_df.values

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
scaled_features

ks = range(1, 11)
inertias = []

for k in ks:
    # 用k个clusters:model创建一个KMeans实例
    model = KMeans(n_clusters= k)

    # 使模型与样本匹配
    model.fit(scaled_features)

    # 将惯性附加到惯性列表中
    inertias.append(model.inertia_)


plt.figure(figsize=(12,6))
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

MODEL = KMeans(n_clusters=3)
MODEL.fit(scaled_features)


data['Cluster'] = MODEL.predict(scaled_features)
data.head()

pd = px.scatter_3d(data_frame=data, x='Max', y='Mean', z='Sum', color='Cluster')
pd.show()