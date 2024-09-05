#!/usr/bin/env python
# coding: utf-8

# # **About Dataset**

# # **Context**
# 
# The growth of supermarkets in most populated cities are increasing and market competitions are also high. 
# The dataset is one of the historical sales of supermarket company which has recorded in 3 different branches for 3 months data.
# Predictive data analytics methods are easy to apply with this dataset.
# 
# 

# # Objective
# 
# The objective of this project is to conduct a comprehensive Exploratory Data Analysis (EDA) on the sales data of a supermarket chain operating across three branches in highly populated cities. This analysis aims to uncover insights into sales trends, customer behavior, and product performance over a three-month period.

# # **Attribute information**
# 
# *Invoice id*: Computer generated sales slip invoice identification number
#     
# *Branch*: Branch of supercenter (3 branches are available identified by A, B and C).
#     
# *City*: Location of supercenters
#     
# *Customer type*: Type of customers, recorded by Members for customers using member card and Normal for without member card.
#     
# *Gender*: Gender type of customer
#     
# *Product line*: General item categorization groups - Electronic accessories, Fashion accessories, Food and beverages, Health
#                 and beauty, Home and lifestyle, Sports and travel
#         
# *Unit price*: Price of each product in $
#     
# *Quantity*: Number of products purchased by customer
#     
# *Tax*: 5% tax fee for customer buying
#     
# *Total*: Total price including tax
#     
# *Date*: Date of purchase (Record available from January 2019 to March 2019)
#     
# *Time*: Purchase time (10am to 9pm)
#     
# *Payment*: Payment used by customer for purchase (3 methods are available – Cash, Credit card and Ewallet)
#     
# *COGS*: Cost of goods sold
#     
# *Gross margin percentage*: Gross margin percentage
#     
# *Gross income*: Gross income
#     
# *Rating*: Customer stratification rating on their overall shopping experience (On a scale of 1 to 10)
# 

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')


# In[ ]:





# In[2]:


market = pd.read_csv("C:\\Users\\Ishwari\\Downloads\\supermarket_sales - Sheet1.csv")


# In[3]:


market.head()


# In[4]:


market.tail()


# In[5]:


market.shape


# In[6]:


market.info()


# In[7]:


market.columns


# In[8]:


market.columns=market.columns.str.lower().str.strip()


# In[9]:


market.columns


# # Data Cleaning

# In[10]:


market['date']=pd.to_datetime(market['date'])


# In[11]:


market['time']=pd.to_datetime(market['time'])


# In[12]:


market.info()


# In[13]:


market.isnull().sum()


# In[14]:


market.duplicated().sum()


# In[15]:


for column in market.columns:
    print(column,':',market[column].nunique())


# In[16]:


market.columns


# In[17]:


market.city.unique()


# In[18]:


market.head()


# In[19]:


sns.countplot(data=market,x='gender')


# In[20]:


pyy=market.groupby('payment',as_index=False)['gross income'].sum().sort_values('gross income',ascending=False)


# In[21]:


px.bar(data_frame=pyy,x='payment',y='gross income')


# In[22]:


pyy=market.groupby('city',as_index=False)['gross income'].sum().sort_values('gross income',ascending=False)


# In[23]:


px.pie(data_frame=pyy,names='city',values='gross income')


# In[24]:


market.head()


# In[25]:


market.info()


# In[80]:


date_city=market[['date','city']]


# In[81]:


date_city=date_city.set_index('date')
date_city


# In[82]:


date_city=date_city.city.resample(rule='1M').nunique()
date_city


# In[29]:


date_city.head()


# In[30]:


px.line(data_frame=date_city,y='city')


# In[31]:


fig=px.bar(market.groupby('customer type')['gross income'].count().sort_values(ascending=False),text_auto=True,title='customer gross income')
fig.show()


# In[32]:


market.head()


# In[33]:


total=market[['date','gross income','city']]


# In[34]:


total=total.set_index('date')


# In[35]:


total.head()


# In[36]:


total=total.resample('1M').agg({'city':'nunique','gross income':'sum'})


# In[37]:


total.head(10)


# In[38]:


px.line(data_frame=total)


# In[39]:


fig=px.bar(data_frame=total,x=total.index,y=round(total['gross income'],0),color=total.city,text_auto=True)
fig.show()


# In[40]:


market.head()


# In[41]:


# Group the data by Product Name and sum up the sales by product
product_group = market.groupby('product line',as_index=False)['gross income'].sum().sort_values('gross income',ascending=False)


# In[42]:


px.bar(data_frame=product_group,x='product line',y='gross income')


# In[43]:


# Group the data by Product Name and sum up the sales by product
product_grouppM = market.groupby('product line',as_index=False)['gross margin percentage'].sum().sort_values('gross margin percentage',ascending=False)


# In[45]:


Gender_Payment = pd.crosstab(market['gender'], market['payment'])
plt.figure(figsize=(8, 6))
Gender_Payment.plot(kind='bar')
plt.title('Gender of Payment')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.show()


# In[83]:


px.bar(data_frame=product_grouppM,x='product line',y='gross margin percentage')


# In[49]:


Gender_Payment = pd.crosstab(market['gender'], market['payment'])
Gender_Payment


# In[48]:


Gender_Payment = pd.crosstab(market['gender'], market['payment'])
plt.figure(figsize=(8, 6))
Gender_Payment.plot(kind='bar')
plt.title('Gender of Payment')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.show()


# In[50]:


City_Customer_type = pd.crosstab(market['customer type'], market['city'])
City_Customer_type


# In[46]:


City_Customer_type = pd.crosstab(market['customer type'], market['city'])
plt.figure(figsize=(8, 6))
City_Customer_type.plot(kind='bar')
plt.title('City&Customer type')
plt.xlabel('Customer type')
plt.ylabel('Count')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.show()


# In[52]:


city_payment =  pd.crosstab(market['city'], market['payment'])
city_payment


# In[51]:


city_payment =  pd.crosstab(market['city'], market['payment'])
plt.figure(figsize=(10, 8))
city_payment.plot(kind='bar')
plt.title('City&Payment')
plt.xlabel('City')
plt.ylabel('Count')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.show()


# In[54]:


city_Product_line = pd.crosstab(market['product line'], market['city'])
city_Product_line


# In[53]:


city_Product_line = pd.crosstab(market['product line'], market['city'])
plt.figure(figsize=(10, 8))
city_Product_line.plot(kind='bar')
plt.title('City&Product line')
plt.xlabel('Product line')
plt.ylabel('Count')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()


# In[55]:


fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 6))

ax1.pie(market['gender'].value_counts(), labels=market['gender'].value_counts().index, colors=['#2a77db','#e01bbf'], autopct='%1.1f%%')
ax1.set_title('Gender')

ax2.pie(market['customer type'].value_counts(), labels=market['customer type'].value_counts().index,colors=['#76B7B2', '#59A14F']
        , autopct='%1.1f%%')
ax2.set_title('Customer Type')

ax3.pie(market['branch'].value_counts(), labels=market['branch'].value_counts().index, colors=['#EDC948', '#E15759', '#B07AA1'],
        autopct='%1.1f%%')
ax3.set_title('Branch')

ax4.pie(market['payment'].value_counts(), labels=market['payment'].value_counts().index,colors=['#FF9F40', '#FFCD56', '#36A2EB'],
        autopct='%1.1f%%')
ax4.set_title('Payment method')

plt.tight_layout()
plt.show()


# # show relation between city and Total sales

# In[56]:


fig, axes = plt.subplots(1, 2, figsize=(12,5))

ax1 = sns.barplot(data=market, x='city', y='total', hue = 'branch',palette="ch:start=.2,rot=-.3", width=0.5, ax=axes[0])
axes[0].legend(title = 'Branch',bbox_to_anchor=(1,1))
axes[0].set_title("sales in city's branches")

ax2 = sns.countplot(x='city', hue='gender', data=market, edgecolor="#4D4D4D", palette="ch:start=.2,rot=-.3", ax=axes[1])
axes[1].legend(title='gender', bbox_to_anchor=(1, 1))
axes[1].set_title("Sales'count with gender")

for i in ax1.containers:
    ax1.bar_label(i,label_type='center',color = '#FF7F50', fmt='%.0f')

for i in ax2.containers:
    ax2.bar_label(i,label_type='center',color = '#FF7F50')

plt.tight_layout()
plt.show()


# # Show which gender buys more

# In[57]:


fig,axes=plt.subplots(1,2,figsize=(9,5))

ax1=sns.barplot(data = market,x='gender',y='total',palette="ch:start=.2,rot=-.3",width=0.5,ax=axes[0])
axes[0].set_title("Gender's Sales")

ax2 = sns.countplot(data= market,x='gender' , hue='customer type',palette="ch:start=.2,rot=-.3",ax = axes[1])
axes[1].legend(title = 'Customer Type',bbox_to_anchor = (1,1))
axes[1].set_title('Customer Type with Gender')

for i in ax1.containers:
    ax1.bar_label(i,label_type='center',color = '#FF7F50', fmt='%.0f')
for i in ax2.containers:
    ax2.bar_label(i,label_type='center',color = '#FF7F50', fmt='%.0f')
    
plt.tight_layout()
plt.show()


# # Month with the most sales

# In[62]:


ax1=sns.barplot(x=market['date'].dt.month.map({1:'January',2:'February',3:'March'}),y=market['total'],palette="ch:start=.8,rot=-.6",width=0.5)

for i in ax1.containers:
    ax1.bar_label(i,label_type='center',color = 'Red', fmt='%.0f')
    
plt.xlabel('Month')  
plt.ylabel('Total')  
plt.title('Total Sales by Month')  
plt.show()


# # Day with Most Sales

# In[63]:


palette = {'January': 'Red', 'February': 'Blue', 'March': 'Green'}
months = ['January', 'February', 'March']

fig, axes = plt.subplots(3, 1, figsize=(15, 8))  # 3 rows, 1 column

for i, month in enumerate(months):
    month_number = i + 1
    month_data = market[market['date'].dt.month == month_number]
    
    sns.lineplot(x=month_data['date'].dt.day,y=month_data['total'],ax=axes[i],color=palette[month],ci=None)
    
    axes[i].set_xlabel('Day', fontsize=12)
    axes[i].set_ylabel('Total Sales', fontsize=12)
    axes[i].set_title(f'Total Sales by Day for {month}', fontsize=14)
    axes[i].grid(True, linestyle='--', alpha=0.7)

    axes[i].set_xlim(1, 31)
    axes[i].set_xticks(np.arange(1, 32))  
    axes[i].set_xticklabels(np.arange(1, 32), rotation=45, ha='right')
    
plt.tight_layout()
plt.show()


# # Day in week with the most sales

# In[64]:


data_temp = market.copy()
data_temp['Day'] = market['date'].dt.day_name()
day_data = data_temp.groupby('Day')['total'].sum().to_frame()
day_data = day_data.reindex(['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'])
day_data.head(10)


# In[67]:


plt.figure(figsize=(10,6))
days = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
ax = sns.barplot(x=days,y=day_data['total'],data = day_data,palette = "ch:start=.5,rot=.2")

for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='baseline', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')

ax.set_title('Weekly Sales of Supermarket', size=16)
ax.set_xlabel('Week Day')
ax.set_ylabel('Total Sales', fontsize=14)
ax.set_xticklabels(days, fontsize=14)
plt.show()


# In[68]:


num_cols = market.select_dtypes(include=['number'])
correlation_matrix = num_cols.corr()

plt.figure(figsize=(6, 4))
sns.heatmap(correlation_matrix, annot=True, linewidths=0.5, fmt='.2f')
plt.title('Correlation Heatmap of supermarket Dataset')
plt.show()


# In[69]:


plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
market.groupby('product line').total.mean().plot.pie(autopct='%1.1f%%',explode=[0.09,0.03,0.08,0.05,0.08,0.07],shadow=True, colors=["#FFA07A", "#FFE4B5", "#A52A2A", "#FF8C00", "#FFFF00", "#D2691E"])
plt.ylabel('')
plt.subplot(1,2,2)
market.groupby('product line').total.mean().plot.bar(color=["#FFA07A", "#FFE4B5", "#A52A2A", "#FF8C00", "#FFFF00", "#D2691E"])


# In[70]:


sns.pairplot(data=market)


# In[71]:


sns.set_style('darkgrid')
ax = sns.lineplot(x=market['product line'],y=market['quantity'],data=market,hue=market['gender'],err_style=None)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.title('Number of Products bought by Different Genders from Different Product Lines')
plt.legend(title='Gender',loc='upper right')
plt.show()
warnings.filterwarnings('ignore')


# In[72]:


plt.figure(figsize=(10,5))
sns.lineplot(data=market,y='rating',x='product line')
plt.show()


# In[73]:


market.head(10)


# In[77]:


plt.figure(figsize=(10,5))
sns.barplot(data=market,y='tax 5%',x='payment')
plt.show()


# In[ ]:




