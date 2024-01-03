import pandas as pd
import time
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import re


# 讀取資料
data = pd.read_excel('交易資料集(1).xlsx')
# 保留'QUANTITY'大於0的資料
df = data.loc[data['QUANTITY'] > 0]
# 把df裡面的資料全部改成文字類型
df = df.astype(str)
# 刪除'INVOICE_NO'和'PRODUCT_TYPE'皆相同的資料(只保留一筆)
df_unique = df.drop_duplicates(subset=['INVOICE_NO', 'PRODUCT_TYPE'])
# 把相同'INVOICE_NO'所購買的'PRODUCT_TYPE'合併在一個集合內
merged_df = df_unique.groupby('INVOICE_NO')['PRODUCT_TYPE'].apply(lambda x: ','.join(x)).reset_index()


te = TransactionEncoder()
te_ary = te.fit(merged_df['PRODUCT_TYPE'].apply(lambda x: x.split(',')))\
    .transform(merged_df['PRODUCT_TYPE'].apply(lambda x: x.split(',')))
binary_df = pd.DataFrame(te_ary, columns=te.columns_)

# print(merged_df)
# print(binary_df)

start = time.time()
frequent_itemsets = apriori(binary_df, min_support=0.01, use_colnames=True)
# frequent_itemsets = apriori(binary_df, min_support=0.03, use_colnames=True)
# 根據頻繁項集生成關聯規則
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.05)
# rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.09)
end = time.time()

# 顯示結果
print("Frequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(rules)

print(f"Apriori spend {end - start} seconds")

rules.to_csv('Apriori_output.csv', index=False)


rules['antecedents'] = rules['antecedents'].astype(str)
rules['consequents'] = rules['consequents'].astype(str)

# print(rules['antecedents'])
# print(rules['consequents'])

count = 0
ant = {}
con = {}
new_data = rules.groupby('antecedents')['consequents'].apply(lambda x: ','.join(x)).reset_index()

new_data = new_data['consequents'].unique()

for i in range(0, len(rules['antecedents'])):
    x = rules['antecedents'][i]
    x = re.findall(r'[, A-Z]', x)
    x = ''.join(x)
    ant[i] = str(x)
    # print(ant[i])
for i in range(0, len(rules['antecedents'])):
    y = rules['consequents'][i]
    y = re.findall(r'[, A-Z]', y)
    y = ''.join(y)
    con[i] = str(y)

# print(ant)
# print(con)
merged_matrix = [[ant[key], con[key]] for key in set(ant) & set(con)]

# print(merged_matrix)
ans = pd.DataFrame(merged_matrix, columns=['0', '1'])
# print(ans)

ans = ans.groupby('0')['1'].apply(lambda x: ','.join(x)).reset_index()
# print(ans)


for i in range(0, len(ans['1'])):
    # print(f"ans[{i}] = {ans['1'][i]}")
    factor = ans['1'][i].split(', ')
    # print(f"factor = {factor}")
    ans['1'][i] = [','.join(factor)]
    # print(f"ans[{i}] = {ans['1'][i]}")
    ans['1'][i] = ans['1'][i][0].split(',')
    # print(f"ans[{i}] = {ans['1'][i]}")



for i in range(0, len(ans['1'])):
    ans['1'][i] = list(set(ans['1'][i]))

# print(ans)
count = 0
user = input("Please enter the Product Type：")

for i in range(0, len(ans['0'])):
    if user == ans['0'][i]:
        print(f"Has Association rule with{ans['1'][i]}")
        count = count + 1

if count == 0:
    print("No Association rule with any product type")

