#################
### LIBRARIES ###
#################

import os
import sys

import numpy as np
import pandas as pd

#############
### NUMPY ###
#############

### KONSTRUKCJA TABLICY ###

np.array([1, 2, 3, 4, 5, 6])
np.array([[1, 2, 3], [4, 5, 6]])

np.zeros([4, 3])
np.ones([4, 3])
np.diag(4 * [1])
np.arange(0, 10, 2)
np.random.rand(4, 3)

### WŁAŚCIWOŚCI TABLICY ###

x = np.random.rand(4, 3); x

x.shape
x.shape = (6, 2); x
x = x.reshape(4, 3); x

x.ndim
x.shape = (12); x
x.ndim
x.shape = (2, 2, 3); x
x.ndim
x.shape = (4, 3); x

x.dtype
np.array([1, 2, '3'])
np.array([1, 2, 3], dtype=str)
np.array([1, 2, 3]).astype(str)

### OPERATORY ###

x = np.array([1, 3, 5])
y = np.array([2, 1, 3])

# ARYMETYCZNE
x + y
x * y

# PORÓWNANIA
x <= y

# LOGICZNE
(2 < x) & (x < 6)
(x < 2) | (x > 4)

((2 < x) & (x < 6)) | ((x < 2) | (x > 4))
~(x < 3)

### INDEKSOWANIE ###

# JEDEN WYMIAR
x = np.array([0, 1, 2, 3, 4, 5]); x

x[2]
x[-2]

len(x)

x[2:5]
x[2:-2] #Bez przedostatniego i ostatniego elementu 

x[2:]
x[:2]

# WIELE WYMIARÓW
x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]); x

x[1]
x[:2]
x[[0, 2]]

x[1, 0] # x[(1, 0)] # przekazanie krotki
x[:2, 1:]
x[:, 1:]
x[[0, 2], [1, 0]]


x[:, 0] > 3
x[x[:, 0] > 3]

x > 3
x[x > 3]

x[1][0] # dwie kolejne operacje
x[:2][1:]
x[[0, 2]][:, [1, 0]]

### FUNKCJE ###

x = np.random.randint(1, 9, 10); x

help(np.random.randint)

np.random.randint(11, size=(5, 5))

# STATYSTYCZNE
np.min(x)
np.max(x)
np.mean(x)
np.median(x)
np.quantile(x, [0.1, 0.2, 0.8, 0.9])

# STATYSTYCZNE - POMOCNICZE
np.sum(x)
np.prod(x) # 4 * 1 * 6 * 3 * 1 * 8 * 3 * 6 * 7 * 3

np.cumsum(x)
np.cumprod(x)

np.sort(x)
help(np.sort)

np.sort(x)[::-1]
-np.sort(-x)


# MATEMATYCZNE

np.sqrt(x)
np.exp(x)
np.log(x)

# INSTRUKCJA WARUNKOWA

np.where(x > 5, 'TAK', 'NIE')

# ALTERNATYWNE MOŻLIWOŚCI

x.mean() # <=> np.mean(x)

x.reshape(2, 5)
x.reshape(2, 5).mean(axis = 0)
x.reshape(2, 5).mean(axis = 1)

#######################
### PANDAS - SERIES ###
#######################

### TWORZENIE ###

pd.Series([3, 2, 6, 1]) # np.array([3, 2, 6, 1]).dtype
pd.Series([3, 2, 6, 1], index = ['st1', 'st2', 'st3', 'st4'])
pd.Series({'st1' : 3, 'st2' : 2, 'st3' : 6, 'st4' : 1})

### WŁAŚCIWOŚCI ###

x = pd.Series({'st1' : 3, 'st2' : 2, 'st3' : 6, 'st4' : 1}); x

x.values
type(x.values)

x.index
type(x.index)

x.name
x.name = 'pts'; x

x.index.name
x.index.name = 'st'; x

x.dtype
x.astype(float)
### OPERATORY ###

x1 = pd.Series({'st1' : 3, 'st2' : 2, 'st3' : 6, 'st4' : 1}); x1
y1 = pd.Series({'st1' : 5, 'st2' : 6, 'st3' : 2, 'st4' : 2}); y1

x2 = pd.Series({'st1' : 3, 'st3' : 6, 'st4' : 1}); x2
y2 = pd.Series({'st1' : 5, 'st2' : 6, 'st3' : 2}); y2

# ARYMETYCZNE
x1 + y1
x1 * y1

x2 + y2
x2 * y2

pd.concat([x2, y2], axis=1).sum(axis=1)

# PORÓWNANIA
x1 <= y1
# x2 <= y2

# LOGICZNE
(2 < x1) & (x1 < 6)
(x1 < 2) | (x1 > 4)
~(x1 < 3)

### INDEKSOWANIE ###

x = pd.Series({'st1' : 3, 'st2' : 2, 'st3' : 6, 'st4' : 1}); x

x['st2']
x[['st2', 'st1', 'st4']]
x['st2':'st4']
x.sort_values()['st2':'st4']


x[x > 2]

x[2]
x = pd.Series({0 : 3, 1 : 2, 3 : 6, 2 : 1}); x
x[2]
x.iloc[2] # po pozycji

x.reset_index(drop=True)

### METODY ###

x = pd.Series({'st1' : 3, 'st2' : np.nan, 'st3' : 6, 'st4' : 1, 'st5' : 7}); x

# MODYFIKOWANIE
x = x.append(pd.Series({'st6' : 4})); x
x.drop('st1'); x #inplace=False
x1 = x.drop('st1'); x1
x.drop('st6', inplace = True); x

# SORTOWANIE
x.sort_values()
x.sort_values(inplace = True); x
x.sort_index(inplace = True); x

# BRAKI DANYCH
x.isnull()
x.notnull()
x.fillna(0)
x.dropna(inplace = True); x

# STATYSTYKA
x.describe()
x.count()
x.mean()
x.min()
x.median()
x.quantile([0.25, 0.75])
x.max()
x.var()
x.std()
x.kurt() # kurtoza
x.skew() # skosnosc

# STATYSTYKA - SZEREGI CZASOWE
x.diff()
x.pct_change()
x.cummin()
x.cummax()
x.cumprod()
x.cumsum()

# STATYSTYKA - POMOCNICZE
x.sum()
x.prod()
x.rank()
x.sample(3)

x = pd.Series({'st1' : 3, 'st2' : np.nan, 'st3' : 6, 'st4' : 1, 'st5' : 7}); x
# ITEROWANIE
x.apply(lambda x: 0 if np.isnan(x) else x**2) # w seriach lub ramkach danych
x.map(lambda x: 0 if np.isnan(x) else x**2) # w seriach

###########################
### PANDAS - DATA FRAME ###
###########################

### TWORZENIE ###

data = {
    'species' : ['cat', 'cat', 'cat', 'dog', 'dog', 'dog'],
    'age' : [9, 12, 5, 3, 9, 6],
    'weight' : [3.5, 6.1, 4.5, 8.1, 6.6, 12.3]
}

pd.DataFrame(data)
pd.DataFrame(data, columns = ['species', 'weight', 'age'])
pd.DataFrame(
    np.array([
        ['cat', 'cat', 'cat', 'dog', 'dog', 'dog'],
        [9, 12, 5, 3, 9, 6],
        [3.5, 6.1, 4.5, 8.1, 6.6, 12.3]
    ]).T,
    columns = ['species', 'weight', 'age']
)

data = pd.DataFrame(
    data,
    columns = ['species', 'weight', 'age'],
    index = ['Tigger', 'Kitty', 'Smokey', 'Charlie', 'Max', 'Buddy']
); data

### WŁAŚCIWOŚCI ###

data.columns
data.index
data.values

type(data.columns)
type(data.index)
type(data.values)

data.shape
len(data)

### OPERATORY ###

# ARYMETYCZNE
extra = pd.DataFrame({
    'species' : {'Tigger' : 'cat', 'Smokey' : 'cat', 'Charlie' : 'dog', 'Rocky' : 'dog'},
    'size' : {'Tigger' : 0.4, 'Smokey' : 0.5, 'Charlie' : 0.7, 'Rocky' : 0.9},
    'weight' : {'Tigger' : 4.0, 'Smokey' : 5.2, 'Charlie' : 7.6, 'Rocky' : 12.5}
}); extra

data
extra

data + extra

X = data[['weight', 'age']]; X

# .loc - indeksowanie wierszy/kolumn po nazwie (lub kolejnosci indeksow)

y1 = data.loc['Smokey', ['age', 'weight']]; y1
X - y1
y2 = data['age']; y2
# X - y2 # ŹLE
X.sub(y2, axis = 'index')

# PORÓWNANIA
data[['weight', 'age']] < 5

# LOGICZNE
(data[['weight', 'age']] > 3) & (data[['weight', 'age']] < 5)

# OR -> |
# NOT -> ~
# AND -> & 

### INDEKSOWANIE ###

# KOLUMNA
data['age']
data.age

# KOLUMNY
data[['age', 'weight']]
data['species']

type(data[['age', 'weight']])
type(data['species'])
# WIERSZE
# data['Kitty'] # ŹLE
# data[:, 'Kitty'] # ŹLE
data.loc['Kitty']
data.loc['Kitty':'Max']

data.iloc[1:5]

# KOLUMNY RAZ JESZCZE
data.loc[:, 'age']
data.loc[:, 'weight':'age'] # <=> data.iloc[:, 1:3]
# data.loc[:, 'weight':'species']
data.iloc[:, 2]
data.iloc[:, 1:2]

type(data.iloc[:, 2])
type(data.iloc[:, 1:2])

# iloc - indeksowanie wierszy/kolumn wg ich kolejnosci

# MODYFIKACJA
data.reindex(['Tigger', 'Kitty', 'Smokey'])
data.reindex(['Tigger', 'Lucky', 'Kitty', 'Misty', 'Smokey'])
data.reindex(columns = ['age', 'size', 'weight', 'color'])

data.rename(index = lambda name: 'pet_' + name)
data.rename(columns = lambda name: name + '_202011')

data.rename({'species':'new_species', 'weight':'new_weight'}, axis=1).head(2)
data.rename(columns={'species':'new_species', 'weight':'new_weight'}).head(2)
data.rename({'species':'new_species', 'weight':'new_weight'}).head(2)

data1 = data.rename({'Tigger': 1, 'Kitty' : 2}).head(2)
data1.rename({'1':'new_1', '2':'new_2'}).head(2)
data1.rename({1: 'new_1', 2: 'new_2'}).head(2)

# SORTOWANIE
data.sort_index()
data.sort_index(ascending = False)
data.sort_index(axis = 1)

# GRUPOWANIE PO INDEKSIE
data.set_index('species').mean(level = 'species')

# MULTI-INDEKSY
data.index.name = 'name'
tmp = data.reset_index().set_index(['name', 'species']); tmp

tmp.unstack()
tmp.stack()

### MODYFIKACJA ###

# KOLUMNA - DODANIE
data['size'] = [0.6, 0.8, 0.4, 0.9, 0.7, 1.1]; data
data['date'] = '2020-06-20'; data
data['color'] = pd.Series({'Tigger' : 'black', 'Charlie' : 'brown', 'Rocky' : 'white'}); data

# KOLUMNA - MODYFIKACJA
data['date'] = '2020-06-01'; data

# KOLUMNA - USUNIĘCIE
del data['color']; data
data.drop('date', axis = 1, inplace = True); data

help(data.drop)
# OBSERWACJA - DODANIE
data = data.append(pd.Series({'weight' : 8, 'age' : 14}, name = 'Rocky')); data

# OBSERWACJA - MODYFIKACJA
data.loc['Rocky'] = ['dog', 9, 12, 1.1]; data

# OBSERWACJA - USUNIĘCIE
data.drop('Rocky', inplace = True); data

# WARTOŚĆ - MODYFIKACJA
data.loc['Buddy', 'weight'] = 12.1; data

### SORTOWANIE ###

data.sort_values(by = 'age', ascending = False)
data.sort_values(by = ['species', 'age'])

### OBLICZENIA ###

data = pd.DataFrame({
    'cl1' : {'st1' : 5, 'st2' : 3, 'st3' : 3, 'st4' : 3},
    'cl2' : {'st1' : 4, 'st2' : 5, 'st3' : 5, 'st4' : 4},
    'cl3' : {'st1' : 4, 'st2' : 4, 'st3' : 5, 'st4' : 3}
}); data

data.mean()
data.mean(axis = 1)

data.agg(['min', 'mean', 'max'])
data.agg([np.min, np.mean, np.max])
data.agg([np.min, np.mean, np.max], axis = 1)
data.agg(lambda x: np.max(x) - np.min(x))

### ITEROWANIE ###

data.apply(lambda x: str(min(x)) + ' - ' + str(max(x)))
data.apply(lambda x: str(min(x)) + ' - ' + str(max(x)), axis = 1)

data['min_max'] = data.apply(lambda x: str(min(x)) + ' - ' + str(max(x)), axis = 1); data
del data['min_max']

data.apply(lambda x: pd.Series([x.min(), x.max()], index = ['min', 'max']))

data.cl1.map(lambda x: x + 1 if x < 5 else 5)
data.applymap(lambda x: x + 1 if x < 5 else 5)

########################################
### PANDAS - CZYTANIE / ZAPIS DANYCH ###
########################################

### CZYTANIE DANYCH ###

os.chdir('E:/Sages/Python2')

help(pd.read_csv)

pd.read_csv('../materialy/diamonds.csv', sep = ',', header = 0, index_col = 0)
pd.read_csv('../materialy/diamonds.csv', index_col = ['color', 'cut']).sort_index()
# skiprows, na_values, encoding, nrows

### ZAPIS DANYCH ###

dm = pd.read_csv('../materialy/diamonds.csv', sep = ',', header = 0, index_col = 0)
dm.head(5)
dm.tail(5)

dm.to_csv('./tmp5.csv', index = False)
pd.read_csv('./tmp5.csv')

dm.to_excel('./tmp5.xlsx', index = False)
pd.read_excel('./tmp5.xlsx')

data = pd.DataFrame({
    'species' : ['cat', 'cat', 'cat', 'dog', 'dog', 'dog'],
    'age' : [9, np.nan, 5, 3, 9, 6],
    'weight' : [3.5, 6.1, 4.5, np.nan, 6.6, 12.3]
}, index = ['Tigger', 'Kitty', 'Smokey', 'Charlie', 'Max', 'Buddy']); data

data.to_csv(
    sys.stdout, columns = ['age', 'weight'],
    sep = '_', na_rep = 'NIL', index = True, header = True
)

data.to_csv(
    './tmp.csv', columns = ['age', 'weight'],
    sep = '_', na_rep = 'NIL', index = True, header = True
)

pd.read_csv('./tmp.csv')

#####################################
### PANDAS - PRZETWARZANIE DANYCH ###
#####################################

os.chdir('E:/Sages/Python2')

wig = pd.read_csv('../materialy/wig_d.csv', sep = ',', header = 0)
data = pd.read_csv('../materialy/diamonds.csv', sep = ',', header = 0, index_col = 0)

### ZMIANA STRUKTURY DANYCH ###

long_data = wig.melt(
    'Date', ['Open', 'High', 'Low', 'Close'],
    var_name = 'type', value_name = 'price'
); long_data

wide_data = long_data.pivot('Date', 'type', 'price'); wide_data

### ZLICZANIE ###

data.cut.value_counts()
data.cut.value_counts(normalize=True)

data.cut.unique()
pd.unique(data['cut'])

pd.cut(
    data.price,
    bins = [0, 5000, 10000, 15000, 20000],
    labels = ['very_low', 'low', 'high', 'very_high']
).value_counts()

pd.qcut(data.price, 4).value_counts()

### GRUPOWANIA ###

data.price.groupby(data.cut).agg(['mean', 'median'])
data.price.groupby([data.color, data.clarity]).mean().unstack()

data.groupby('cut').mean()
data.groupby('cut').agg(['mean', 'median']).stack()

data.groupby('cut').agg(
    avg_price = ('price', 'mean'), 
    med_depth = ('depth', 'median')
).reset_index()

data.groupby(['cut', 'color']).agg(
    avg_price = ('price', 'mean'), 
    med_depth = ('depth', 'median')
).reset_index()


data.groupby('cut')[['carat', 'price']].mean()

{
    (col, cla) : [len(val), round(val.price.mean())]
    for (col, cla), val in data.groupby(['color', 'clarity'])
}

### ŁĄCZENIE ###

# CASE 1
df1 = pd.DataFrame({'key' : ['a', 'a', 'a', 'b', 'b', 'c'], 'val1' : range(6)})
df2 = pd.DataFrame({'key' : ['a', 'c', 'd'], 'val2' : range(3)})
df3 = pd.DataFrame({'key' : ['d', 'e', 'f', 'g'], 'val1' : range(4)})

df1
df2

pd.merge(df1, df2)
pd.merge(df1, df2, on = 'key')

pd.merge(df1, df2, how = 'left')
pd.merge(df1, df2, how = 'right') # pd.merge(df2, df1, how = 'left')
pd.merge(df1, df2, how = 'outer') # full join

# CASE 2
df1 = pd.DataFrame({'k1' : ['a', 'a', 'a', 'b', 'b', 'c'], 'val' : range(6)})
df2 = pd.DataFrame({'k2' : ['a', 'c', 'd'], 'val' : range(3)})

df1
df2

# pd.merge(df1, df2) # ŹLE
pd.merge(df1, df2, left_on = 'k1', right_on = 'k2')
pd.merge(df1, df2, left_on = 'k1', right_on = 'k2', suffixes = ['_v1', '_v2'])

# CASE 3
df1 = pd.DataFrame({'k1' : ['a', 'a', 'a', 'b', 'b', 'c'], 'val1' : range(6)})
df2 = pd.DataFrame({'val2' : range(3)}, index = ['a', 'c', 'd'])

df1
df2

# pd.merge(df1, df2) # ŹLE
pd.merge(df1, df2, left_on = 'k1', right_index = True, how = 'outer')

# CASE 4
df1 = pd.DataFrame({'val1' : range(6)}, index = ['a', 'a', 'a', 'b', 'b', 'c'])
df2 = pd.DataFrame({'val2' : range(3)}, index = ['a', 'c', 'd'])

df1
df2

# pd.merge(df1, df2) # ŹLE
pd.merge(df1, df2, left_index = True, right_index = True, how = 'outer')

df1.merge(df2, left_index = True, right_index = True, how = 'outer')

df1.join(df2, how = 'outer')

# case insensitive

df1 = pd.DataFrame({'key' : ['a', 'a', 'a', 'b', 'b', 'c'], 'val1' : range(6)})
df2 = pd.DataFrame({'key' : ['a', 'c', 'd'], 'val2' : range(3)})
df3 = pd.DataFrame({'key' : ['d', 'e', 'f', 'g'], 'val1' : range(4)})

df2['key'] = df2['key'].str.upper()

df1.merge(df2, on='key', how='left')
df1.merge(df2, left_on='key', right_on=df2['key'].str.lower(), how='left')
# łączenie poziome / pionowe

df1
df3

pd.concat([df1, df3], axis=0, ignore_index=True)
pd.concat([df1, df2], axis=0, ignore_index=True)
pd.concat([df1, df2], axis=1)
pd.concat([df1, df2], axis=1, ignore_index=True)
pd.concat([df1, df3], axis=1)
### MODYFIKACJA KOLUMN ###
dft = df1.copy()

dft['val1'] *= 2; dft
dft['new'] = dft['val1'] * 10; dft

# pandas assign, jak dplyr mutate

dft = dft.assign(new2=dft['val1'] * 9, inplace=True)

dft = dft.assign(temp_f=lambda dfx: dfx['new2'] * 9 / 5 + 32,
          temp_k=lambda dfx: (dfx['temp_f'] +  459.67) * 5 / 9)

dft = dft.assign(new2_40=lambda dfx: dfx['new2'].map(lambda x: True if x > 40 else False))

# alternatywnie
for index, row in dft.iterrows():
    if row['new2'] > 40:
        dft.loc[index, 'new2_40_1'] = True
    else:
        dft.loc[index, 'new2_40_1'] = False
dft
### FILTROWANIE ###

dft[dft['temp_k'] > 300]
dft[(dft['temp_k'] > 300) & (dft['new2'] < 80)]
dft[dft['key'] == 'b']

# pandas query jak dplyr filter

dft.query('temp_k > 300')
dft.query('temp_k > 300 & new2 < 80')
dft.query('20 < new2 < 80')
dft.query('key == "b"')

# filtrowanie - problem

dft_filtered = dft.query('temp_k > 300')
dft_filtered['new_col'] = 1

dft_filtered = dft_filtered.assign(new_col = 1)

dft_filtered = dft.query('temp_k > 300').copy()
dft_filtered['new_col'] = 1
#---

dfb = pd.concat([df1, df2], axis=1); dfb

dfb.query('val2 == 1')
dfb.query('~val2.isnull()', engine='python') # numexpr
dfb.query('val2 is not null')

# matplotlib

import matplotlib.pyplot as plt
dft.groupby('key')['new'].sum().plot(kind='pie')
plt.show()

dfx.query(
    'query'
).assign(
    'ddd'
).groupby...
    
dfx \
    .query('query') \
    .assign('ddd') \
    .groupby...
    
 # RDS w Python   
    
!pip install pyreadr

import pyreadr

result = pyreadr.read_r('../materialy/age.rds') # also works for RData

df = result[None]
