#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[1]:


import numpy as np 

import pandas as pd 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 99)

import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set()


# # Importing data

# In[2]:


data_raw= pd.read_csv('loan_data_2007_2014.csv',index_col=0, low_memory=False) 


# # Exploring data

# In[3]:


data_raw.shape


# In[4]:


data_raw.info()


# In[5]:


data_raw.sample()


# In[6]:


data_raw.id.nunique()


# In[7]:


data_raw.member_id.nunique()


# Terlihat bahwa tidak ada id atau member_id yang duplikat, artinya setiap baris sudah mewakili satu individu.

# Selanjutnya, dilakukan pembuangan fitur-fitur yang dianggap tidak berguna atau tidak memberikan informasi yang relevan dalam analisis data. Proses ini bertujuan untuk menyederhanakan dataset dan meningkatkan kualitas analisis.

# In[8]:


# List of columns to drop
columns_to_drop = [
    # Unique IDs and free text
    'id',
    'member_id',
    'url',
    'desc',

    # Columns with all null values, constant values, or other issues
    'zip_code',
    'annual_inc_joint',
    'dti_joint',
    'verification_status_joint',
    'open_acc_6m',
    'open_il_6m',
    'open_il_12m',
    'open_il_24m',
    'mths_since_rcnt_il',
    'total_bal_il',
    'il_util',
    'open_rv_12m',
    'open_rv_24m',
    'max_bal_bc',
    'all_util',
    'inq_fi',
    'total_cu_tl',
    'inq_last_12m',

    # Columns requiring expert judgment
    'sub_grade'
]


# In[9]:


data = data_raw.drop(columns_to_drop, axis=1)


# In[10]:


data.sample(5)


# # DEFINE TARGET VARIABLE / LABELING

# Tujuan utama model risiko kredit proyek adalah untuk memprediksi kemampuan seseorang untuk membayar pinjaman atau kredit yang diberikan. Oleh karena itu, variabel target yang digunakan harus menunjukkan kemampuan seseorang.

# Dalam dataset ini, variabel loan_status dapat digunakan sebagai variabel target karena menunjukkan bagaimana masing-masing individu membayar pinjaman atau kredit sebelumnya.
# 

# In[11]:


data.loan_status.value_counts(normalize=True)*100


# Seperti yang ditunjukkan, variabel loan_status memiliki berbagai nilai:
# 
# Current menunjukkan pembayaran lancar; Late menunjukkan pembayaran telat dilakukan; In Grace Period menunjukkan dalam masa tenggang; Fully Paid menunjukkan pembayaran lunas; Default menunjukkan pembayaran macet.
# 
# Menurut definisi-definisi tersebut, setiap pinjam dapat diklasifikasikan sebagai baik atau buruk.
# 
# Kebutuhan bisnis dapat menyebabkan definisi utang baik dan buruk berubah. Pada contoh ini, saya menggunakan keterlambatan pembayaran lebih dari tiga puluh hari, yang merupakan kondisi yang lebih buruk sebagai penanda kredit yang buruk.

# In[12]:


# Define a dictionary to map bad statuses to their flag values
status_mapping = {
    'Charged Off': 1,
    'Default': 1,
    'Does not meet the credit policy. Status:Charged Off': 1,
    'Late (31-120 days)': 1,
}

# Create a new column 'bad_flag' based on the loan_status using the mapping
data['bad_flag'] = data['loan_status'].map(status_mapping).fillna(0).astype(int)


# In[13]:


data['bad_flag'].value_counts(normalize=True)*100


# Setelah melakukan flagging untuk kredit buruk dan baik, ditemukan bahwa jumlah orang yang ditandai sebagai kredit buruk jauh lebih sedikit daripada yang ditandai sebagai kredit baik. Akibatnya, masalah ini menjadi masalah data yang tidak seimbang.
# 
# Tanggalkan kolom asal loan_status.
# 

# In[14]:


data.drop('loan_status', axis=1, inplace=True)


# # CLEANING, PREPROCESSING, FEATURE ENGINEERING

# Pada tahap ini, beberapa fitur dibersihkan atau diubah untuk menjadi format yang dapat digunakan untuk modeling.

# # emp_length 
# menambah atau mengurangi emp_length. Contoh: 4 tahun -> 4

# In[15]:


data['emp_length'].unique()


# In[16]:


import re

# Use regular expressions for replacing patterns in 'emp_length'
data['emp_length_int'] = data['emp_length'].str.replace(r'(\+|<| years?|\s)', '', regex=True).fillna(0).astype(int)


# In[17]:


data['emp_length_int'] = data['emp_length_int'].astype(float)


# In[18]:


data.drop('emp_length', axis=1, inplace=True)


# # term 
# Memodifikasi term. Contoh: 36 months -> 36

# In[19]:


data['term'].unique()


# In[20]:


# Use regular expression to extract the numeric part of 'term'
data['term_int'] = data['term'].str.extract(r'(\d+)').astype(int)


# In[21]:


data.drop('term', axis=1, inplace=True)


# # earliest_cr_line
# 
# mengubah earliest_cr_line dari bulan-tahun menjadi jumlah waktu yang telah berlalu sejak saat itu. Untuk melakukan hal ini, biasanya digunakan date = today. Namun, karena dataset ini mencakup data dari tahun 2007 hingga 2014, akan lebih masuk akal untuk menggunakan tanggal referensi yang berasal dari sekitar tahun 2017. Saya menggunakan tanggal 2017-12-01 sebagai referensi dalam contoh ini.

# In[22]:


data['earliest_cr_line'].head(3)


# In[23]:


# Convert 'earliest_cr_line' to datetime
data = data.assign(earliest_cr_line_date=pd.to_datetime(data['earliest_cr_line'], format='%b-%y'))

# Calculate 'mths_since_earliest_cr_line'
data = data.assign(mths_since_earliest_cr_line=((pd.to_datetime('2017-12-01') - data['earliest_cr_line_date']) / pd.Timedelta(days=30)).round())

# Display the head of the new columns
print(data['earliest_cr_line_date'].head(3))
print(data['mths_since_earliest_cr_line'].head(3))

# Display the description of 'mths_since_earliest_cr_line'
print(data['mths_since_earliest_cr_line'].describe())


# Terlihat ada nilai yang aneh, yaitu negatif.

# In[24]:


data[data['mths_since_earliest_cr_line']<0][['earliest_cr_line', 'earliest_cr_line_date', 'mths_since_earliest_cr_line']].head(3)


# Ternyata nilai negatif muncul karena fungsi Python salah menginterpretasikan tahun 62 menjadi tahun 2062, padahal seharusnya merupakan tahun 1962.
# 
# Untuk mengatasi hal ini, dapat dilakukan preprocessing lebih jauh jika ingin membenarkan tahun 2062 menjadi 1962. Namun, kali ini saya hanya mengubah nilai yang negatif menjadi nilai maximum dari fitur tersebut. Karena di sini saya mengetahui bahwa nilai-nilai yang negatif artinya adalah data yang sudah tua (tahun 1900an), maka masih masuk akal jika saya mengganti nilai-nilai tersebut menjadi nilai terbesar.

# In[25]:


data.loc[data['mths_since_earliest_cr_line']<0, 'mths_since_earliest_cr_line'] = data['mths_since_earliest_cr_line'].max()


# In[26]:


data.drop(['earliest_cr_line', 'earliest_cr_line_date'], axis=1, inplace=True)


# # issue_d
# Konsep preprocessing yang dilakukan sama dengan yang dilakukan terhadap variabel earliest_cr_line

# In[27]:


data['issue_d_date'] = pd.to_datetime(data['issue_d'], format='%b-%y')
data['mths_since_issue_d'] = round(pd.to_numeric((pd.to_datetime('2017-12-01') - data['issue_d_date']) / np.timedelta64(1, 'M')))


# In[28]:


data['mths_since_issue_d'].describe()


# In[29]:


data.drop(['issue_d', 'issue_d_date'], axis=1, inplace=True)


# # last payment d
# Konsep preprocessing yang dilakukan sama dengan yang dilakukan terhadap variabel earliest_cr_line

# In[30]:


data['last_pymnt_d_date'] = pd.to_datetime(data['last_pymnt_d'], format='%b-%y')
data['mths_since_last_pymnt_d'] = round(pd.to_numeric((pd.to_datetime('2017-12-01') - data['last_pymnt_d_date']) / np.timedelta64(1, 'M')))


# In[31]:


data['mths_since_last_pymnt_d'].describe()


# In[32]:


data.drop(['last_pymnt_d', 'last_pymnt_d_date'], axis=1, inplace=True)


# # next_pymnt_d
# Konsep preprocessing yang dilakukan sama dengan yang dilakukan terhadap variabel earliest_cr_line

# In[33]:


data['next_pymnt_d_date'] = pd.to_datetime(data['next_pymnt_d'], format='%b-%y')
data['mths_since_next_pymnt_d'] = round(pd.to_numeric((pd.to_datetime('2017-12-01') - data['next_pymnt_d_date']) / np.timedelta64(1, 'M')))


# In[34]:


data['mths_since_next_pymnt_d'].describe()


# In[35]:


data.drop(['next_pymnt_d', 'next_pymnt_d_date'], axis=1, inplace=True)


# # last_credit_pull_d
# Konsep preprocessing yang dilakukan sama dengan yang dilakukan terhadap variabel earliest_cr_line

# In[36]:


data['last_credit_pull_d_date'] = pd.to_datetime(data['last_credit_pull_d'], format='%b-%y')
data['mths_since_last_credit_pull_d'] = round(pd.to_numeric((pd.to_datetime('2017-12-01') - data['last_credit_pull_d_date']) / np.timedelta64(1, 'M')))
data['mths_since_last_credit_pull_d'].describe()


# In[37]:


data.drop(['last_credit_pull_d', 'last_credit_pull_d_date'], axis=1, inplace=True)


# # EXPLORATORY DATA ANALYSIS

# # Correlation Check

# In[38]:


plt.figure(figsize=(20,20))
sns.heatmap(data.corr())


# Di sini, jika ada pasangan fitur-fitur yang memiliki korelasi tinggi maka akan diambil salah satu saja. Nilai korelasi yang dijadikan patokan sebagai korelasi tinggi tidak pasti, umumnya digunakan angka 0.7.

# In[39]:


corr_matrix = data.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop_hicorr = [column for column in upper.columns if any(upper[column] > 0.7)]


# In[40]:


to_drop_hicorr


# In[41]:


data.drop(to_drop_hicorr, axis=1, inplace=True)


# # Check Categorical Features

# In[42]:


data.select_dtypes(include='object').nunique()


# Pada tahap ini, fitur dengan nilai unik yang sangat tinggi (high cardinality) dan fitur dengan nilai unik hanya satu dibuang.

# In[43]:


data.drop(['emp_title', 'title', 'application_type'], axis=1, inplace=True)


# In[44]:


data.select_dtypes(exclude='object').nunique()


# Ternyata, pada tipe data selain object juga terdapat fitur yang hanya memiliki satu nilai unik saja, maka akan ikut dibuang juga.

# In[45]:


data.drop(['policy_code'], axis=1, inplace=True)


# In[46]:


for col in data.select_dtypes(include='object').columns.tolist():
    print(data[col].value_counts(normalize=True)*100)
    print('\n')


# Fitur yang sangat didominasi oleh salah satu nilai saja akan dibuang pada tahap ini.

# In[47]:


data.drop('pymnt_plan', axis=1, inplace=True)


# # MISSING VALUES

# # Missing Value Checking

# In[49]:


# Calculate the percentage of missing values for each column
check_missing = data.isnull().mean() * 100

# Filter and sort columns with missing values in descending order
missing_columns = check_missing[check_missing > 0].sort_values(ascending=False)

# Display the result
print(missing_columns)


# Di sini, kolom-kolom dengan missing values di atas 75% dibuang

# In[50]:


data.drop('mths_since_last_record', axis=1, inplace=True)


# # Missing Values Filling

# In[51]:


data['annual_inc'].fillna(data['annual_inc'].mean(), inplace=True)
data['mths_since_earliest_cr_line'].fillna(0, inplace=True)
data['acc_now_delinq'].fillna(0, inplace=True)
data['total_acc'].fillna(0, inplace=True)
data['pub_rec'].fillna(0, inplace=True)
data['open_acc'].fillna(0, inplace=True)
data['inq_last_6mths'].fillna(0, inplace=True)
data['delinq_2yrs'].fillna(0, inplace=True)
data['collections_12_mths_ex_med'].fillna(0, inplace=True)
data['revol_util'].fillna(0, inplace=True)
data['emp_length_int'].fillna(0, inplace=True)
data['tot_cur_bal'].fillna(0, inplace=True)
data['tot_coll_amt'].fillna(0, inplace=True)
data['mths_since_last_delinq'].fillna(-1, inplace=True)


# # FEATURE SCALING AND TRANSFORMATION

# # One Hot Encoding
# Semua kolom kategorikal dilakukan One Hot Encoding.

# In[52]:


categorical_cols = [col for col in data.select_dtypes(include='object').columns.tolist()]


# In[53]:


onehot = pd.get_dummies(data[categorical_cols], drop_first=True)


# In[54]:


onehot.head()


# # Standardization
# Semua kolom numerikal dilakukan proses standarisasi dengan StandardScaler.

# In[55]:


numerical_cols = [col for col in data.columns.tolist() if col not in categorical_cols + ['bad_flag']]


# In[56]:


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
std = pd.DataFrame(ss.fit_transform(data[numerical_cols]), columns=numerical_cols)


# In[57]:


std.head()


# # Transformed Dataframe
# Menggabungkan kembali kolom-kolom hasil transformasi

# In[58]:


data_model = pd.concat([onehot, std, data[['bad_flag']]], axis=1)


# # MODELING
# Train-Test Split

# In[59]:


from sklearn.model_selection import train_test_split

X = data_model.drop('bad_flag', axis=1)
y = data_model['bad_flag']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[60]:


X_train.shape, X_test.shape


# # Training
# Pada contoh ini digunakan algoritma Random Forest untuk pemodelan.

# In[62]:


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(max_depth=4)
rfc.fit(X_train, y_train)


# Feature Importance dapat ditampilkan.

# In[63]:


arr_feature_importances = rfc.feature_importances_
arr_feature_names = X_train.columns.values
    
df_feature_importance = pd.DataFrame(index=range(len(arr_feature_importances)), columns=['feature', 'importance'])
df_feature_importance['feature'] = arr_feature_names
df_feature_importance['importance'] = arr_feature_importances
df_all_features = df_feature_importance.sort_values(by='importance', ascending=False)
df_all_features


# # Validation
# Untuk mengukur performa model, dua metrik yang umum dipakai dalam dunia credit risk adalah AUC dan KS.

# In[64]:


y_pred_proba = rfc.predict_proba(X_test)[:][:,1]

df_actual_predicted = pd.concat([pd.DataFrame(np.array(y_test), columns=['y_actual']), pd.DataFrame(y_pred_proba, columns=['y_pred_proba'])], axis=1)
df_actual_predicted.index = y_test.index


# # AUC

# In[65]:


from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, tr = roc_curve(df_actual_predicted['y_actual'], df_actual_predicted['y_pred_proba'])
auc = roc_auc_score(df_actual_predicted['y_actual'], df_actual_predicted['y_pred_proba'])

plt.plot(fpr, tpr, label='AUC = %0.4f' %auc)
plt.plot(fpr, fpr, linestyle = '--', color='k')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()


# # KS

# In[69]:


# Sort the DataFrame by 'y_pred_proba'
df_actual_predicted = df_actual_predicted.sort_values('y_pred_proba')

# Calculate cumulative statistics using vectorized operations
df_actual_predicted['Cumulative N Population'] = df_actual_predicted.index + 1
df_actual_predicted['Cumulative N Bad'] = df_actual_predicted['y_actual'].cumsum()
df_actual_predicted['Cumulative N Good'] = df_actual_predicted['Cumulative N Population'] - df_actual_predicted['Cumulative N Bad']
df_actual_predicted['Cumulative Perc Population'] = df_actual_predicted['Cumulative N Population'] / df_actual_predicted.shape[0]
df_actual_predicted['Cumulative Perc Bad'] = df_actual_predicted['Cumulative N Bad'] / df_actual_predicted['y_actual'].sum()
df_actual_predicted['Cumulative Perc Good'] = df_actual_predicted['Cumulative N Good'] / (df_actual_predicted.shape[0] - df_actual_predicted['y_actual'].sum())

# Reset the index if needed
df_actual_predicted = df_actual_predicted.reset_index(drop=True)


# In[70]:


df_actual_predicted.head()


# In[71]:


import matplotlib.pyplot as plt

# Calculate KS
KS = max(df_actual_predicted['Cumulative Perc Good'] - df_actual_predicted['Cumulative Perc Bad'])

# Plot Cumulative Percentage
plt.plot(df_actual_predicted['y_pred_proba'], df_actual_predicted['Cumulative Perc Bad'], color='r', label='Bad')
plt.plot(df_actual_predicted['y_pred_proba'], df_actual_predicted['Cumulative Perc Good'], color='b', label='Good')
plt.xlabel('Estimated Probability for Being Bad')
plt.ylabel('Cumulative %')
plt.title(f'Kolmogorov-Smirnov: {KS:.4f}')
plt.legend()

# Show the plot
plt.show()


# Model yang dibangun menghasilkan performa dengan AUC = 0.857 dan KS = 0.56. Biasanya, AUC di atas 0.7 dan KS di atas 0.3 menunjukkan performa yang baik.

# 

# In[ ]:




