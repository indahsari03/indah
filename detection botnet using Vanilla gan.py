#!/usr/bin/env python
# coding: utf-8

# In[31]:


import psutil ; print(list(psutil.virtual_memory())[0:2])

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')

import xgboost as xgb

import pickle

import gc
gc.collect()
print(list(psutil.virtual_memory())[0:2])


# In[32]:


import VanillaGAN

# For reloading after making changes
import importlib
importlib.reload(VanillaGAN) 
from VanillaGAN import *


# In[33]:


#access dataset
data = pd.read_csv('BotNeTIoT-L01_label_NoDuplicates.csv')
data = data.drop(['Unnamed: 0'], axis=1)
data.head()


# In[4]:


#data populations
data = pd.read_csv('BotNeTIoT-L01_label_NoDuplicates.csv')
len(data)


# In[5]:


#creating dataframe
data_cols = ['MI_dir_L0.1_weight', 'MI_dir_L0.1_mean', 'MI_dir_L0.1_variance',
       'H_L0.1_weight', 'H_L0.1_mean', 'H_L0.1_variance', 'HH_L0.1_weight',
       'HH_L0.1_mean', 'HH_L0.1_std', 'HH_L0.1_magnitude', 'HH_L0.1_radius',
       'HH_L0.1_covariance', 'HH_L0.1_pcc', 'HH_jit_L0.1_weight',
       'HH_jit_L0.1_mean', 'HH_jit_L0.1_variance', 'HpHp_L0.1_weight',
       'HpHp_L0.1_mean', 'HpHp_L0.1_std', 'HpHp_L0.1_magnitude',
       'HpHp_L0.1_radius', 'HpHp_L0.1_covariance', 'HpHp_L0.1_pcc', 'label']
label_cols = ['label']


# In[6]:


dff = data[data_cols]
dff


# In[7]:


dfd = data[label_cols]
dfd


# In[8]:


dff.info()


# In[9]:


data['label'].value_counts()


# In[10]:


dfd = data[label_cols]
dfd


# In[11]:


dff.isnull().sum()


# In[12]:


#MinMax Scaler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
dfz = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dfz


# In[13]:


dfz.head()


# In[14]:


dfz['target'] = data['label']
dfz.head()


# In[15]:


import sklearn.cluster as cluster

train = dfz.loc[dfz['target']==1 ].copy()

algorithm = cluster.KMeans
args, kwds = (), {'n_clusters':2, 'random_state':0}
labels = algorithm(*args, **kwds).fit_predict(train[data_cols ])

print(pd.DataFrame([[np.sum(labels==i)] for i in np.unique(labels) ], columns=['count'], index=np.unique(labels) ) )

obj_classes = train.copy()
obj_classes['target'] = labels

num_cases = int(0.8 * len(train))
if(num_cases > 40000) : num_cases = 20000


# In[16]:


num_cases = 242310


# In[17]:


rand_dim = 32 # 32 # needs to be ~data_dim
base_n_count = 128 # 128

nb_steps = 10000 + 1 # 50000 # Add one for logging of the last interval
batch_size = 128 # 64

k_d = 1  # number of critic network updates per adversarial training step
k_g = 1  # number of generator network updates per adversarial training step
critic_pre_train_steps = 100 # 100  # number of steps to pre-train the critic before starting adversarial training
log_interval = 100 # 100  # interval (in steps) at which to log loss summaries and save plots of image samples to disc
learning_rate = 5e-4 # 5e-5
data_dir = 'cache_vanilagan/'
generator_model_path, discriminator_model_path, loss_pickle_path = None, None, None

#show = False
show = True 

# train = create_toy_spiral_df(1000)
# train = create_toy_df(n=1000,n_dim=2,n_classes=4,seed=0)
train = obj_classes.copy().reset_index(drop=True) # fraud only with labels from classification

# train = pd.get_dummies(train, columns=['Class'], prefix='Class', drop_first=True)
label_cols = [ i for i in train.columns if 'target' in i ]
data_cols = [ i for i in train.columns if i not in label_cols ]
train[ data_cols ] = train[ data_cols ]#/10 # scale to random noise size, one less thing to learn
train_no_label = train[ data_cols ]


# In[18]:


k_d = 1  # number of critic network updates per adversarial training step
learning_rate = 5e-4 # 5e-5
arguments = [rand_dim, nb_steps, batch_size, 
             k_d, k_g, critic_pre_train_steps, log_interval, learning_rate, base_n_count,
            data_dir, generator_model_path, discriminator_model_path, loss_pickle_path, show ]

adversarial_training_GAN(arguments, train_no_label, data_cols ) # GAN


# In[19]:


seed = 17

train = obj_classes.copy().reset_index(drop=True) # fraud only with labels from classification

# train = pd.get_dummies(train, columns=['Class'], prefix='Class', drop_first=True)
label_cols = [ i for i in train.columns if 'target' in i ]
data_cols = [ i for i in train.columns if i not in label_cols ]
train[ data_cols ] = train[ data_cols ] #/ 10 # scale to random noise size, one less thing to learn
train_no_label = train[ data_cols ]

data_dim = len(data_cols) 
label_dim = len(label_cols)
#with_class = False
#if label_dim > 0: with_class = True
np.random.seed(seed)


# In[20]:


# with_class = False
train = train_no_label
label_cols = []


# In[21]:


generator_model, discriminator_model, combined_model = define_models_GAN(rand_dim, data_dim, base_n_count)
generator_model.load_weights('cache_vanilagan/GAN_generator_model_weights_step_10000.h5')


# In[22]:


test_size = num_cases # Equal to all of the fraud cases

x = get_data_batch(train, test_size, seed=seed)
z = np.random.normal(size=(test_size, rand_dim))
#if with_class:
#    labels = x[:,-label_dim:]
#    g_z = generator_model.predict([z, labels])
#else:
g_z = generator_model.predict(z)


# In[23]:


print( CheckAccuracy( x, g_z, data_cols, label_cols, seed=0, # with_class=with_class, 
                     data_dim=data_dim ) )

PlotData( x, g_z, data_cols, label_cols, seed=0, # with_class=with_class, 
         data_dim=data_dim)


# In[24]:


real_samples = pd.DataFrame(x, columns=data_cols+label_cols)
test_samples = pd.DataFrame(g_z, columns=data_cols+label_cols)
real_samples['syn_label'] = 0
test_samples['syn_label'] = 1

training_fraction = 0.5
n_real, n_test = int(len(real_samples)*training_fraction), int(len(test_samples)*training_fraction)
train_df = pd.concat([real_samples[:n_real],test_samples[:n_test]],axis=0)
test_df = pd.concat([real_samples[n_real:],test_samples[n_test:]],axis=0)

# X_col = test_df.columns[:-(label_dim+1)]
X_col = test_df.columns[:-1]
y_col = test_df.columns[-1]
dtrain = xgb.DMatrix(train_df[X_col], train_df[y_col], feature_names=X_col)
dtest = xgb.DMatrix(test_df[X_col], feature_names=X_col)
y_real = test_df['syn_label']

xgb_params = {
    'max_depth': 4,
    'objective': 'binary:logistic',
    'random_state': 0,
    'eval_metric': 'auc', # allows for balanced or unbalanced classes 
}
xgb_test = xgb.train(xgb_params, dtrain, num_boost_round=10)

y_pred = np.round(xgb_test.predict(dtest))
y_true = y_real.values.tolist()
#print( '{:.2f}'.format(SimpleAccuracy(y_pred, y_true)) )
print( '{:.2f}'.format(SimpleAccuracy(y_pred, y_true)) )


# In[25]:


y_pred0 = xgb_test.predict(dtest)

for i in range(0,len(X_col)-1, 2):

    f, axarr = plt.subplots(1, 2, figsize=(6,2) )

    axarr[0].scatter( test_df[:n_real][X_col[i]], test_df[:n_real][X_col[i+1]], c=y_pred0[:n_real], cmap='plasma')
    axarr[0].set_title('real')
    axarr[0].set_ylabel(X_col[i+1])

    axarr[1].scatter( test_df[n_real:][X_col[i]], test_df[n_real:][X_col[i+1]], c=y_pred0[n_real:], cmap='plasma')
    axarr[1].set_title('test')
    axarr[1].set_xlim(axarr[0].get_xlim()), axarr[1].set_ylim(axarr[0].get_ylim())

    for a in axarr:
        a.set_xlabel(X_col[i])

    plt.show()


# In[26]:


colors = ['red','blue']
markers = ['^','o']
labels = ['0','1']

class_label = 'syn_label'

for i in range(0,len(X_col), 2):
    col1, col2 = i, i+1
    if i+1 >= len(X_col): continue
    
    f, axarr = plt.subplots(1, 2, figsize=(6,2) )
    for group, color, marker, label in zip( test_df[:n_real].groupby(class_label), colors, markers, labels ):
        axarr[0].scatter( group[1][X_col[col1]], group[1][X_col[col2]], label=label, c=color, marker=marker, alpha=0.2) 
    axarr[0].legend()
    axarr[0].set_title('real')
    axarr[0].set_ylabel(X_col[col2])

    for group, color, marker, label in zip( test_df[n_real:].groupby(class_label), colors, markers, labels ):
        axarr[1].scatter( group[1][X_col[col1]], group[1][X_col[col2]], label=label, c=color, marker=marker, alpha=0.2) 
    axarr[1].set_xlim(axarr[0].get_xlim()), axarr[1].set_ylim(axarr[0].get_ylim())
    axarr[1].legend()
    axarr[1].set_title('generated') ;

    for a in axarr:
        a.set_xlabel(X_col[col1])

    plt.show()


# In[27]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

gru_val = confusion_matrix(y_true, y_pred)
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(gru_val, annot=True, linewidth=0.7, linecolor='cyan', fmt='g', ax=ax, cmap="BuPu")
plt.title('Confusion Matrix')
plt.xlabel('Y predict')
plt.ylabel('Y test')
plt.show()


# In[28]:


from sklearn.metrics import classification_report
target_names = ['0', '1']
print(classification_report(y_true, y_pred, target_names=target_names))


# In[29]:


# Receiver Operating Characteristic (ROC)
# How to Use ROC Curves and Precision-Recall Curves for Classification in Python
# https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/

from sklearn.metrics import roc_curve, auc

fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_true, y_pred)
auc_keras = auc(fpr_keras, tpr_keras)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.2f})'.format(auc_keras))

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


# In[30]:


fig, ax = plt.subplots(1, 1, figsize=(8, 8))
xgb.plot_importance(xgb_test, max_num_features=20, height=0.5, ax=ax);


# In[ ]:




