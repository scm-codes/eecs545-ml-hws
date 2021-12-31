#!/usr/bin/env python
# coding: utf-8

# # HW2 Problem 2
# 

# In[1]:


# import relevant packages
import numpy as np


# In[2]:


# load train and test data
train_x = np.load('hw2p2_data/hw2p2_train_x.npy')
train_y = np.load('hw2p2_data/hw2p2_train_y.npy')
test_x = np.load('hw2p2_data/hw2p2_test_x.npy')
test_y = np.load('hw2p2_data/hw2p2_test_y.npy')


# In[3]:


# dimensions
n_train = train_x.shape[0] # No of train documents
n_test = test_x.shape[0]   # No of test documents 
d = train_x.shape[1]       # dimension of feature vector

print('n_train: {} \nn_test: {} \nd: {}\n'.format(n_train, n_test, d))


# In[4]:


### Problem 2c - computation of log(p_kj) and log(pi_k) (k = 0 or 1)

## (i) esitmation of log(p_kj) 
# computing n_k and n_kj
n_k = np.array([np.sum(train_x[train_y==k,:]) for k in range(2)])
n_kj = np.array([np.sum(train_x[train_y==k,:],axis = 0) for k in range(2)])

# computing p_kj 
alpha = 1 # Laplace smoothing constant 
p_kj = np.array([(n_kj[k,:] + alpha)/(n_k[k] + alpha*d) for k in range(2)])

# log(p_kj)
log_p_kj = np.log(p_kj)


## (ii) estimation of log(pi_k)  
pi_k = np.array([np.sum(train_y==k)/n_train for k in range(2)])
log_pi_k = np.log(pi_k)

print('log-prior \nclass 0: {} \nclass 1: {}\n'.format(log_pi_k[0],log_pi_k[1]))


# In[6]:


### problem 2d - predction of classes for test data

predicted_y = np.zeros(n_test)   # predefine 

for i in range(n_test):
    # computation of posterior for classes (k = 0,1) for each document (i = 0,1,2...)
    eta = np.array([log_pi_k[k] + np.sum(test_x[i,:]*log_p_kj[k,:])  for k in range(2)])
    
    # assign class that maximizes posterior
    predicted_y[i] = np.argmax(eta)
    
# error associated with naive-bayes classification
n_misclassified = np.sum(test_y != predicted_y)
test_error = n_misclassified/n_test*100

print('number of misclassified documents: {} \nerror in Naive Bayes classification: {} %\n'.format(n_misclassified,test_error))


# In[117]:


### problem 2e - sanity check

# select the maximum ouccuring class in training data
majority_class_train = np.argmax([np.sum(train_y==0), np.sum(train_y==1)])

# error associated with dominant class classification
majority_class_error = np.sum(test_y != majority_class_train)/n_test*100

print('error in majority class classification: {} %\n'.format(majority_class_error))

