 In this homework, I ran some packages with LR (logistic regression), RF (random forest), and GBM (generalized boosted regression models ). 

First, for the raw data, the column ('income') I want to work with is denoted by ">50K" and "<=50K". Since the data might not be large enough for my later practise, I combined another set of data so that I have around 50000 obversations to work with. 

For the part of LR, my 'bestlambda' came out from the function "cv.glmnet" which is the cross-validation of glmnet. During the cross-validation process, the value of alpha was set to 1 which leads to the Lasso regularization. Comparing to other values of auc (that shows the goodness of the model) with lambda values 0 and 0.1, the auc value of the 'bestlambda' is the largest among all three values. 

For the part of RF, I changed the numbers of 'max_depth' and 'num_parallel_tree' to see the effects of auc values. As the 'max_depth' has larger value, the auc value increases; as the 'num_parallel_tree' has smaller value, the auc value decreases. Then I setup the table of prediction to calculate my accuracy rate to see if the model fits well. 

For the part of the GBM, I ran the xgb.cv which also does the cross-validation for 'xgb'. I used the best iteration number as my value of nround (in function 'xgb.train'). The accuracy rate is around 0.8 which could support the idea that the model is fitted. Then with the increased value of the learning rate ('eta' in function 'xgb.train', while the default value is 0.3), the auc value decreases; with the increased value of maximum depth of the tree ('max_depth' in function 'xgb.train, while the default value is 1), the auc value decreases. 
