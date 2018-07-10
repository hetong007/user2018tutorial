XGBoost
========================================================
author: Tong He 
date: July 11th, 2018
autosize: true
width: 1920
height: 1080

Outline
========================================================

- Introduction
- Boosting
- Training
- Interpretation
- Parameter-Tuning

Introduction
========================================================

Simply 


```r
install.packages('xgboost')
```

Load in


```r
library('xgboost')
```

eXtreme Gradient Boosting
========================================================

Base Model + Boosting

Tree-based Model
========================================================

[Explanation in Animation](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/)

Why Tree-based Model
========================================================

Advantages

- Interpretable
- Efficient
- Accurate
- No need to normalize

Boosting
========================================================

What to do with a weak model?

- Make it stronger!
- **Ask for help from the others**

Boosting
========================================================

- Add simultaneously
  - random forest
- Add **in sequential**

Boosting
========================================================
incremental: true

Iterative algorithm

- Iter 1
  - predict $y$ with $f_1(x)$
  - calculate $r_1 = y - f_1(x)$
- Iter 2
  - predict $r_1$ with $f_2(x)$
  - calculate $r_2 = r_1 - f_2(x)$
- ...


Every new $f_i(x)$ improves from $f_{i-1}(x)$

Boosting
========================================================

One step further

- replace $r_1$ with $L(y, f_1(x))$
- ...
- replace $r_T$ with $L(y, \sum_t^T f_t(x))$

$$
\begin{align}
L[y, \sum_t^T f_t(x)] & = L[y, \sum_t^{T-1} f_t(x) + f_T(x)] \\
& \approx L[y, \sum_t^{T-1} f_t(x)] + g(x)f_T(x)
\end{align}
$$

Boosting
========================================================

Model

$$ pred = \sum_{t=1}^{T} f_t(x) $$

Objective

$$ Obj =  L[\sum_{t=1}^{T} f_t(x), y] $$


Why XGBoost
========================================================

- Regularization

$$ Obj = \sum_{t=1}^{T} L(f_t(x), y) + \Omega(f_t)$$

- Using both first and second order gradient

$$L(f_t(x), y) \approx g(x)f_t(x) + h(x)f^2_t(x)$$

- Prune on a full binary tree

Training
========================================================

[Human Resource Analytics](https://github.com/ryankarlos/Human-Resource-Analytics-Kaggle-Dataset)

- Moderate size
- Meaningful features
- Binary classification


```r
load('data/hr.rda')
dim(train_hr)
```

```
[1] 14999    10
```

Training
========================================================

Prepare data

- No need to normalize


```r
ind <- sample(nrow(train_hr))
train_ind <- ind[1:10000]
test_ind <- ind[10001:14999]

x <- train_hr[train_ind, -1]
# zero-based class label
y <- train_hr[train_ind, 1]

x.test <- train_hr[test_ind, -1]
y.test <- train_hr[test_ind, 1]
```

Training
========================================================

Cross Validation

![](./img/cv.png)

Training
========================================================

Cross Validation


```r
param <- list("objective" = "binary:logistic",
              "eval_metric" = "auc")

bst.cv <- xgb.cv(param = param, data = x, label = y, 
                 nfold = 3, nrounds = 10)
```

```
[1]	train-auc:0.978599+0.000447	test-auc:0.975242+0.002895 
[2]	train-auc:0.981348+0.000142	test-auc:0.978738+0.002966 
[3]	train-auc:0.983922+0.001773	test-auc:0.980184+0.001425 
[4]	train-auc:0.986486+0.000525	test-auc:0.981324+0.002428 
[5]	train-auc:0.988725+0.000744	test-auc:0.982920+0.003322 
[6]	train-auc:0.989555+0.000583	test-auc:0.983388+0.003444 
[7]	train-auc:0.990368+0.000672	test-auc:0.983852+0.003448 
[8]	train-auc:0.990825+0.000525	test-auc:0.984002+0.003121 
[9]	train-auc:0.991419+0.000424	test-auc:0.984948+0.002051 
[10]	train-auc:0.992068+0.000390	test-auc:0.985499+0.001955 
```

Training
========================================================

Cross Validation


```r
bst.cv
```

```
##### xgb.cv 3-folds
 iter train_auc_mean train_auc_std test_auc_mean test_auc_std
    1      0.9785987  0.0004474523     0.9752417  0.002895268
    2      0.9813480  0.0001415721     0.9787380  0.002966072
    3      0.9839220  0.0017727777     0.9801840  0.001425282
    4      0.9864857  0.0005247205     0.9813240  0.002428220
    5      0.9887253  0.0007435323     0.9829203  0.003321747
    6      0.9895550  0.0005831506     0.9833880  0.003443770
    7      0.9903680  0.0006723189     0.9838520  0.003448104
    8      0.9908253  0.0005250348     0.9840020  0.003121250
    9      0.9914187  0.0004242345     0.9849477  0.002051323
   10      0.9920677  0.0003897883     0.9854987  0.001954823
```

Training
========================================================

- Num of trees: `nrounds`, `eta`
- Depth of trees: `max_depth`, `min_child_weight`
- Randomness: `subsample`, `colsample_bytree`
- Penalty: `gamma`, `lambda`

Training
========================================================

Cross Validation


```r
param <- list("objective" = "binary:logistic",
              "eval_metric" = "auc",
              "max_depth" = 2, eta = 0.5)

bst.cv <- xgb.cv(param = param, data = x, label = y, 
                 nfold = 3, nrounds = 10)
```

```
[1]	train-auc:0.909498+0.002989	test-auc:0.909492+0.006007 
[2]	train-auc:0.950349+0.002721	test-auc:0.949823+0.002524 
[3]	train-auc:0.941693+0.000570	test-auc:0.941798+0.001084 
[4]	train-auc:0.960917+0.000610	test-auc:0.960067+0.002240 
[5]	train-auc:0.963842+0.000914	test-auc:0.962569+0.001243 
[6]	train-auc:0.967303+0.001027	test-auc:0.966326+0.001375 
[7]	train-auc:0.969945+0.000419	test-auc:0.968644+0.002030 
[8]	train-auc:0.971829+0.001973	test-auc:0.970051+0.001923 
[9]	train-auc:0.974385+0.001804	test-auc:0.972931+0.001828 
[10]	train-auc:0.975390+0.001844	test-auc:0.974222+0.001154 
```

Practice
========================================================

Play with parameters to see how the results change

Training
========================================================


```r
param <- list("objective" = "binary:logistic",
              "eval_metric" = "auc")
model <- xgboost(param = param, data = x, label = y, nrounds = 10)
```

```
[1]	train-auc:0.978684 
[2]	train-auc:0.981701 
[3]	train-auc:0.985328 
[4]	train-auc:0.986515 
[5]	train-auc:0.987923 
[6]	train-auc:0.988077 
[7]	train-auc:0.989356 
[8]	train-auc:0.989982 
[9]	train-auc:0.990683 
[10]	train-auc:0.991196 
```

Training
========================================================

predict


```r
pred <- predict(model, x.test)
length(pred)
```

```
[1] 4999
```

```r
require(AUC)
auc(roc(pred, as.factor(y.test)))
```

```
[1] 0.9842787
```

Practice
========================================================

Compare cross validation and test set.

Interpretation
========================================================

Feature importance


```r
importance <- xgb.importance(model = model)
importance
```

```
                Feature         Gain        Cover   Frequency
1:   satisfaction_level 0.5160098962 0.3097356470 0.216617211
2:   time_spend_company 0.1656944509 0.2123908592 0.112759644
3:      last_evaluation 0.1305349784 0.0778834751 0.145400593
4:       number_project 0.1136120785 0.2092420703 0.145400593
5: average_montly_hours 0.0696784151 0.1779476701 0.308605341
6:                sales 0.0031719049 0.0113177099 0.059347181
7:               salary 0.0006808647 0.0006908816 0.008902077
8:        Work_accident 0.0006174113 0.0007916869 0.002967359
```

Interpretation
========================================================

Feature importance


```r
xgb.plot.importance(importance)
```

![plot of chunk unnamed-chunk-11](xgboost-figure/unnamed-chunk-11-1.png)

Practice
========================================================

How parameters change the importance

Interpretation
========================================================

visualize a tree


```r
xgb.plot.tree(feature_names = colnames(x),
              model = model, trees = 0)
```

Interpretation
========================================================

Information from the tree

- Cover
  - Sum of second order gradient
- Gain
  - improvement from this split

Interpretation
========================================================

*There are too many trees!*


```r
xgb.plot.multi.trees(model)
```

Interpretation
========================================================

*Even a single tree is too large!*


```r
xgb.plot.deepness(model)
```

![plot of chunk unnamed-chunk-14](xgboost-figure/unnamed-chunk-14-1.png)

Practice
========================================================

How parameters change the depth

Parameter Tuning
========================================================

A new dataset: [Otto Group Product Classification Challenge](https://www.kaggle.com/c/otto-group-product-classification-challenge)

- Moderate size
- Anonymous features
- Multi-classification


```r
load('data/otto.rda')
dim(train_otto)
```

```
[1] 61878    94
```

Parameter Tuning
========================================================


```r
x <- train_otto[, -1]
# zero-based class label
y <- train_otto[, 1] - 1

param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 9)
```

Parameter Tuning
========================================================


```r
bst.cv <- xgb.cv(param = param, data = x, label = y, 
                 nfold = 3, nrounds = 5)
```

```
[1]	train-mlogloss:1.540260+0.001847	test-mlogloss:1.554565+0.004277 
[2]	train-mlogloss:1.281524+0.001495	test-mlogloss:1.304310+0.005395 
[3]	train-mlogloss:1.111748+0.002148	test-mlogloss:1.141550+0.005204 
[4]	train-mlogloss:0.991884+0.001905	test-mlogloss:1.027316+0.005783 
[5]	train-mlogloss:0.902271+0.001718	test-mlogloss:0.942944+0.006453 
```

Parameter Tuning
========================================================

What are tunable parameters?

Function doc:


```r
?xgb.train
```

Online doc:

- http://xgboost.readthedocs.io/en/latest/parameter.html#parameters-in-r-package

Parameter Tuning
========================================================

where to look at first

- Objective
- Metric
- eta/nrounds

Parameter Tuning
========================================================

Overfitting

- shallower trees: `max_depth`, `min_child_weight`
- stronger randomness: `subsample`, `colsample_bytree`
- stronger penalty: `gamma`, `lambda`
- domain knowledge: `monotone_constraints`
  
Parameter Tuning
========================================================

Underfitting

- deeper trees: `max_depth`, `min_child_weight`
- weaker randomness:  `subsample`, `colsample_bytree`
- weaker penalty: `gamma`, `lambda`
- parallel trees: `num_parallel_tree`

Practice
========================================================

Tune your parameters and hit mlogloss 0.5!

Parameter Tuning
========================================================

cross validation

- The silver bullet?
  - Imbalanced class
  - Time-sensitive data

Parameter Tuning
========================================================

- trial-and-error
- grid search
- automatic tuning

Go even Faster
========================================================

Histogram


```r
params <- list(tree_method = 'hist')
```

Go even Faster
========================================================

Depth-wise  v.s. Loss-wise

- depth-wise

```r
params <- list(grow_policy = 'depthwise')
```
- loss-wise

```r
params <- list(grow_policy = 'lossguide')
```

Practice
========================================================

Use histogram to speed training up

About
========================================================

- github: https://github.com/dmlc/xgboost
- forum: https://discuss.xgboost.ai
- doc: https://xgboost.readthedocs.io/en/latest

Q&A
========================================================

