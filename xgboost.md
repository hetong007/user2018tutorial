XGBoost
========================================================
author: Tong He 
date: July 11th, 2018
autosize: true

Outline
========================================================

- Introduction
- A tree-based model
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

![](./img/cart.png)

Tree-based Model
========================================================

[Explanation in Animation](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/)

Why Tree-based Model
========================================================

Advantages

- Interpretable
- Efficient
- Accurate

Boosting
========================================================

What to do with a weak model?

- Make it stronger!
- **Ask for help from the others**

Boosting
========================================================

Go beyond a single tree

![](./img/twocart.png)

Boosting
========================================================

Additive boosting

$$ r_1 = y - f_1(x) $$

Boosting
========================================================

Additive boosting

$$ r_2 = y - f_1(x) - f_2(x) = r_1 - f_2(x) $$

Boosting
========================================================

Additive boosting

$$ r_3 = y - f_1(x) - f_2(x) - f_3(x) = r_2 - f_3(x) $$

Boosting
========================================================

Additive boosting

$$\cdots$$

Boosting
========================================================

Iterative algorithm

- Iter 1
  - predict $y$ with $f_1(x)$
  - calculate $r_1 = y - f_1(x)$
- Iter 2
  - predict $r_1$ with $f_2(x)$
  - calculate $r_2 = r_1 - f_2(x)$
- ...
- $\sum_i f_i(x)$ is better than $f_1(x)$

Boosting
========================================================

One step further

- replace $r_1$ with $L(y, f_1(x))$
- ...
- replace $r_T$ with $L(y, \sum_t^T f_t(x))$

$$L(y, \sum_t^T f_t(x)) = L(y, \sum_t^{T-1} f_t(x) + f_T(x)) \approx L(y, \sum_t^{T-1} f_t(x)) + g(x)f_t(x)$$

Boosting
========================================================

Model

$$ pred = \sum_{t=1}^{T} \cdot f_t(x) $$

Objective

$$ Obj =  L(\sum_{t=1}^{T} f_t(x), y) $$


Why XGBoost
========================================================

- $L_1$ and $L_2$ regularization

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

x <- train_hr[train_ind,-1]
# zero-based class label
y <- train_hr[train_ind,1]

x.test <- train_hr[test_ind,-1]
y.test <- train_hr[test_ind,1]
```

Training
========================================================

define parameters


```r
param <- list("objective" = "binary:logistic",
              "eval_metric" = "auc")
```

Training
========================================================

Cross Validation


```r
bst.cv <- xgb.cv(param = param, data = x, label = y, 
                 nfold = 3, nrounds = 10)
```

```
[1]	train-auc:0.975808+0.003401	test-auc:0.972281+0.002864 
[2]	train-auc:0.977659+0.002517	test-auc:0.972277+0.001352 
[3]	train-auc:0.982610+0.002353	test-auc:0.975647+0.003157 
[4]	train-auc:0.984508+0.001708	test-auc:0.978080+0.002698 
[5]	train-auc:0.986067+0.001547	test-auc:0.979804+0.003576 
[6]	train-auc:0.986875+0.001302	test-auc:0.981943+0.003359 
[7]	train-auc:0.987745+0.001235	test-auc:0.982114+0.003335 
[8]	train-auc:0.988681+0.001241	test-auc:0.983029+0.003496 
[9]	train-auc:0.989076+0.001266	test-auc:0.982995+0.003373 
[10]	train-auc:0.991163+0.001137	test-auc:0.983716+0.003266 
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
    1      0.9758080   0.003401134     0.9722810  0.002864236
    2      0.9776590   0.002517461     0.9722773  0.001351559
    3      0.9826100   0.002352536     0.9756473  0.003157445
    4      0.9845077   0.001707747     0.9780800  0.002698321
    5      0.9860673   0.001547118     0.9798037  0.003575691
    6      0.9868753   0.001302422     0.9819427  0.003359254
    7      0.9877453   0.001234834     0.9821143  0.003334505
    8      0.9886807   0.001241175     0.9830287  0.003495935
    9      0.9890757   0.001265941     0.9829947  0.003373196
   10      0.9911630   0.001137177     0.9837160  0.003266101
```

Training
========================================================

- Num of trees: `nrounds`, `eta`
- Depth of trees: `max_depth`, `min_child_weight`
- Randomness: `subsample`, `colsample_bytree`
- Penalty: `gamma`, `lambda`

Practice
========================================================

Play with parameters to see how the results change

Training
========================================================


```r
model <- xgboost(param = param, data = x, label = y, nrounds = 10)
```

```
[1]	train-auc:0.975602 
[2]	train-auc:0.979455 
[3]	train-auc:0.983702 
[4]	train-auc:0.985109 
[5]	train-auc:0.986201 
[6]	train-auc:0.986646 
[7]	train-auc:0.987687 
[8]	train-auc:0.988145 
[9]	train-auc:0.988687 
[10]	train-auc:0.990535 
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

Practice
========================================================

Compare the evaluation on test set and cross validation.

Interpretation
========================================================

Feature importance


```r
importance <- xgb.importance(model = model)
importance
```

```
                Feature         Gain        Cover   Frequency
1:   satisfaction_level 0.5136536137 0.2798328773 0.224489796
2:   time_spend_company 0.1672146194 0.2236614448 0.116618076
3:      last_evaluation 0.1340233220 0.0782284751 0.163265306
4:       number_project 0.1081485857 0.2321557423 0.148688047
5: average_montly_hours 0.0742442506 0.1772625850 0.306122449
6:               salary 0.0013053471 0.0014265538 0.011661808
7:                sales 0.0011808129 0.0067436483 0.026239067
8:        Work_accident 0.0002294486 0.0006886734 0.002915452
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
[1]	train-mlogloss:1.541981+0.001387	test-mlogloss:1.555663+0.004200 
[2]	train-mlogloss:1.284259+0.002486	test-mlogloss:1.305575+0.004844 
[3]	train-mlogloss:1.115395+0.002894	test-mlogloss:1.143237+0.005966 
[4]	train-mlogloss:0.994528+0.003006	test-mlogloss:1.028737+0.006407 
[5]	train-mlogloss:0.902859+0.002493	test-mlogloss:0.942097+0.007646 
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

Tune your parameters and hit mlogloss 0.4!

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

Use loss-wise and histogram to speed training up

About
========================================================

- github: https://github.com/dmlc/xgboost
- forum: https://discuss.xgboost.ai
- doc: https://xgboost.readthedocs.io/en/latest

Q&A
========================================================

