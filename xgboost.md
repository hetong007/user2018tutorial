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

Xtreme Gradient Boosting
========================================================

Base Model + Boosting

Tree-based Model
========================================================

![](./img/cart.png)

Tree-based Model
========================================================

[Explanation in Animation](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/)

Tree-based Model
========================================================

![](./img/twocart.png)

Why Tree-based Model
========================================================

Advantages :

- Interpretable
- Efficient
- Accurate

Why XGBoost
========================================================

- $L_1$ and $L_2$ regularization
- Using both first and second order gradient
- Prune on a full binary tree

Training
========================================================

[Otto Group Product Classification Challenge](https://www.kaggle.com/c/otto-group-product-classification-challenge)

- Moderate size
- Anonymous features
- Multi-class classification


```r
load('otto.rda')
dim(train_otto)
```

```
[1] 61878    94
```

```r
dim(test_otto)
```

```
[1] 144368     93
```

Training
========================================================

Prepare data

- No need to normalize


```r
x <- train_otto[,-1]
y <- train_otto[,1] - 1

x.test <- test_otto
```


Training
========================================================

define parameters


```r
numberOfClasses <- max(y) + 1
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = numberOfClasses,
              "nthread" = 4)
```

Training
========================================================

Cross Validation


```r
system.time({
  bst.cv <- xgb.cv(param = param, data = x, label = y, 
                   nfold = 3, nrounds = 5)
})
```

Training
========================================================

Cross Validation


```r
bst.cv
```

Training
========================================================


```r
system.time({
  model <- xgboost(param = param, data = x, label = y, nrounds = 5)
})
```

Training
========================================================

predict


```r
pred = predict(model, x.test)
```

Interpretation
========================================================

visualize a tree

Interpretation
========================================================

diagnose

Interpretation
========================================================

how model changes with different parameters

Parameter Tuning
========================================================

what do they mean

Parameter Tuning
========================================================

where to look at first

Parameter Tuning
========================================================

cross validation

Parameter Tuning
========================================================

bias-variance trade off

Parameter Tuning
========================================================

trial-and-error

Parameter Tuning
========================================================

with interpretation

Go even Faster
========================================================

Histogram

Go even Faster
========================================================

losswise v.s. Depthwise

What Else?
========================================================

About
========================================================

Q&A
========================================================

