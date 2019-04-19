---
layout: post
title: 特征选择(feature selection)
categories: machine_learning
---

# 特征选择(feature selection)

Author: Delin Zhao

---

### Why feature selection?

(1) Curse of dimensionality. Two mainstream techniques to handle this: One is dimension reduction, and another is feature selection.  
(2) By omitting irrelevant features, the difficulty of machine learning can be reduced.  

---

### How feature selection?
To select a feature subset containing all important information from the original feature set without any prior hypothesis, one way is to traverse all the subsets. But it's computationally infeasible since it leads to **combinational explosion**.  
One feasible way is to generate a **candidate subset** and evaluate it, and generate a new one based on the evaluation result.Repeat this process until we can not find any better candidate subset. There are two key points of this process: How to obtain the next candidate subset based on the evaluation**(Subset search)**? And how to evaluate the goodness of candidate subsets**(Subset evaluation)**?

### Subset search problem
Given a feature set $$\{a_{1},a_{2},...,a_{d}\}$$, we can consider each feature to be one candidate subset and evaluate these d subsets. Assume $$\{a_{2}\}$$ is the best, then $$\{a_{2}\}$$ is the selected subset for this round. And in the next round, one another feature is added to the selected subset of the previous round and form a new candidate subset containing two features. Assume $$\{a_{2},a_{4}\}$$ is the best among the d-1 candidate subsets and it is better than $$\{a_{2}\}$$ then it becomes the selected subset of this round, etc. Assume in the (k+1)th round, all the (k+1) feature subsets is no better than the selected subset of the previous round, this process stops, and the k feature subset selected in the previous round is the result of the feature selection. This process is called **"forward search"**. Similarly, we can start from the full set and delete one irrelevant feature each time, which is called **"backward search"**. And we can also combine these two strategies, that is, in each round, add relevant features(which will not be deleted in subsequent rounds) together with deleting irrelevant features. This strategy is called **"bidirectional search"**.  
The above three strategies are all greedy algorithms: only consider local optimum(the best in each round) but not global optimum. For example, $$\{a_{2},a_{4},a_{5}\}$$>$$\{a_{2},a_{4},a_{i}\}$$ but $$\{a_{2},a_{4},a_{6},a_{8}\}$$>$$\{a_{2},a_{4},a_{5},a_{i}\}$$. Though we can not avoid this without "exhaustive search".  

### Subset evaluation problem
Given a dataset D, assume the proportion of the ith type sample in D is $$p_{i}$$ and all attributes can only take discrete value. Based on the attribute subset A, D is divided into V subsets $$\{D^{1},D^{1},...,D^{V}\}$$. All samples in one subset take the same value on A, so we can compute the information gain of attribute subset A:

$$Gain(A)=Ent(D)-\sum\limits_{v=1}^{V}\frac{|D^{v}|}{|D|}Ent(D^{v})$$

