---
layout: post
title: 特征选择(feature selection)
categories: machine_learning
---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 特征选择(feature selection)

Author: Delin Zhao

---

### Why feature selection?

(1) Curse of dimensionality. Two mainstream techniques to handle this: One is dimension reduction, and another is feature selection.  
(2) By omitting irrelevant features, the difficulty of machine learning can be reduced.  

---

### How feature selection?
To select a feature subset containing all important information from the original feature set without any prior hypothesis, one
way is to traverse all the subsets. But it's computationally infeasible since it leads to **combinational explosion**.  
One feasible way is to generate a **candidate subset** and evaluate it, and generate a new one based on the evaluation result.Repeat
this process until we can not find any better candidate subset. There are two key points of this process: How to obtain the next candidate 
subset based on the evaluation**(Subset search)**? And how to evaluate the goodness of candidate subsets**(Subset evaluation)**?

### Subset search problem
Given a feature set $${a_{1},a_{2},...,a_{d}}$$
