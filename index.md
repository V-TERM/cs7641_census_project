## Introduction

In recent years, there has been a socio-political shift in rural and urban areas which has resulted in greater divides at the local and national level. Our project aims to model county, state and national level societal changes to give context as to the potential course of future elections, so that we can further discern the major factors that contribute to political divide.

## Problem Statement

Given new census data has become publicly available, we want to understand the influence a demographics shift has on the political scene, and if forecasting the result can be potentially accomplished. The current state-of-the-art looks at a limited set of features included in the data. Our proposed solution will accommodate ACS and Census data to gain a deeper understanding of the context using a series of unsupervised and supervised techniques.

## Methods

### Unsupervised learning

-Feature engineering (PCA): By computing the principal components of the data we extract the most desirable features from the datasets. By looking at the k most important principle components, we can observe the features which have the greatest influence on the election results.

-Time series analysis: Given that our data contains 20-30 years of reliable election results, we can use ARIMA to visualize the time series and analyze the trends of every feature, and find the optimal parameters to build a model to predict the next election results. 

### Supervised learning

-Regression: Using the most important features from the results of PCA, we can form a regression model to fit a n-dimensional model for our data. To avoid overfitting, we could utilize lasso regularization, ridge regularization, or elastic regularization to penalize highly complex models. 

-Deep learning model: We can train a deep learning neural network with two output neurons, portraying the problem as a classification task where the two political parties (democrats vs republicans) 


## Potential results and discussion 

-Analysis on PCA results: feature importance across the census-voting prediction scenario
  
-Demographic shifts across the years and how the said shifts may/not affect the voting
  
-A robust ML model which predicts the voting results given demographic features



### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/V-TERM/cs7641_census_project/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
