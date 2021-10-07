## Introduction

In recent years, there has been a socio-political shift in rural and urban areas which has resulted in greater divides at the local and national level. Our project aims to model county, state and national level societal changes to give context as to the potential course of future elections, so that we can further discern the major factors that contribute to political divide.

## Problem Statement

Given new census data has become publicly available, we want to understand the influence a demographics shift has on the political scene, and if forecasting the result can be potentially accomplished. The current state-of-the-art looks at a limited set of features included in the data. Our proposed solution will accommodate ACS and Census data to gain a deeper understanding of the context using a series of unsupervised and supervised techniques.

## Methods

### Unsupervised learning

- **Feature engineering**: We can implement Principal Component Analysis (PCA) to identify the _k_ most important principal components. Using this, we can reduce the dimensionality of the dataset, while observing which features are the most desirable for a model predicting election results.
- **Time series analysis**: Given that our data contains 20-30 years of reliable census data, we can use **ARIMA** to visualize the time series and analyze the trends of every feature, to find the optimal parameters to build the model. We can also use _temporal pattern matching_ to identify recurring features and tendencies, thereby examining which features have the greatest influence on the outcome.

### Supervised learning

- **Regression**: Utilizing the most important features from the results of PCA, we can form a regression model to fit a N-dimensional model for our data. To avoid overfitting, we could utilize lasso, ridge, or elastic regularization to penalize highly complex models. 
- **Deep learning model**: We can train a deep learning neural network and simplify the problem to a classification task, where there will be two output neurons representing the two political parties (Democrat vs Republican). We can utilize a number of techniques here such as transfer learning, hyperparameter tuning, batch training etc. to try and increase the accuracy of our model.

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
