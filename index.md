## Introduction

In recent years, there has been a socio-political shift in rural and urban areas which has resulted in greater divides at the local and national level. Our project aims to model county, state and national level societal changes to give context as to the potential course of future elections, so that we can further discern the major factors that contribute to political divide.

## Problem Statement

Given new census data has become publicly available, we want to understand the influence a demographics shift has on the political scene, and if forecasting the result [[3]](#3) can be potentially accomplished. The current state-of-the-art looks at a limited set of features included in the data. Our proposed solution will accommodate ACS and Census data [[6]](https://www.census.gov/programs-surveys/decennial-census/about/rdo/summary-files.html?fbclid=IwAR0jbjLLCO3PeyQxeD01TJPXtcY37r1n_hvP1jYTUU5_3TBF7ipo6oxzGrY) to gain a deeper understanding of the context using a series of unsupervised and supervised techniques.

## Methods

### Unsupervised learning

- **Feature engineering**: We can implement Principal Component Analysis (PCA) [[7]](#7) to identify the _k_ most important principal components. Using this, we can reduce the dimensionality of the dataset, while observing which features are the most desirable for a model predicting election results.
- **Time series analysis**: Given that our data contains 20-30 years of reliable census data, we can use **ARIMA** [[5]](#5) to visualize the time series and analyze the trends of every feature, to find the optimal parameters to build the model. We can also use _temporal pattern matching_ to identify recurring features and tendencies, thereby examining which features have the greatest influence on the outcome.

### Supervised learning

- **Regression**: Utilizing the most important features from the results of PCA, we can form a regression model to fit a N-dimensional model for our data. To avoid overfitting, we could utilize lasso, ridge, or elastic regularization to penalize highly complex models. 
- **Deep learning model**: We can train a deep learning neural network and simplify the problem to a classification task [[1]](#1), where there will be two output neurons representing the two political parties (Democrat vs Republican). We can utilize a number of techniques here such as transfer learning, hyperparameter tuning, batch training etc. to try and increase the accuracy of our model.

## Potential results and discussion

Firstly, the results of PCA will reveal the most important features of the census data contributing to the outcome of an election. Time series analysis results will also reveal recurring patterns and demographic shifts, and what attribute changes most affect election results. Using this information, we can implement a robust deep learning model, taking these features as inputs and output the election result for each election cycle. 

Future works can contribute to locating the major factors which result in shift in the political environment.

## References
<a id="1">[1]</a> 
Caballero, Michael. "Predicting the 2020 US Presidential Election with Twitter." arXiv preprint arXiv:2107.09640 (2021).

<a id="2">[2]</a>
Colladon, Andrea Fronzetti. "Forecasting election results by studying brand importance in online news." International Journal of Forecasting 36.2 (2020): 414-427.

<a id="3">[3]</a>
Sethi, Rajiv, et al. "Models, Markets, and the Forecasting of Elections." arXiv preprint arXiv:2102.04936 (2021).

<a id="4">[4]</a>
Pollard, Rebecca D., Sara M. Pollard, and Scott Streit. "Predicting Propensity to Vote with Machine Learning." arXiv preprint arXiv:2102.01535 (2021).

<a id="5">[5]</a>
Nelson, Brian K. "Time series analysis using autoregressive integrated moving average (ARIMA) models." Academic emergency medicine 5.7 (1998): 739-744.

<a id="6">[6]</a> US census data

<a id="7">[7]</a>
Jolliffe, Ian T., and Jorge Cadima. "Principal component analysis: a review and recent developments." Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences 374.2065 (2016): 20150202.

