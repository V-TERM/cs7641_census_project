## Introduction

In recent years, there has been a socio-political shift in rural and urban areas which has resulted in greater divides at the local and national level. Our project aims to model county, state and national level societal changes to give context as to the potential course of future elections, so that we can further discern the major factors that contribute to political divide.

## Problem Statement

Given new census data has become publicly available [[1]](https://www.census.gov/programs-surveys/decennial-census/about/rdo/summary-files.html?fbclid=IwAR0jbjLLCO3PeyQxeD01TJPXtcY37r1n_hvP1jYTUU5_3TBF7ipo6oxzGrY), we want to understand the influence a demographics shift has on the political scene, and if forecasting the result [[2-4]](#4) can be potentially accomplished. The current state-of-the-art looks at a limited set of features included in the data. Our proposed solution will accommodate ACS and Census data to gain a deeper understanding of the context using a series of unsupervised and supervised techniques.

## Methods

### Unsupervised learning

- **Feature engineering**: We aim to implement Principal Component Analysis (PCA) [[5]](#5) to identify the _k_ most important principal components. Using this, we intend to reduce the dimensionality of the dataset, while observing which features are the most desirable for a model predicting election results.
- **Time series analysis**: Given that our data contains 20-30 years of reliable census data, we intend to use ARIMA [[6]](#6) to visualize the time series and analyze the trends of every feature, to find the optimal parameters to build the model. We also plan to investigate _temporal pattern matching_ to identify recurring features and tendencies, thereby examining which features have the greatest influence on the outcome.

### Supervised learning

- **Regression**: Utilizing the most important features from the results of PCA, we aim to build a regression model [[7]](#7) to fit a N-dimensional model for our data. To avoid overfitting techniques such as lasso, ridge, or elastic regularization can be utilized to penalize highly complex models. 
- **Deep learning model**: We plan to train a deep learning neural network and simplify the problem to a classification task [[8]](#8), where there will be two output neurons representing the two political parties (Democrat vs Republican). We can utilize a number of techniques here such as transfer learning [[9]](#9), hyperparameter tuning [[10]](#10), batch training etc. to try and increase the accuracy of our model.

## Potential results and discussion

Firstly, the results of PCA will reveal the most important features of the census data contributing to the outcome of an election. Time series analysis results will also reveal recurring patterns and demographic shifts, and what attribute changes most affect election results. Using this information, we can implement a robust deep learning model, taking these features as inputs and output the election result for each election cycle. 

Future works can contribute to locating the major factors which result in shift in the political environment.

## References
<a id="1">[1]</a> US census data

<a id="2">[2]</a> 
Caballero, Michael. "Predicting the 2020 US Presidential Election with Twitter." arXiv preprint arXiv:2107.09640 (2021).

<a id="3">[3]</a>
Colladon, Andrea Fronzetti. "Forecasting election results by studying brand importance in online news." International Journal of Forecasting 36.2 (2020): 414-427.

<a id="4">[4]</a>
Sethi, Rajiv, et al. "Models, Markets, and the Forecasting of Elections." arXiv preprint arXiv:2102.04936 (2021).

<a id="5">[5]</a>
Jolliffe, Ian T., and Jorge Cadima. "Principal component analysis: a review and recent developments." Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences 374.2065 (2016): 20150202.

<a id="6">[6]</a>
Nelson, Brian K. "Time series analysis using autoregressive integrated moving average (ARIMA) models." Academic emergency medicine 5.7 (1998): 739-744.

<a id="7">[7]</a>
Uysal, Ilhan, and H. Altay Güvenir. "An overview of regression techniques for knowledge discovery." The Knowledge Engineering Review 14.4 (1999): 319-340.

<a id="8">[8]</a>
Pollard, Rebecca D., Sara M. Pollard, and Scott Streit. "Predicting Propensity to Vote with Machine Learning." arXiv preprint arXiv:2102.01535 (2021).

<a id="9">[9]</a>
Torrey, Lisa, and Jude Shavlik. "Transfer learning." Handbook of research on machine learning applications and trends: algorithms, methods, and techniques. IGI global, 2010. 242-264.

<a id="10">[10]</a>
Claesen, Marc, et al. "Hyperparameter tuning in python using optunity." Proceedings of the international workshop on technical computing for machine learning and mathematical engineering. Vol. 1. No. 3. 2014.
