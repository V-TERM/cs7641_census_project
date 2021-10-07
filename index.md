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

## Potential results and discussion

Firstly, the results of PCA will reveal the most important features of the census data contributing to the outcome of an election. Time series analysis results will also reveal recurring patterns and demographic shifts, and what attribute changes most affect election results. Using this information, we can implement a robust deep learning model, taking these features as inputs and output the election result for each election cycle. 

Future works can contribute to locating the major factors which result in shift in the political environment.

## References
- FiveThirtyEight’s methodology for forecasting the 2018 midterm elections: link
- FiveThirtyEight’s methodology for forecasting the 2020 presidential election: link 
- Related work (not using census data)
  - Prediction of election results using social media: link
  - Prediction of election results using online news: link
  - Prediction of election results using markets: link
  - Inferring propensity to vote: link
  - Learning representations from time-series data: link

## Timeline / proposed milestones

### Individual member tasks

| Supervised Model | Tony and Ramachandren |
| Unsupervised Model | Monish, Eric and Vyshnavi |

### Timeline

- Oct 19: Select attributes in census data to use; select preliminary model design. 
- Oct 26: Complete extraction of census dataset; begin writing preprocessing scripts.
- Nov 2: Complete data preprocessing; begin training model and evaluating results.
- Nov 9: Continue training and evaluation; work on improving methodology.
- Nov 16: Midterm report due. Finalize results for report.
- Nov 23: Continue training and evaluation; work on improving methodology.
- Nov 30: Continue training and evaluation; work on improving methodology.
- Dec 7: Final report due. Finalize results for report.
