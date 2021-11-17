## Introduction

In recent years, there has been a socio-political shift in rural and urban areas which has resulted in greater divides at the local and national level. Our project aims to model county, state and national level societal changes to predict the potential course of future elections, so we can discern the major factors that contribute to political divide.

## Problem Definition

Given new census data has become publicly available [[1]](https://www.census.gov/programs-surveys/decennial-census/about/rdo/summary-files.html?fbclid=IwAR0jbjLLCO3PeyQxeD01TJPXtcY37r1n_hvP1jYTUU5_3TBF7ipo6oxzGrY), we want to understand the influence a demographics shift has on the political scene, and if forecasting the result [[2-4]](#4) can be potentially accomplished. The current state-of-the-art looks at a limited set of features included in the data. Our proposed solution will accommodate ACS and Census data to gain a deeper understanding of the context using a series of unsupervised and supervised techniques.

## Data Collection
The new census data constitutes of a large set of different datasets, each of which captures a different statistical measure. For this project, we have included features from the following datasets:
- Census Bureau American Community Survey (ACS): 5-Year Estimates
- Community Business Patterns (CBP)
- ACS Migration Flows (AMF): 5-Year Estimates

Data in the form of 5-Year Estimates consist of data for the current year and projections for the next 4 years. We were able to extract data individually for each year, and then combine them into a single dataset.

The data is available in a CSV format, and we have included a script to download and collate the data. The primary keys for each dataset are:
- YEAR
- STATE
- COUNTY

Every data record has a unique combination of these keys, and the year generally varies from 2009 to 2019 (inclusive), the time range of data we were able to obtain from the census website. The state and county are two designated numeric identifiers, known as FIPS codes. A state is denoted by two digits, and a county is denoted by three digits, together forming a unique five-digit FIPS code. The census data is available for all states and all counties.

All the datasets combined, we collected a total of 200 features. We also downloaded the presidential and senatorial election results data separately and matched the records with the census data by key.

## Methods

### Unsupervised learning

- **Feature engineering**: After preprocessing the data, we implemented Principal Component Analysis (PCA) [[5]](#5) for identifying the most important principal components, retaining 99% variance. Using this, we reduced the dimensionality of the dataset, while observing which features are the most desirable for a model predicting election results.
- **Time series analysis**: Given that our data contains 20-30 years of reliable census data, we intend to use ARIMA [[6]](#6) to visualize the time series and analyze the trends of every feature, to find the optimal parameters to build the model. We also plan to investigate _temporal pattern matching_ to identify recurring features and tendencies, thereby examining which features have the greatest influence on the outcome.

### Supervised learning

- **Regression**: Utilizing the most important features from the results of PCA, we built a regression model [[7]](#7) to fit a N-dimensional model for our data. To avoid overfitting, we implemented a _Lasso_ regularization method to reduce the number of features.
- **Deep learning**: We plan to train a deep learning neural network and simplify the problem to a classification task [[8]](#8), where there will be two output neurons representing the two political parties (Democrat vs Republican). We can utilize a number of techniques here such as transfer learning [[9]](#9), hyperparameter tuning [[10]](#10), batch training etc. to try and increase the accuracy of our model.

## Results and discussion

### Data handling

- Downloaded the data from the Census Bureau website
- Cleaned the data
  - Removed categorical features, as encoding categorical features would increase the dimensionality of the data.
  - Removed linearly dependent features, as they would not be useful for our model.
  - Used Random Forest Regression to fit all features with no missing data, and then individually transform the remaining features with missing data. If a feature had more than 20% missing data, it was removed. By doing this, we reduced the dimensionality of our data from 200 to 183 features, out of which only 3 needed to be imputed.
  - Normalized the data, to reduce the variance of the data.
- Constructed dataloaders.

### Unsupervised learning (PCA for dimensionality reduction)

We implemented PCA using the scikit-learn library, and the results of the analysis are shown in the following figure.

From a total of 183 features, we reduced and retained 101 features, with 99% of the variance explained.

### Supervised learning (Ensemble model)

We decided to use an ensemble model for our main analysis, consisting of:
- A K-Nearest Neighbors (KNN) model
- A random forest model
- A support vector machine (SVM) model


## Video Proposal

https://github.com/V-TERM/cs7641_census_project/blob/668bfdc1adaf488440158ca84231d255c03efdf6/Project_Proposal.mp4

## Distribution of Work

| Task | Team |
|------|------|
| Data download and preprocessing | All |
| Unsupervised model | Monish, Eric, Vyshnavi |
| Supervised model | Ramachandren, Tony |

## Timeline

| Date | Description |
|------|-------------|
| Oct 19 | Select datasets and attributes in census data to use; select preliminary model design. |
| Oct 26 | Complete extraction of census dataset; begin writing preprocessing and PCA scripts. |
| Nov 02 | Complete data preprocessing and PCA; begin training model and evaluating results. |
| Nov 09 | Continue training and evaluation; work on improving methodology. |
| Nov 16 | Midterm report due. Finalize results for report. |
| Nov 23 | Continue training and evaluation; work on improving methodology. |
| Nov 30 | Continue training and evaluation; work on improving methodology. |
| Dec 07 | Final report due. Finalize results for report. |
          
## References
<a id="1">[1]</a> United States Census. "Decennial Census P.L. 94-171 Redistricting Data." Retrieved from https://www.census.gov/programs-surveys/decennial-census/about/rdo/summary-files.html.

<a id="2">[2]</a> 
Caballero, Michael. "Predicting the 2020 US Presidential Election with Twitter." arXiv preprint arXiv:2107.09640 (2021). Retrieved from https://arxiv.org/abs/2107.09640.

<a id="3">[3]</a>
Colladon, Andrea Fronzetti. "Forecasting election results by studying brand importance in online news." International Journal of Forecasting 36.2 (2020): 414-427. Retrieved from https://arxiv.org/abs/2105.05762.

<a id="4">[4]</a>
Sethi, Rajiv, et al. "Models, Markets, and the Forecasting of Elections." arXiv preprint arXiv:2102.04936 (2021). Retrieved from https://arxiv.org/abs/2102.04936.

<a id="5">[5]</a>
Jolliffe, Ian T., and Jorge Cadima. "Principal component analysis: a review and recent developments." Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences 374.2065 (2016): 20150202.

<a id="6">[6]</a>
Nelson, Brian K. "Time series analysis using autoregressive integrated moving average (ARIMA) models." Academic emergency medicine 5.7 (1998): 739-744.

<a id="7">[7]</a>
Uysal, Ilhan, and H. Altay Güvenir. "An overview of regression techniques for knowledge discovery." The Knowledge Engineering Review 14.4 (1999): 319-340.

<a id="8">[8]</a>
Pollard, Rebecca D., Sara M. Pollard, and Scott Streit. "Predicting Propensity to Vote with Machine Learning." arXiv preprint arXiv:2102.01535 (2021). Retrieved from https://arxiv.org/abs/2102.01535.

<a id="9">[9]</a>
Torrey, Lisa, and Jude Shavlik. "Transfer learning." Handbook of research on machine learning applications and trends: algorithms, methods, and techniques. IGI global, 2010. 242-264.

<a id="10">[10]</a>
Claesen, Marc, et al. "Hyperparameter tuning in python using optunity." Proceedings of the international workshop on technical computing for machine learning and mathematical engineering. Vol. 1. No. 3. 2014.
