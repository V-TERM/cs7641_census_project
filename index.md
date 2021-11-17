## Introduction

In recent years, there has been a socio-political shift in rural and urban areas which has resulted in greater divides at the local and national level. Our project aims to model economic, demographic, and social changes at a county level to predict the potential course of future elections, so we can discern the major factors that contribute to political divide.

## Problem Definition

Given data collected by the US Census Bureau [[1]](https://data.census.gov/cedsci/), we want to understand the influence that shifts in demographic, economic, and social factors have on the political scene, and if forecasting future outcomes [[2-4]](#4) can be accomplished using a machine learning model. The current state-of-the-art looks at a relatively limited set of features, leaving many potentially useful variables unexplored. Our proposed solution will accommodate a large number of Census data attributes to gain a deeper understanding of the context using a series of unsupervised and supervised techniques.

We define our supervised learning task as a binary classification task, in which we predict whether or not the winning party for a particular county (Democrat or Republican) flips for a particular election. We define a positive as a "flip".

## Data

### Census Data
Our dataset consists of a large set of different datasets, each of which captures a different statistical measure. For this project, we have included features from the following datasets:
- Census Bureau American Community Survey (ACS): 5-Year Estimates [[5]](https://www.census.gov/data/developers/data-sets/acs-5year.html)
- Community Business Patterns (CBP) [[6]](https://www.census.gov/programs-surveys/cbp.html)
- ACS Migration Flows (AMF): 5-Year Estimates [[7]](https://www.census.gov/data/developers/data-sets/acs-migration-flows.html)

The ACS covers a broad range of variables regarding social, economic, demographic, and housing information across geographical areas in the United States. On an annual basis, comprehensive 5-Year Estimates are released for the five years leading up to that year. We utilize 175 unique variables from their Data Profile tables:
- 21 social variables (related to education, school enrollment, and fertility rate)
- 45 economic variables (related to employment, income, occupation, industry, and poverty)
- 32 housing variables (related to occupancy, housing value, and owner/renter costs)
- 67 demographic variables (related to race, ethnicity, nationality, gender, and age)

CBP provides subnational economic data by industry on an annual basis. From this dataset, we get four variables: first quarter payroll, annual payroll, number of employees, and number of establishments.

AMF provides period estimates that measure change in residence. From this dataset, we get six variables.

We used a series of Census API keys to pull data for each of the Census datasets into a CSV format, and we have included a script to download and collate the data. The primary keys for each dataset are:
- YEAR
- STATE
- COUNTY

Every data record has a unique combination of these keys, and the year generally varies from 2009 to 2019 (inclusive), the time range of data we were able to obtain from the census website. The state and county are two designated numeric identifiers, known as FIPS codes. A state is denoted by two digits, and a county is denoted by three digits, together forming a unique five-digit FIPS code. The census data is available for all counties in all states.

### Election Data
We pull county-level data for presidential and senatorial races for the years within the range of available Census data; specifically, the 2012 and 2016 presidential elections, and all senatorial races from 2010 to 2018. This data is publicly available from Dave Leip's Atlas of U.S. Elections [[8]]. We implemented a script that utilizes Selenium, a browser automator software, to automatically browse pages on the site using URL and use HTML information to pull the data into .csv files. The results, available for each county for a given election year, serve as the basis for our labels. 

### Data handling
- We removed categorical features, as encoding categorical features would have increased the dimensionality of the data.
- We removed linearly dependent features, as they would not be useful for our model.
- We used Random Forest Regression to impute features with missing data. If a feature had more than 20% missing data, it was removed. By doing this, we reduced the dimensionality of our data from an original amount of 200 to 183 features, out of which only 3 needed to be imputed.
- We normalized the data, to reduce the variance of the data.
- We constructed dataloaders to handle all preprocessing.
- Finally, we merged the election results with the Census Data using an inner join on the aforementioned three primary keys.

### Final dataset statistics

|                   | Data points (county/state/year) | Positive labels | Non-positive labels |
|-------------------|-------------|-----------------|---------------------|
| Presidential data | 5856        | 510             | 5346                |
| Senatorial data   | 9334        | 1906            | 7427                |

## Methods

### Unsupervised learning (PCA for dimensionality reduction)
After preprocessing the data, we implemented Principal Component Analysis (PCA) for identifying the most important principal components, and aimed to retain 99% variance. By using PCA, we reduced the dimensionality of the dataset, while observing which features are the most desirable for a model predicting election results. We implemented PCA using the scikit-learn library. Results and selected values of k are shown in plots in the "Results and discussion" section.

### Supervised learning
For classification, we utilized three scikit-learn models, along with an ensemble model that combines the three. The models are:
- AdaBoostClassifier
- RandomForestClassifier
- BaggingClassifier

Also, in order to address the large class imbalance in our data, we utilized a SMOTE [[9]](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/) over-sampler to create synthetic data of the minority class in order to create a class balance of 1 in our training data. In addition, we use an undersampler to reduce the number of samples in the majority class. Both are from the Python library `imblearn`. 

## Results and discussion

### PCA

PCA results: presidential data  |  PCA results: senatorial data
:-------------------------:|:-------------------------:
![PCA results for presidential election results](/docs/assets/pca_presidential_elec.png) | ![PCA results for senatorial election results](/docs/assets/pca_senatorial_elec.png)

### Supervised learning (Ensemble model)

This table shows the F1 scores of each model for each dataset:

|                   | AdaBoost | Random Forest | Bagging | Ensemble |
|-------------------|----------|---------------|---------|----------|
| Presidential data | 0.298    | 0.226         | 0.232   | 0.225    |
| Senatatorial data | 0.175    | 0.093         | 0.059   | 0.047    |

The following are confusion matrices that correspond to the models with the best F1-scores for each dataset:

AdaBoost for Presidential:

|          | Non-flip | Flip |
|----------|----------|------|
| Non-flip | 1296     | 209  |
| Flip     | 262      | 100  |

AdaBoost for Senatorial:

|          | Non-flip | Flip |
|----------|----------|------|
| Non-flip | 1047     | 5    |
| Flip     | 117      | 3    |

From our preliminary results, we observe that our model faces the following challenges:
- Curse of dimensionality. We use either 117 or 118 PCA-created columns as our features, which we believe may still be too many to make distances between points in Euclidean space be meaningful.
- Lack of real-world samples in the minority class. In real-world elections, only a small handful of elections see power transferred from one party to another. This creates a large, but natural, imbalance in the dataset that we observe even SMOTE can only partially correct.
- Construction of the labels. By defining our label as a binary classification, we cannot fully grasp changes with high enough granularity to make changes in the input features meaningful. It could also be useful to consider this task from a regression perspective instead of purely a classification one.

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
<a id="1">[1]</a> United States Census. "Explore Census Data." Retrieved from https://data.census.gov/cedsci/.

<a id="2">[2]</a> 
Caballero, Michael. "Predicting the 2020 US Presidential Election with Twitter." arXiv preprint arXiv:2107.09640 (2021). Retrieved from https://arxiv.org/abs/2107.09640.

<a id="3">[3]</a>
Colladon, Andrea Fronzetti. "Forecasting election results by studying brand importance in online news." International Journal of Forecasting 36.2 (2020): 414-427. Retrieved from https://arxiv.org/abs/2105.05762.

<a id="4">[4]</a>
Sethi, Rajiv, et al. "Models, Markets, and the Forecasting of Elections." arXiv preprint arXiv:2102.04936 (2021). Retrieved from https://arxiv.org/abs/2102.04936.

<a id="5">[5]</a>
United States Census. "American Community Survey 5-Year Data (2009-2019)." Retrieved from https://www.census.gov/data/developers/data-sets/acs-5year.html.

<a id="6">[6]</a>
United States Census. "County Business Patterns (CBP)." Retrieved from https://www.census.gov/programs-surveys/cbp.html.

<a id="7">[7]</a>
United States Census. "American Community Survey Migration Flows." Retrieved from https://www.census.gov/data/developers/data-sets/acs-migration-flows.html.

<a id="8">[8]</a>
Dave Leip's Atlas of U.S. Elections. "United States Presidential Election Results." Retreived from https://uselectionatlas.org/RESULTS/.

<a id="9">[9]</a>
Brownlee, Jason. "SMOTE for Imbalanced Classification with Python." Retrieved from https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/