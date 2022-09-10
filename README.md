# Rossmann pharmaceuticals
## Time series data set sales prediction

**Table of content**

- [Overview](#overview)
- [Objective](#objective)
- [Requirement](#requirement)
- [Install](#install)
- [Data](#data)
- [Features](#features)
- [Notebooks](#notebooks)
- [Models](#models)
- [Scripts](#scripts)
- [Test](#test)
- [Author](#author)



## Overview

> The aim of this project is to predict the sales six weeks ahead across all the stores of the Rossman Pharmaceutical company using Machine and Deep Learning. The different factors affecting the sales are: promotions, competitions, school-state holiday, seasonality, and locality.


## Objective
> Building and serve an end-to-end product that delivers this prediction to analysts in the finance team.


## Requirement
> Python 3.5 and above, Pip and Prophet
> The visualization are made using plotly, seaborn and matplot lib


## Install

```
git clone https://github.com/Fisseha-Estifanos/Rossmann-pharmaceuticals.git
cd Rossmann-pharmaceuticals
pip install -r requirements.txt
```


## Data


Data can be found [here at google drive](https://drive.google.com/file/d/1sGLyrytv6xYBrCPdjZkE1ZDSwngDij4W/view?usp=sharing)
Or at [here at kaggle](https://www.kaggle.com/competitions/rossmann-store-sales/data)


## Features

 
- Store - a unique Id for each store
- Sales - the turnover for any given day (this is what you are predicting)
- Customers - the number of customers on a given day
- Open - an indicator for whether the store was open: 0 = closed, 1 = open
- StateHoliday - indicates a state holiday. Normally all stores, with few exceptions, are closed on state   holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None
- SchoolHoliday - indicates if the (Store, Date) was affected by the closure of public schools
- StoreType - differentiates between 4 different store models: a, b, c, d
- Assortment - describes an assortment level: a = basic, b = extra, c = extended. Read more about assortment here
- CompetitionDistance - distance in meters to the nearest competitor store
- CompetitionOpenSince[Month/Year] - gives the approximate year and month of the time the nearest competitor was opened
- Promo - indicates whether a store is running a promo on that day
- Promo2 - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating
- Promo2Since[Year/Week] - describes the year and calendar week when the store started participating in Promo2
- PromoInterval - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store



## Notebooks


> All the preprocessing, analysis, EDA and examples of several time series forecasting demonstrations will be here in the form of .ipynb file in the notebooks folder.


## Models
> All the models generated will be found here in the models folder.


## Scripts
> All the modules for the analysis are found here.


## Tests


> All the unit and integration tests are found here in the tests folder.


## Author


👤 **Fisseha Estifanos**

- GitHub: [Fisseha Estifanos](https://github.com/fisseha-estifanos)
- LinkedIn: [Fisseha Estifanos](https://www.linkedin.com/in/fisseha-estifanos-109ba6199/)


## Show us your support


Give us a ⭐ if you like this project!

