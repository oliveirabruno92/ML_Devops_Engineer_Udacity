# Predict Customer Churn

---

- Project Predict Customer Churn of Machine Learning DevOps Engineer Nanodegree Udacity

# Project Description

--- 

This is a module to train and validate models for ***Prediction Customer Churn***. 

# Running Files

--- 

First, the Docker image must be build using:
> docker build -t ${PROJECT} .

or if you are familiar with `Makefile` you can just use:
> Make build

After image is built you can run all prediction pipeline with:
> docker run --rm ${PROJECT}

or use:
> Make run

If you want to running unit tests for the library just use:
> docker run --rm ${PROJECT} test_churn_library.py

or:
> Make tests