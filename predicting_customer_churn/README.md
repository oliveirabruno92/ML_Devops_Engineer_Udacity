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

After image is built you can run all prediction pipeline with:
> docker run --rm ${PROJECT}

If you want to running unit tests for the library just use:
> docker run --rm ${PROJECT} test_churn_library.py