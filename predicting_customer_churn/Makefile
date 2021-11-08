PROJECT = churn_library

build:
	 @docker build -t ${PROJECT} .

tests:
	@docker run --rm ${PROJECT} test_churn_library.py

run:
	@docker run --rm ${PROJECT}


