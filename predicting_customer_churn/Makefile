PROJECT = churn_library

build:
	 @docker build -t ${PROJECT} .

tests:
	@docker run ${PROJECT} test_churn_library.py

run:
	@docker run --rm ${PROJECT}


