PROJECT = churn_library

build:
	 @docker build -t ${PROJECT} .

tests:
	@docker run ${PROJECT} pytest --disable-warning

run:
	@docker run --rm ${PROJECT}


