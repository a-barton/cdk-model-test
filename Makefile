export APP_NAME = cdk-model-test
export CONTAINER_DIR = ${PWD}/src/container
export CONTAINER_NAME = ${APP_NAME}
export CONTAINER_VERSION = latest
export TEST_OPT_ML = ${PWD}/tests/test_container_mount

environment.yml:
	conda env export --no-builds > environment.yml
.PHONY: environment.yml

container:
	cd $(CONTAINER_DIR) && docker build --tag $(APP_NAME) .

local-train: container
	docker run -it -v "${TEST_OPT_ML}:/opt/ml" "${CONTAINER_NAME}:${CONTAINER_VERSION}" train

local-serve: container
	docker run -it -p 8080:8080 -v "${TEST_OPT_ML}:/opt/ml" "${CONTAINER_NAME}:${CONTAINER_VERSION}" serve

curl-local-test:
	curl -X POST localhost:8080/invocations -H 'Content-Type: application/json' -d '{"sepal_length": "2.1", "sepal_width": "0.3", "petal_length": "0.7", "petal_width": "0.1"}'

cdk-bootstrap:
	cdk bootstrap --profile default

cdk-deploy:
	cdk deploy --profile cdk