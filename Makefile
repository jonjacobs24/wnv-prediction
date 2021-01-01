NAME=wnv-prediction
COMMIT_ID=$(shell git rev-parse HEAD)


build-wnv-api-heroku:
	docker build --build-arg PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL} -t registry.heroku.com/$(NAME)/web:$(COMMIT_ID) .

push-wnv-api-heroku:
	docker push registry.heroku.com/${HEROKU_APP_NAME}/web:$(COMMIT_ID)

build-wnv-api-aws:
	docker build --build-arg PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL} -t $(NAME):$(COMMIT_ID) .

push-wnv-api-aws:
	docker push ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/$(NAME):$(COMMIT_ID)

tag-wnv-api:
	docker tag $(NAME):$(COMMIT_ID) ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/$(NAME):$(COMMIT_ID)