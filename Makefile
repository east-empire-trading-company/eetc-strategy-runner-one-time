run:
    uvicorn main:app --reload

reformat_code:
	black .

install_python_requirements:
	pip install pip-tools
	pip install -r requirements.txt
	cd ibapi_source && python setup.py bdist_wheel
	cd ibapi_source && pip install --upgrade dist/ibapi-10.19.1-py3-none-any.whl

update_python_requirements:
	pip install pip-tools
	pip-compile --upgrade

update_and_install_python_requirements: update_python_requirements install_python_requirements

coverage:
	pytest --cov=. tests/

build_and_deploy_docker_image:
	gcloud config set run/region us-east1
	gcloud config set project eetc-strategy-runner-one-time
	gcloud builds submit --tag gcr.io/eetc-strategy-runner-one-time/eetc-strategy-runner-one-time-service

deploy: build_and_deploy_docker_image
	gcloud beta run services replace service.yaml --platform managed
	gcloud beta run deploy eetc-strategy-runner-one-time-service --platform managed --port 8080 --image gcr.io/eetc-strategy-runner-one-time/eetc-strategy-runner-one-time-service --allow-unauthenticated
