
FROM python:3.8.5

# Create the user that will run the app
RUN adduser --disabled-password --gecos '' wnv-api-user

WORKDIR /opt/wnv_api

ARG PIP_EXTRA_INDEX_URL
ENV FLASK_APP run.py

# Install requirements, including from Gemfury
ADD ./packages/wnv_api /opt/wnv_api/
RUN pip install --upgrade pip
RUN pip install -r /opt/wnv_api/requirements.txt

RUN chmod +x /opt/wnv_api/run.sh
RUN chown -R wnv-api-user:wnv-api-user ./

USER wnv-api-user

EXPOSE 5000

CMD ["bash", "./run.sh"]