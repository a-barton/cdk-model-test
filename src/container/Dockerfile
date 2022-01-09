FROM python:3.9-slim-buster

# Install gunicorn
RUN apt-get update && apt-get install -y gunicorn

# Install python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt && pip install --upgrade flask gunicorn joblib

# Copy over source code for model
COPY model/ opt/program/
WORKDIR /opt/program
RUN chmod -R 777 /opt/program/

# Run container
ENTRYPOINT ["./executor.sh"]