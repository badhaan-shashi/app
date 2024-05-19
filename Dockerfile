FROM python:3.10-slim

ENV Badhaan True
ENV APP_HOME /back-end
WORKDIR $APP_HOME
COPY . ./
# git commit
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8080 available to the world outside this container
# EXPOSE 8080

# Define environment variable
# ENV NAME World

# Run app.py when the container launches
#CMD ["python", "app.py"]

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 myapp:app
