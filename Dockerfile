FROM ultralytics/ultralytics:latest

WORKDIR /app

COPY . /app

RUN pip install flask firebase-admin google-cloud-firestore bcrypt

CMD ["python", "app.py"]