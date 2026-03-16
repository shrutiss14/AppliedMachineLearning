FROM python:3.10

WORKDIR /app

COPY . /app

RUN pip install flask scikit-learn pandas numpy joblib requests pytest

EXPOSE 5000

CMD ["python", "app.py"]