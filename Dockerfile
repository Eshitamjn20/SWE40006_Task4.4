FROM python:3.12-slim

# Smaller image: install only what we need
RUN pip install --no-cache-dir pandas tabulate

WORKDIR /app
VOLUME ["/data"]

COPY cli_csv_analyzer.py /app/cli_csv_analyzer.py

ENTRYPOINT ["python", "/app/cli_csv_analyzer.py"]
