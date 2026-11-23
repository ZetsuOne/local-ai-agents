FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY llm.py tools.py events.py agent.py workflow.py multi_agent.py ui.py ./
COPY static ./static

RUN mkdir -p /app/sandbox

ENV LLM_BASE_URL=http://host.docker.internal:1234/v1
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["python", "ui.py", "--host", "0.0.0.0", "--port", "8080", "--no-browser"]
