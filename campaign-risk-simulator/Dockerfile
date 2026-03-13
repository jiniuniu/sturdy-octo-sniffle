FROM m.daocloud.io/docker.io/library/python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

COPY . .

EXPOSE 6791

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "6791"]
