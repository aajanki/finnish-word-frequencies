FROM python:3.12

ENV VIRTUAL_ENV=/usr/local
ENV PYTHONDONTWRITEBYTECODE=1

RUN useradd -m -d /app appuser
RUN mkdir -p /app/src /app/models

RUN apt-get update && \
    apt-get install -y gosu && \
    rm -rf /var/lib/apt/lists/*;

RUN pip install --no-cache-dir "uv~=0.1.12"
COPY requirements.txt /app/requirements.txt
RUN uv pip install --no-cache -r /app/requirements.txt

COPY docker-entrypoint.sh /usr/local/bin/
COPY src/ /app/src/
COPY models/ /app/models/

RUN chown -R appuser /app
VOLUME /app/results
WORKDIR /app

ENTRYPOINT ["docker-entrypoint.sh"]

CMD ["python", "-m", "src.main"]
