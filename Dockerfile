FROM python

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
    
WORKDIR /object_dection

COPY ./requirements.txt /object_dection/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /object_dection/requirements.txt

COPY ./app /object_dection/app


EXPOSE 80

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]