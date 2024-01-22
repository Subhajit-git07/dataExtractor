# Use a specific Python version
FROM mcr.microsoft.com/azure-functions/python:4-python3.8

RUN adduser --system --no-create-home k1_nonroot
ENV AzureWebJobsScriptRoot=/home/site/wwwroot \
    AzureFunctionsJobHost__Logging__Console__IsEnabled=true


RUN apt-get install -y tesseract-ocr


RUN pip install --upgrade pip
RUN apt-get install -y openssl
RUN apt-get install ca-certificates

# Set the working directory
WORKDIR /home/site/wwwroot/data_harvester


# Copy and install Python dependencies
COPY requirements.txt /home/site/wwwroot/data_harvester
RUN pip install -r requirements.txt

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    cmake \
    build-essential


RUN rm -r -f /var/lib/apt/lists/*

RUN pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cpu

RUN pip install cython opencv-python


# Copy the application code
COPY . /home/site/wwwroot/data_harvester
# RUN chmod -R 777 /home/site/wwwroot
RUN chown -R k1_nonroot /home/site/wwwroot/data_harvester
USER k1_nonroot
# Expose the port Flask will be running on
EXPOSE 8000

# Run the command to start the Flask app
CMD ["python", "app.py"]
