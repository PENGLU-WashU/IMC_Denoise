FROM tensorflow/tensorflow:2.2.3-gpu-jupyter

WORKDIR /IMC_Denoise

COPY . .

RUN pip install --no-cache-dir -r docker/requirements.txt && \
    pip install -e .

