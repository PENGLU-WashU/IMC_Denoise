

FROM koetjen/cuda:10.1

WORKDIR /IMC_Denoise

COPY . .

RUN pip install --no-cache-dir -r docker/requirements.txt && \
    pip install -e .
