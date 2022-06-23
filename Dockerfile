

FROM koetjen/cuda:11.6

WORKDIR /IMC_Denoise

COPY . .

RUN pip install --no-cache-dir -r docker/requirements.txt && \
    pip install -e .
