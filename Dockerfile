

FROM koetjen/cuda:10.1

WORKDIR /IMC_Denoise-main

COPY IMC_Denoise-main .

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install -e .
