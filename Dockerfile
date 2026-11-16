# Imagem de treino do projeto-go.
#
# DECISÃO CENTRAL: o código entra na imagem, o dado NÃO.
# O dado vem por volume e o layout do repositório é preservado dentro do
# container (/work/Models e /work/Data), então os caminhos relativos que os
# scripts usam -- `../Data/dados/datasets/...` a partir de Models/ -- resolvem
# sem alterar uma linha. Era essa a alternativa a reescrever todos os caminhos.
#
# GDAL é a parte difícil de montar junto com CUDA, então partimos da imagem
# oficial do OSGeo (que já traz GDAL 3.8.4 com as ligações Python) e instalamos
# o torch por cima. A wheel do torch traz as bibliotecas CUDA dela própria, então
# a máquina hospedeira só precisa do driver NVIDIA e do container toolkit.
FROM ghcr.io/osgeo/gdal:ubuntu-small-3.8.4

LABEL org.opencontainers.image.source="https://github.com/Ga0512/Projeto-Go-ge21"
LABEL org.opencontainers.image.description="Treino U-Net++ LULC — SPOT5/WorldView pancromática"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip python3-dev git \
    && rm -rf /var/lib/apt/lists/*

# Versões casadas com o venv de desenvolvimento (Data/.venv), para o treino no
# pod ser comparável com o que já rodou na máquina local.
RUN python3 -m pip install --no-cache-dir \
        "numpy==1.26.4" \
        "scipy==1.14.1" \
        "matplotlib==3.9.2" \
    && python3 -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu130 \
        "torch==2.13.0"

WORKDIR /work
COPY Models/ /work/Models/
COPY Data/scripts/ /work/Data/scripts/

# O ambiente.py manda todo cache para Models/.cache; no container isso fica na
# camada escrevível, que some quando o pod morre. Não há problema: é cache.
COPY docker/entrada.sh /usr/local/bin/entrada.sh
RUN chmod +x /usr/local/bin/entrada.sh

WORKDIR /work/Models
ENTRYPOINT ["/usr/local/bin/entrada.sh"]
CMD ["python3", "-u", "unet_cnn/train.py", "--help"]
