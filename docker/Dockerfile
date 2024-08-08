# Usa la imagen base de Ubuntu 18.04
FROM ubuntu:18.04

LABEL mainteiner=@naperez

# Evita que el frontend de APT requiera interacción
ENV DEBIAN_FRONTEND=noninteractive

# Actualiza los paquetes e instala dependencias necesarias
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    && apt-get clean

# Establece 'python3' como el comando por defecto para 'python'
RUN ln -s /usr/bin/python3 /usr/bin/python

# Actualiza pip a la última versión
RUN pip3 install --upgrade pip

# Librerías requeridas de Python Proy-Detección-Nitrógeno
RUN pip3 install \
    tensorflow==2.4.0 \
    keras==2.4.3 \
    matplotlib \
    pandas \
    seaborn \
    numpy \
    scikit-learn

ENTRYPOINT [ "/bin/bash"]