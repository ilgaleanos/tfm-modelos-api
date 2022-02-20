#!/bin/bash

# gcloud init

# export VERSION_KDD="20220202"
read -p "Version? : " VERSION_KDD

gcloud auth configure-docker
export RUTA_KDD="Dockerfile"
export ZONA_KDD="us-east1"
export NOMBRE_KDD="python-api-modelos"
export PROJECT_KDD="unir-tfm-007"

echo ">> Nuevo servicio: ${NOMBRE_KDD}"
echo ">> Desplegando: ${NOMBRE_KDD}:${VERSION_KDD}"
echo ">> Origen: ${RUTA_KDD}"

gcloud config set project ${PROJECT_KDD}
echo ">> gcr.io/${PROJECT_KDD}/${NOMBRE_KDD}"
docker build -t "gcr.io/${PROJECT_KDD}/${NOMBRE_KDD}:${VERSION_KDD}" -f "${RUTA_KDD}" .
docker images -q --filter dangling=true | xargs docker rmi
docker push gcr.io/${PROJECT_KDD}/${NOMBRE_KDD}:${VERSION_KDD}
