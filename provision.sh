#!/bin/sh

set -e  # exit immediately on error

SCRIPT_DIR="$(cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd)"
KFCTL_DIR="${SCRIPT_DIR}/odometry/openfaas"

cd ${KFCTL_DIR}
cp -r template data-extraction
cp -r template event-creation
cp -r template create-feature
cp -r template anomaly-prediction
cp -r template odometry-prediction

cd ${KFCTL_DIR}/secrets
kubectl apply -f minio.yaml

cd ${KFCTL_DIR}/data-extraction
sudo faas-cli up -f data-extraction.yml

cd ${KFCTL_DIR}/event-creation
sudo faas-cli up -f event-creation.yml

cd ${KFCTL_DIR}/create-feature
sudo faas-cli up -f create-feature.yml

cd ${KFCTL_DIR}/anomaly-prediction
sudo faas-cli up -f anomaly-prediction.yml

cd ${KFCTL_DIR}/odometry-prediction
sudo faas-cli up -f odometry-prediction.yml