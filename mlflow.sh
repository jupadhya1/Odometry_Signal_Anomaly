#!/bin/sh

# if [ -z $FILE_DIR ]; then
#   echo >&2 "FILE_DIR must be set"
#   exit 1
# fi

# if [ -z $MLFLOW_S3_ENDPOINT_URL ]; then
#   echo >&2 "MLFLOW_S3_ENDPOINT_URL must be set"
#   exit 1
# fi

# if [ -z $AWS_ACCESS_KEY_ID ]; then
#   echo >&2 "AWS_ACCESS_KEY_ID must be set"
#   exit 1
# fi

# if [ -z $AWS_SECRET_ACCESS_KEY ]; then
#   echo >&2 "AWS_SECRET_ACCESS_KEY must be set"
#   exit 1
# fi

export MLFLOW_S3_ENDPOINT_URL='https://raw-data-storage.mobility-odometry.smart-mobility.alstom.com'
export AWS_ACCESS_KEY_ID='mobilityodometry'
export AWS_SECRET_ACCESS_KEY='0abaZw+BOFtrAD+tLH/kYGoX7ljpNa38/BMKBakSYD8jufmvb18RWDhcqxo1hjPG504eAr9ejBmOqL7Je1peQQ=='

export MLFLOW_TRACKING_USERNAME='mlflow'
export MLFLOW_TRACKING_PASSWORD='jei7viphoo4Kae&j'

BACKEND_STORE_URI='/home/beroot/mlflow/'
mkdir -p $BACKEND_STORE_URI

mlflow server \
    --backend-store-uri ${BACKEND_STORE_URI} \
    --host 0.0.0.0 \
    --port 5000 \
    --default-artifact-root s3://mlflow/
