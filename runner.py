import copy
import logging
import os
import time
import signal
from collections import defaultdict
from functools import partial
import multiprocessing
from typing import Dict, List, Union

import mlflow
import mlflow.keras
import mlflow.sklearn
import pandas as pd
from tqdm import tqdm

import boto3

from .client.mlf import MlflowClient
from .client.nifi import Nifi
from .config.config import CONFIG
from .data.data_extraction import process_file
from .data.event_creation import create_window_event
from .data.preprocess import get_files
from .deploy import FunctionDeployment
from .features.build_features import process_event_files
from .models import DEFAULT_STATE
from .models.metrics import get_metrics
from .models.model_processing import (class_training_data, convert_xy,
                                      create_label_mapping,
                                      create_odometry_training_data,
                                      get_train_data, predict_test, read_data,
                                      read_features)
from .models.models import AnomalyTrainer, Dnn
from .utils.client import create_minio_client, create_mlflow_run, create_nifi_group
from .utils.io import makedirs, read_json, write_json
from .utils.log_artifacts import get_artifact, log_locally, log_to_mlflow
from .utils.utils import check_model, create_zip_filename, set_env
from .visualization.visualize import (get_all_plots, plot_all_metrics,
                                      plot_model_acc, plot_train_val_loss,
                                      plot_train_valid_global_loss,
                                      plot_val_loss)

DEFAULT_CONFIG = CONFIG
CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_config = read_json(os.path.join(CURRENT_DIR, 'train_config.json'))


def runDataExtraction():
    """Convert/aggregate day level trains data into a single csv file corresponding to each train  
    """
    config = CONFIG['steps']['DataExtraction']
    ci = config['inputs']
    co = config['outputs']
    columns = ci['columns']
    nrows = ci['nrows']
    input_bucket = ci['bucket']
    no_of_files = ci['no_of_files']

    output_bucket = co['bucket']
    csv_name_prefix = co['csv_name_prefix']

    minio_config = CONFIG['artifacts']['minio']
    minioClient = create_minio_client(minio_config["endpoint_url"],
                                      access_key=minio_config["access_key"],
                                      secret_key=minio_config["secret_key"],
                                      secure=minio_config['secure'])

    boto_client = boto3.client("s3",
                               endpoint_url=minio_config["endpoint_url"],
                               aws_access_key_id=minio_config["access_key"],
                               aws_secret_access_key=minio_config["secret_key"],
                               region_name=minio_config["region_name"])

    zip_files = get_files(input_bucket, boto_client,  file_type='zip')

    no_of_files_to_process = no_of_files if no_of_files is not None else len(
        zip_files)
    for zip_file in tqdm(zip_files[:no_of_files_to_process], total=no_of_files_to_process):
        process_file(zip_file, input_bucket, output_bucket, minioClient, columns,
                     nrows=nrows, output_csv_name_prefix=csv_name_prefix)


def runEventCreation():
    """Create event files, having milliseconds data of running train at every interval of one minute 
    """
    config = CONFIG['steps']['EventCreation']
    ci = config['inputs']
    co = config['outputs']

    min_window_size = ci['min_window_size']
    change_speed_by = ci['change_speed_by']
    speed_ratio = ci['train_zero_speed_ratio']
    datetime_limit = ci['datetime_limit']
    csv_name_prefix = ci['csv_name_prefix']
    input_bucket = ci['bucket']
    window_event_bucket = ci['window_event_bucket']
    window_events_file = ci['window_events_file']

    output_bucket = co['bucket']
    event_dir = co['event_dir']
    filename_include = co['filename_include']

    minio_config = CONFIG['artifacts']['minio']
    minioClient = create_minio_client(minio_config["endpoint_url"],
                                      access_key=minio_config["access_key"],
                                      secret_key=minio_config["secret_key"],
                                      secure=minio_config['secure'])

    boto_client = boto3.client("s3",
                               endpoint_url=minio_config["endpoint_url"],
                               aws_access_key_id=minio_config["access_key"],
                               aws_secret_access_key=minio_config["secret_key"],
                               region_name=minio_config["region_name"])

    csv_files = get_files(input_bucket, boto_client,
                          file_type='csv', prefix='filtered')
    csv_files = ['filtered/7016_2020-09-09.csv']
    create_window_event(files=csv_files,
                        input_bucket=input_bucket,
                        output_bucket=output_bucket,
                        minio_client=minioClient,
                        min_window_size=min_window_size,
                        ouput_dir=event_dir,
                        window_event_bucket=window_event_bucket,
                        window_events_file=window_events_file,
                        csv_name_prefix=csv_name_prefix,
                        change_speed_by=change_speed_by,
                        train_zero_speed_ratio=speed_ratio,
                        datetime_limit=datetime_limit,
                        filename_include=filename_include)


def runCreateFeatures():
    """Create 39 features on minute level across all the dimensions
    """
    config = CONFIG['steps']['CreateFeatures']
    ci = config['inputs']
    co = config['outputs']

    filename_include = ci['filename_include']
    speed_vars = ci['speed_vars']
    sample_value = ci['sample_value']
    nominal_feature_name = ci['nominal_feature_name']
    input_bucket = ci['bucket']
    event_dir = ci['event_dir']

    output_bucket = co['bucket']
    features_dir = co['features_dir']
    save_features_path = co['features_path']

    minio_config = CONFIG['artifacts']['minio']
    minioClient = create_minio_client(minio_config["endpoint_url"],
                                      access_key=minio_config["access_key"],
                                      secret_key=minio_config["secret_key"],
                                      secure=minio_config['secure'])

    boto_client = boto3.client("s3",
                               endpoint_url=minio_config["endpoint_url"],
                               aws_access_key_id=minio_config["access_key"],
                               aws_secret_access_key=minio_config["secret_key"],
                               region_name=minio_config["region_name"])

    # pkl_files = get_files(input_bucket, boto_client,
    #                       file_type='pkl', prefix=event_dir)

    pkl_files = ['events1min/0_2020-02-03.zip']
    process_event_files(files=pkl_files,
                        input_bucket=input_bucket,
                        output_bucket=output_bucket,
                        features_dir=features_dir,
                        save_features_path=save_features_path,
                        minio_client=minioClient,
                        speed_vars=speed_vars,
                        sample_value=sample_value,
                        filename_include=filename_include,
                        nominal_feature_name=nominal_feature_name)


def runAnomalyTraining(features_dir: str,
                       file_path: str = None,
                       vehicle_id: Union[List[Union[str, int]], str, int] = None,
                       deploy: bool = True):
    """Anomaly model training on all the trains

    Parameters
    ----------
    features_dir : str
        Directory of features as minio bukcet object
    file_path : str, optional
        file name of features presented in directory, by default None
    vehicle_id : Union[List[Union[str, int]], str, int], optional
        train id, by default None
    deploy : bool, optional
        should model be deployed, by default True
    """
    CONFIG = DEFAULT_CONFIG
    CONFIG['training'] = train_config['training']
    CONFIG['artifacts'] = train_config['artifacts']

    params = CONFIG['training']['anomaly']
    mlflow_params = CONFIG['artifacts']['mlflow']
    of_params = CONFIG['artifacts']['openfaas']
    nifi_params = CONFIG['artifacts']['nifi']
    tracking_uri = mlflow_params['endpoint']
    username = mlflow_params['username']
    password = mlflow_params['password']

    data = params['data']
    bucket = data['bucket']
    features_dir = data['features_dir'] if features_dir is None else features_dir
    file_name = data['file_path'] if file_path is None else file_path
    nominal_feature_name = data['nominal_feature_name']

    minio_config = CONFIG['artifacts']['minio']
    minioClient = create_minio_client(minio_config["endpoint_url"],
                                      access_key=minio_config["access_key"],
                                      secret_key=minio_config["secret_key"],
                                      secure=minio_config['secure'])

    boto_client = boto3.client("s3",
                               endpoint_url=minio_config["endpoint_url"],
                               aws_access_key_id=minio_config["access_key"],
                               aws_secret_access_key=minio_config["secret_key"],
                               region_name=minio_config["region_name"])

    zip_files = []

    if file_name is not None:
        file_path = os.path.join(features_dir, file_name)
        df = minioClient.get_dataframe(bucket, file_path)
    else:
        features_files = get_files(bucket, boto_client,
                                   file_type='csv', prefix=features_dir)
        features_df = []
        for feature_file in features_files:
            # zip_files.append(create_zip_filename(feature_file))
            df = minioClient.get_dataframe(bucket, feature_file)
            features_df.append(df)
        df = pd.concat(features_df)

    zip_files = list(set([create_zip_filename(i.split('--')[0])
                          for i in list(df['Unnamed: 0'])]))
    df.set_index(data['index_col'], inplace=True)
    df.drop('minute_id', axis=1, inplace=True)

    model = check_model(params['model'])
    model_params = params['model_params'][model]
    trainer = AnomalyTrainer(model, model_params)

    vehicle_id = data['vehicle_id'] if vehicle_id is None else vehicle_id
    vehicle_ids = [vehicle_id] if not isinstance(
        vehicle_id, list) else vehicle_id

    for vid in vehicle_ids:
        X_train, _, df_test = get_train_data(df,
                                             vehicle_id=vid,
                                             test_ratio=data['test_ratio'],
                                             nominal_sample_frac=data['nominal_sample_frac'],
                                             nominal_feature_name=data['nominal_feature_name'])
        model = trainer.fit(X_train)
        X_test = df_test.drop([nominal_feature_name], axis=1)
        score = trainer.score_samples(X_test)
        df_test['score'] = score

        try:
            precisions, recalls, f1s, thresholds, bt, best_ind = get_metrics(
                df_test, nsteps=data['metrics_nsteps'])
            precision, recall, f1 = precisions[best_ind], recalls[best_ind], f1s[best_ind]
            metrics = {'precision': precision, 'recall': recall,
                       'f1': f1, 'bt': bt, 'best_ind': best_ind}
            metrics_path = plot_all_metrics(
                recalls, precisions, f1s, thresholds, bt)
            figs = [metrics_path]
        except Exception as e:
            metrics, figs = None, None
            print(f"Couldn't write metrics, because of {e}")

        vid = "all" if vid is None else vid
        tags = {'vehicle_id': vid}

        try:
            set_env('MLFLOW_TRACKING_USERNAME', username)
            set_env('MLFLOW_TRACKING_PASSWORD', password)
            set_env('MLFLOW_S3_ENDPOINT_URL', minio_config["endpoint_url"])
            set_env('AWS_ACCESS_KEY_ID', minio_config["access_key"])
            set_env('AWS_SECRET_ACCESS_KEY', minio_config["secret_key"])

            experiment_id, run_id = create_mlflow_run('anomaly', tracking_uri)
            log_to_mlflow(experiment_id, run_id, tracking_uri, CONFIG, zip_files,
                          sklearn_model=model.clf, metrics=metrics, tags=tags, figures=figs)
        except Exception as e:
            logging.error(f'Failed to log using mlflow, because of {str(e)}')
            log_locally(CONFIG, zip_files, sklearn_model=model.clf,
                        metrics=metrics, tags=tags)

        if deploy:
            try:
                exit_code = 0
                artifacts_path = get_artifact('anomaly', run_id, tracking_uri)
                funcDeploy = FunctionDeployment(name='anomaly-prediction', gateway=of_params['gateway'],
                            username=of_params['username'], password=of_params['password'], version=of_params['version'])                    
                funcDeploy.build_yamls(model_path=f"{artifacts_path}/{vid}",
                                       label_mapping_path=f"{artifacts_path}/label_mapping.json")
                exit_code = funcDeploy.deploy()

                if exit_code == 0:
                    time.sleep(15)
                    logging.info(f"Deployed succesfulyy !")

                    pg_name = "AnomalyPrediction"
                    openfaas_url = f"https://openfaasgw.mobility-odometry.smart-mobility.alstom.com/function/anomaly-prediction-all" + \
                        "?filename=${filename}&bucket=${s3.bucket}"
                    create_nifi_group(
                        pg_name, minio_config, of_params, nifi_params, openfaas_url, bucket, "features")
                else:
                    logging.info(f"Failed to deploy !")
            except Exception as e:
                logging.error(
                    f"Failed to deploy, because of {e}")


def _runOdometryTraining(file_path: str,
                         vehicle_id: Union[List[Union[str, int]], str, int],
                         epochs: int = None,
                         deploy: bool = True):
    """Run training for single or multiple train to classify multiple dimensions

    Parameters
    ----------
    file_path : str
        training csv file path as minio bucket object
    vehicle_id : Union[List[Union[str, int]], str, int]
        train id
    epochs : int, optional
        Number of epochs to train, by default None
    deploy : bool, optional
        should model get deployed if it's best model, by default True
    """

    CONFIG = DEFAULT_CONFIG
    CONFIG['training'] = train_config['training']
    CONFIG['artifacts'] = train_config['artifacts']

    params = CONFIG['training']['odometry']
    mlflow_params = CONFIG['artifacts']['mlflow']
    of_params = CONFIG['artifacts']['openfaas']
    nifi_params = CONFIG['artifacts']['nifi']
    tracking_uri = mlflow_params['endpoint']
    username = mlflow_params['username']
    password = mlflow_params['password']

    data = params['data']
    bucket = data['bucket']
    features_dir = data['features_dir']
    model_params = params['Dnn']
    model_params['epochs'] = model_params['epochs'] if epochs is None else epochs

    minio_config = CONFIG['artifacts']['minio']
    minioClient = create_minio_client(minio_config["endpoint_url"],
                                      access_key=minio_config["access_key"],
                                      secret_key=minio_config["secret_key"],
                                      secure=minio_config['secure'])

    boto_client = boto3.client("s3",
                               endpoint_url=minio_config["endpoint_url"],
                               aws_access_key_id=minio_config["access_key"],
                               aws_secret_access_key=minio_config["secret_key"],
                               region_name=minio_config["region_name"])

    data['feedback_file_path'] = data['feedback_file_path'] if file_path is None else file_path

    if data['feedback_file_path'].endswith('.csv'):
        feedback_df, old_df = read_features(
            minioClient, bucket, data['feedback_file_path'], nominal_feature_name=data['nominal_feature_name'])
    else:
        fb_files = get_files(bucket, boto_client,
                             file_type='csv', prefix=data['feedback_file_path'])
        feedback_dfs = []
        for fb_file in fb_files:
            df, old_df = read_features(minioClient, bucket, fb_file,
                                       nominal_feature_name=data['nominal_feature_name'])
            feedback_dfs.append(df)
        feedback_df = pd.concat(feedback_dfs)

    zip_files = list(set([create_zip_filename(i.split('--')[0])
                          for i in list(feedback_df['Unnamed: 0'])]))
    feedback_df.set_index(data['index_col'], inplace=True)

    vehicle_id = data['vehicle_id'] if vehicle_id is None else vehicle_id
    vehicle_ids = [vehicle_id] if isinstance(vehicle_id, str) else vehicle_id

    default_status = copy.deepcopy(DEFAULT_STATE)
    default_status['epochs'] = model_params['epochs']
    status = defaultdict(dict)

    for vid in vehicle_ids:
        try:
            logging.info(f"Started training for vehicle {vid}")
            default_status['run_level']['Training'] = 'RUNNING'
            status[str(vid)] = default_status

            exist_status = read_json(os.path.join(CURRENT_DIR, 'state.json'))
            status.update(exist_status)
            write_json(status, os.path.join(CURRENT_DIR, 'state.json'))

            df_X, multioutput_target, X_test, y_test = class_training_data(feedback_df,
                                                                           validation_split=data['validation_split'],
                                                                           seed=data['seed'],
                                                                           vehicle_id=vid,
                                                                           dims=data['label_columns'])
            try:
                set_env('MLFLOW_TRACKING_USERNAME', username)
                set_env('MLFLOW_TRACKING_PASSWORD', password)
                set_env('MLFLOW_S3_ENDPOINT_URL', minio_config["endpoint_url"])
                set_env('AWS_ACCESS_KEY_ID', minio_config["access_key"])
                set_env('AWS_SECRET_ACCESS_KEY', minio_config["secret_key"])

                experiment_id, run_id = create_mlflow_run(
                    f'odometry_{vid}', tracking_uri)
                mlflowcb_params = {'experiment_id': experiment_id, 'run_id': run_id, 'tracking_uri': tracking_uri,
                                   'vid': vid, 'status': status, 'state_dir': CURRENT_DIR}
            except Exception as e:
                logging.error(
                    f'Failed to create run in mlflow, because of {str(e)}')
                mlflowcb_params = None

            df_X_test, multioutput_target_test = convert_xy(X_test, y_test)

            dnn = Dnn(**model_params, mlflowcb_params=mlflowcb_params)
            dnn.fit(df_X, multioutput_target, validation_data=(
                df_X_test, multioutput_target_test))
            model = dnn.model
            history = dnn.history.history

            mappings = create_label_mapping()
            test_predictions = predict_test(model, X_test, mappings)

            for c in y_test.columns:
                test_predictions[c] = y_test[c].tolist()

            save_dir = './logs/local'
            makedirs(save_dir)

            try:
                figures = get_all_plots(history, save_dir)
            except Exception as e:
                figures = []
                logging.error(f"Failed to create plots, because of {e}")

            tags = {'vehicle_id': vid}
            if experiment_id and run_id:
                log_to_mlflow(experiment_id, run_id, tracking_uri, CONFIG, zip_files, old_df, test_predictions,
                              keras_model=model, label_mapping=mappings, tags=tags, figures=figures)
            else:
                log_locally(CONFIG, zip_files, old_df, test_predictions, keras_model=model,
                            label_mapping=mappings, tags=tags, save_dir=save_dir)

            status[str(vid)]['run_level']['Training'] = 'COMPLETED'
            write_json(status, os.path.join(CURRENT_DIR, 'state.json'))

            perf_metrics = None
            save_metrics = True

            if save_metrics:
                try:
                    status[str(vid)]['run_level']['AggregateMetrics'] = 'RUNNING'
                    write_json(status, os.path.join(CURRENT_DIR, 'state.json'))

                    mlfc = MlflowClient(tracking_uri=tracking_uri)
                    perf_metrics = mlfc.get_perf_metrics(f"odometry_{vid}")
                    minioClient.write_dict_to_minio(
                        perf_metrics, bucket, f'results/{vid}.json')
                    status[str(vid)]['run_level']['AggregateMetrics'] = 'COMPLETED'
                except Exception as e:
                    status[str(vid)]['run_level']['AggregateMetrics'] = 'FAILED'
                    logging.error(
                        f"Failed to calculate performance metrics, because of {e}")
                write_json(status, os.path.join(CURRENT_DIR, 'state.json'))

            if deploy:

                try:
                    status[str(vid)]['run_level']['Deployment'] = 'RUNNING'
                    write_json(status, os.path.join(CURRENT_DIR, 'state.json'))

                    pf = minioClient.get_dict(
                        bucket, f'results/{vid}.json') if perf_metrics is None else perf_metrics
                    artifacts_path = get_artifact(
                        f'odometry_{vid}', run_id, tracking_uri)

                    exit_code = 0
                    if pf['best_run'] is None or pf['best_run'] == run_id:
                        funcDeploy = FunctionDeployment(name='odometry-prediction', gateway=of_params['gateway'],
                                    username=of_params['username'], password=of_params['password'], version=of_params['version'])
                        funcDeploy.build_yamls(model_path=f"{artifacts_path}/{vid}",
                                               label_mapping_path=f"{artifacts_path}/label_mapping.json")
                        exit_code = funcDeploy.deploy()

                    if exit_code == 0:
                        time.sleep(15)
                        status[str(vid)]['run_level']['Deployment'] = 'COMPLETED'
                        logging.info(f"STATUS: {status}")

                        pg_name = f"ClassificationPrediction_{vid}"
                        prefix = f"{features_dir}/{vid}"
                        openfaas_url = f"https://openfaasgw.mobility-odometry.smart-mobility.alstom.com/function/odometry-prediction-{vid}" + \
                                    "?filename=${filename}&bucket=${s3.bucket}"
                        create_nifi_group(
                            pg_name, minio_config, of_params, nifi_params, openfaas_url, bucket, prefix)
                    else:
                        status[str(vid)]['run_level']['Deployment'] = 'FAILED'

                    logging.info(f"STATUS: {status}")

                except Exception as e:
                    status[str(vid)]['run_level']['Deployment'] = 'FAILED'
                    logging.error(f"Failed to deploy, because of {e}")

                write_json(status, os.path.join(CURRENT_DIR, 'state.json'))

        except Exception as e:
            import traceback
            traceback.print_exc()
            status[str(vid)]['run_level']['Training'] = 'FAILED'
            write_json(status, os.path.join(CURRENT_DIR, 'state.json'))
            logging.error(f"Failed training for vehicle {vid}, because of {e}")


def runOdometryTraining(file_path: str,
                        vehicle_id: Union[List[Union[str, int]], str, int],
                        epochs: int = None,
                        num_workers: int = None,
                        deploy: bool = True):
    """Run training for single or multiple train to classify multiple dimensions

    Parameters
    ----------
    file_path : str
        training csv file path as minio bucket object
    vehicle_id : Union[List[Union[str, int]], str, int]
        train id
    epochs : int, optional
        Number of epochs to train, by default None
    num_workers : int, optional
        Number of workers to train, by default it is total cpu - 2        
    deploy : bool, optional
        should model get deployed if it's best model, by default True
    """

    write_json({}, os.path.join(CURRENT_DIR, 'state.json'))

    vehicle_ids = vehicle_id if isinstance(vehicle_id, list) else [vehicle_id]
    num_workers = multiprocessing.cpu_count() - 2 if num_workers is None else num_workers
    with multiprocessing.Pool(num_workers) as pool:
        pool.map(partial(_runOdometryTraining,
                         file_path, epochs=epochs), vehicle_ids)
