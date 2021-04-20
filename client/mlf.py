import json
import logging
import math
from collections import defaultdict
from typing import Dict, List, Union

import mlflow
import pandas as pd
from sklearn.metrics import confusion_matrix


class MlflowClient():
    """A mlflow client to read artifacts from minio bucket
    """    
    def __init__(self, tracking_uri: str):
        mlflow.set_tracking_uri(tracking_uri)
        self.mlflow_client = mlflow.tracking.MlflowClient()

    def get_artifact(self, run_id: str, relative_path: str):
        """read artifact presented in mlflow

        Parameters
        ----------
        run_id (str): run id for an experiment performed
        relative_path (str): path respect to artifact root uri

        Returns:
            Union[dict, pd.DataFrame]: data after reading artifact file
        """        
        local_path = self.mlflow_client.download_artifacts(
            run_id, relative_path)
        data = None
        if local_path.endswith('.json'):
            with open(local_path, 'r') as f:
                data = json.load(f)
        if local_path.endswith('.csv'):
            data = pd.read_csv(local_path)
        return data

    def get_experiment_id(self, name: str):
        """get experiment id by its name

        Parameters
        ----------
        name (str): mlflow experiment name

        Returns:
            int: mlflow experiment id
        """        
        mlflow.set_experiment(name)
        ex = mlflow.get_experiment_by_name(name)
        return ex.experiment_id

    def get_runs(self, name: str):
        """get all the runs with given experiment name

        Parameters
        ----------
        name (str): experiment name

        Returns:
            List: all the run ids
        """        
        ex_id = self.get_experiment_id(name)
        run_infos = self.mlflow_client.list_run_infos(experiment_id=ex_id)
        run_ids = [run_info.run_id for run_info in run_infos]
        return run_ids

    def _change_pred(self, predictions: pd.DataFrame, label_mapping: Dict):
        for dim in label_mapping.keys():
            reverse_dict = {v:k for k,v in label_mapping[dim].items()}
            predictions.replace({dim: reverse_dict}, inplace=True)
        return predictions

    def get_cm(self, y_true: List[str], y_pred: List[str], labels: List[str]):
        """get confusion matrix

        Parameters
        ----------
        y_true (List[str]): true label 
        y_pred (List[str]): predicted label
        labels (List[str]): labels during training

        Returns:
            List: confusion matrix
        """        
        cm_array = confusion_matrix(y_true, y_pred, labels=labels)
        cm = [list(row) for row in cm_array]
        cm_int = [[int(s) for s in sublist] for sublist in cm]
        return cm_int

    def _get_labels(self, label_mapping: Dict):
        reverse_dict = {v:k for k,v in label_mapping.items()}
        labels_list = list(reverse_dict.keys())
        labels_list.sort()
        labels = [reverse_dict[i] for i in labels_list]
        return labels

    def get_perf_metrics(self, ex_name: str, metric: str='val_loss'):
        """get performance metrics corresponding a particular train

        Parameters
        ----------
        ex_name (str): experiment name
        metric (str, optional): metric on which performance has to calculate. Defaults to 'val_loss'.

        Returns:
            Dict: performance metrics
        """     
        run_ids = self.get_runs(ex_name)

        metric_perfs = nested_dict()
        best_run_id = None
        best_val_loss = math.inf
        successful_runs = []

        for run_id in run_ids:
            try:
                logging.info(f"getting artifacts for {run_id} to measure model performance")
                label_mapping = self.get_artifact(run_id, 'label_mapping.json')
                metrics = self.get_artifact(run_id, 'metric.json')
                test_predictions = self.get_artifact(run_id, 'test_predictions.csv')
                predictions = self._change_pred(test_predictions, label_mapping)

                axle_labels = self._get_labels(label_mapping['Axle Event'])
                axle_y_true, axle_y_pred = predictions['Axle Event'], predictions['axle_event']
                metric_perfs['runs_metric'][run_id]['axle_event']['labels'] = axle_labels
                metric_perfs['runs_metric'][run_id]['axle_event']['confusion_matrix'] = self.get_cm(axle_y_true, axle_y_pred, axle_labels)
                metric_perfs['runs_metric'][run_id]['axle_event']['train_loss'] = metrics['axle_output_loss']
                metric_perfs['runs_metric'][run_id]['axle_event']['val_loss'] = metrics['val_axle_output_loss']
                metric_perfs['runs_metric'][run_id]['axle_event']['train_acc'] = metrics['axle_output_accuracy']
                metric_perfs['runs_metric'][run_id]['axle_event']['val_acc'] = metrics['val_axle_output_accuracy']

                odo_labels = self._get_labels(label_mapping['Odo Algo Issue'])
                odo_y_true, odo_y_pred = predictions['Odo Algo Issue'], predictions['odo_algo_issue']
                metric_perfs['runs_metric'][run_id]['odo_algo_issue']['labels'] = self._get_labels(label_mapping['Odo Algo Issue'])
                metric_perfs['runs_metric'][run_id]['odo_algo_issue']['confusion_matrix'] = self.get_cm(odo_y_true, odo_y_pred, odo_labels)
                metric_perfs['runs_metric'][run_id]['odo_algo_issue']['train_loss'] = metrics['odo_output_loss']
                metric_perfs['runs_metric'][run_id]['odo_algo_issue']['val_loss'] = metrics['val_odo_output_loss']
                metric_perfs['runs_metric'][run_id]['odo_algo_issue']['train_acc'] = metrics['odo_output_accuracy']
                metric_perfs['runs_metric'][run_id]['odo_algo_issue']['val_acc'] = metrics['val_odo_output_accuracy']

                speed_labels = self._get_labels(label_mapping['Speed Estimation'])
                speed_y_true, speed_y_pred = predictions['Speed Estimation'], predictions['speed_estimation']
                metric_perfs['runs_metric'][run_id]['speed_estimation']['labels'] = self._get_labels(label_mapping['Speed Estimation'])
                metric_perfs['runs_metric'][run_id]['speed_estimation']['confusion_matrix'] = self.get_cm(speed_y_true, speed_y_pred, speed_labels)
                metric_perfs['runs_metric'][run_id]['speed_estimation']['train_loss'] = metrics['speed_output_loss']
                metric_perfs['runs_metric'][run_id]['speed_estimation']['val_loss'] = metrics['val_speed_output_loss']
                metric_perfs['runs_metric'][run_id]['speed_estimation']['train_acc'] = metrics['speed_output_accuracy']
                metric_perfs['runs_metric'][run_id]['speed_estimation']['val_acc'] = metrics['val_speed_output_accuracy']

                metric_perfs['runs_metric'][run_id]['all']['train_loss'] = metrics['loss']
                metric_perfs['runs_metric'][run_id]['all']['val_loss'] = metrics['val_loss']

                successful_runs.append(run_id)

                if metrics[metric] < best_val_loss:
                    best_val_loss = metrics[metric]
                    best_run_id = run_id
            
            except Exception as e:
                logging.error(f"failed to get artifacts for {run_id}, because of {e}")
                continue

        metric_perfs['runs'] = successful_runs
        metric_perfs['best_run'] = best_run_id
        metric_perfs['best_metric'] = metric
        metric_perfs['metric_value'] = best_val_loss

        return dict(metric_perfs)


def nested_dict():
    """ Fucntion to define a nested dictionary

    Returns:
        collections.defaultdict: a dictionary
    """    
    return defaultdict(nested_dict)

