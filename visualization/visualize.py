import logging
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from ..features.speed_profile_handling import compute_features
from ..utils.io import makedirs, read_json, write_json


def plot_profiles(df, metadata=None, save_dir='/tmp'):
    plt.figure(figsize=(15, 5))
    if metadata is not None:
        vid = metadata['VehicleID']
        rel = metadata['Relative Error (%)']
        com = metadata["Comment"]
        plt.title(
            f"Speed profiles - VehicleID {vid}, Relative Error = {rel}, reason: {com}")
    plt.plot(df["axle1RawSpeed"], label="axle1RawSpeed")
    plt.plot(df["axle2RawSpeed"], label="axle2RawSpeed")
    plt.plot(df["trainSpeed"], label="trainSpeed")
    plt.legend()
    plt.savefig(f'{save_dir}/profiles.jpg')
    return f'{save_dir}/profiles.jpg'    


def plot_features(df, features=[], nominal_feature_name='is_nominal', metadata=None, save_dir='/tmp'):
    for f in features:
        plt.figure(figsize=(10, 7))
        if metadata is not None:
            vid = metadata['VehicleID']
            rel = metadata['Relative Error (%)']
            com = metadata["Comment"]
            plt.title(
                f"Speed profiles - VehicleID {vid}, Relative Error = {rel}, reason: {com}")
        else:
            plt.title(f)

        plt.hist(df[df[nominal_feature_name] == True][f],  bins=40, density=True, range=(
            df[f].min()-1e-6, df[f].max()+1e-6), alpha=0.5, label="nominal")
        plt.hist(df[df[nominal_feature_name] == False][f], bins=40, density=True, range=(
            df[f].min()-1e-6, df[f].max()+1e-6), alpha=0.5, label="anomalous")
        plt.legend()
        plt.savefig(f'{save_dir}/features.jpg')
    return f'{save_dir}/features.jpg'        


def plot_features_grid(df, features=[], vehicle_id: str = "2048", nominal_feature_name='is_nominal', metadata=None, log=False, save_dir='/tmp'):
    if isinstance(vehicle_id, int):
        vehicle_id = str(vehicle_id)

    one_vehicle_df = df[[vehicle_id in f for f in df.index]]
    df = one_vehicle_df

    features = df.columns.drop(nominal_feature_name)
    n_features = len(features)

    n_cols = 5
    nrows = n_features//n_cols if (n_features %
                                   n_cols == 0) else n_features//n_cols + 1
    fig, axs = plt.subplots(nrows, n_cols, figsize=(n_cols*5, nrows*4))
    axs = axs.ravel()

    for i, f in enumerate(features):
        if metadata is not None:
            vid = metadata['VehicleID']
            rel = metadata['Relative Error (%)']
            com = metadata["Comment"]
            axs[i].set_title(
                f"Speed profiles - VehicleID {vid}, Relative Error = {rel}, reason: {com}")
        else:
            axs[i].set_title(f)

        xn, binsn, pn = axs[i].hist(df[df[nominal_feature_name] == True][f],  bins=40, density=True, range=(
            df[f].min()-1e-6, df[f].max()+1e-6), alpha=0.5, label="nominal")
        xa, binsa, pa = axs[i].hist(df[df[nominal_feature_name] == False][f], bins=40, density=True, range=(
            df[f].min()-1e-6, df[f].max()+1e-6), alpha=0.5, label="anomalous")

        for item in pn:
            item.set_height(item.get_height()/np.max(xn))
        for item in pa:
            item.set_height(item.get_height()/np.max(xa))
        if log:
            axs[i].set_yscale('log')
            axs[i].set_ylim(0, 10)
        else:
            axs[i].set_ylim(0, 1.2)

    plt.legend()
    plt.savefig(f'{save_dir}/features_grid.jpg')
    return f'{save_dir}/features_grid.jpg'    


def plot_event(event_filename: str,
                  bucket_name: str,
                  minio_client,
                  features=[],
                  sample_value: str = "500L",
                  speed_vars: List[str] = None,
                  save_dir='/tmp'):
    if speed_vars is None:
        speed_vars = ["axle1RawSpeed", "axle2RawSpeed", "trainSpeed"]
    if isinstance(sample_value, int):
        sample_value = str(str(sample_value) + 'L')

    df_event = minio_client.get_pickle(event_filename, bucket_name)
    df_event = df_event.resample(sample_value).mean()

    for v in speed_vars:
        df_event[v] = df_event[v].astype(np.int16)

    plot_profiles(df_event)
    print(df_event["axle1RawSpeed"].diff().dropna().mean())
#     df_event["axle1RawSpeed"].diff().dropna().plot()
#     df_event["axle1RawSpeed"].plot()

    features = compute_features(df_event, *speed_vars)
    # pprint(features)


def plot_all_metrics(recalls, precisions, f1s, thresholds, bt, save_dir='/tmp'):
    plt.plot(thresholds, precisions, alpha=0.5, label="precision")
    plt.plot(thresholds, recalls,    alpha=0.5, label="recall")
    plt.plot(thresholds, f1s, label="f1")
    plt.axvline(bt, color="red", linewidth=3)

    plt.title(f"Best threshold for anomaly detection: {bt}")

    plt.legend()
    plt.savefig(f'{save_dir}/metrics.jpg')    
    logging.info(f"prf metrics saved !")
    return f'{save_dir}/metrics.jpg'    


def plot_model_acc(model, save_dir='/tmp'):
    plt.figure()
    plt.plot(model["val_axle_output_accuracy"], label="accuracy axle event")
    plt.plot(model["val_odo_output_accuracy"], label="accuracy odometry algo issue")
    plt.plot(model["val_speed_output_accuracy"], label="accuracy speed estimation")
    # plt.ylim(0.9,1.01)
    plt.legend()
    plt.savefig(f'{save_dir}/accuracy.jpg')    
    logging.info(f"model accuracy figure saved !") 
    return f'{save_dir}/accuracy.jpg'    


def plot_val_loss(model, save_dir='/tmp'):
    plt.figure()
    plt.plot(model["val_axle_output_loss"], label="validation loss axle event")
    plt.plot(model["val_odo_output_loss"], label="validation loss odometry algo issue")
    plt.plot(model["val_speed_output_loss"], label="validation loss speed estimation")
    plt.legend()
    plt.semilogy()
    plt.savefig(f'{save_dir}/val_loss.jpg')    
    logging.info(f"validation loss figure saved !")
    return f'{save_dir}/val_loss.jpg'    


def plot_train_val_loss(model, save_dir='/tmp'):
    plt.figure()
    plt.plot(model["axle_output_loss"], label="train loss axle event")
    plt.plot(model["val_axle_output_loss"], label="validation loss axle event")
    plt.legend()
    plt.semilogy()
    plt.savefig(f'{save_dir}/axle_train_val_loss.jpg')
    logging.info(f"axle train loss figure saved !")

    plt.figure()
    plt.plot(model["odo_output_loss"], label="train loss odometry algo issue")
    plt.plot(model["val_odo_output_loss"], label="validation loss odometry algo issue")
    plt.legend()
    plt.semilogy()
    plt.savefig(f'{save_dir}/odo_train_val_loss.jpg')
    logging.info(f"odo train loss figure saved !")

    plt.figure()
    plt.plot(model["speed_output_loss"], label="train loss speed estimation")
    plt.plot(model["val_speed_output_loss"], label="validation loss speed estimation")
    plt.legend()
    plt.semilogy()
    plt.savefig(f'{save_dir}/speed_train_val_loss.jpg')  
    logging.info(f"speed train loss figure saved !")
    return f'{save_dir}/axle_train_val_loss.jpg', f'{save_dir}/odo_train_val_loss.jpg', f'{save_dir}/speed_train_val_loss.jpg'    


def plot_train_valid_global_loss(model, save_dir='/tmp'):
    plt.figure()
    plt.plot(model["loss"], label="train loss global")
    plt.plot(model["val_loss"], label="validation loss global")
    plt.legend()
    plt.semilogy()
    plt.savefig(f'{save_dir}/train_valid_global_loss.jpg')   
    logging.info(f"global train loss figure saved !")   
    return f'{save_dir}/train_valid_global_loss.jpg'    


def get_all_plots(history, save_dir):
    save_figs_dir = os.path.join(save_dir, 'figures')
    makedirs(save_figs_dir)
    plt_model_acc = plot_model_acc(history, save_dir=save_figs_dir)
    plt_val_loss = plot_val_loss(history, save_dir=save_figs_dir)
    plt_axle_loss, plt_odo_loss, plt_speed_loss = plot_train_val_loss(
        history, save_dir=save_figs_dir)
    plt_train_valid_global_loss = plot_train_valid_global_loss(
        history, save_dir=save_figs_dir)
    figures = [plt_model_acc, plt_val_loss, plt_axle_loss,
            plt_odo_loss, plt_speed_loss, plt_train_valid_global_loss] 
    return figures
