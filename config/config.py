CONFIG = {
    'steps': {
        'DataExtraction': {
            'inputs': {
                'no_of_files': 3,
                'bucket': 'odometry',
                'nrows': None,
                'columns': ["Unnamed: 0", "Unnamed: 1", "speedLimit",
                            "trainSpeed", "targetSpeed",
                            "TransponderOK",
                            "MSTEP_A_axle1RawSpeed",
                            "MSTEP_A_axle2RawSpeed",
                            "warningSpeed",
                            "line", "VehicleID"],
            },
            'outputs': {
                'csv_name_prefix': 'filtered',
                'bucket': 'odometryclassification'
            }
        },
        'EventCreation': {
            'inputs': {
                'min_window_size': 1,
                'change_speed_by': 10,
                'train_zero_speed_ratio': 0.4,
                'datetime_limit': None,
                'window_events_file': 'window_events.csv',
                'window_event_bucket': None,
                'csv_name_prefix': 'filtered',
                'bucket': 'odometryclassification'
            },
            'outputs': {
                'bucket': 'odometryclassification',
                'event_dir': 'events1min',
                'filename_include': 'nominal'
            }
        },
        'CreateFeatures': {
            'inputs': {
                'speed_vars': ["axle1RawSpeed", "axle2RawSpeed", "trainSpeed"],
                'sample_value': '500L',
                'nominal_feature_name': 'is_nominal',
                'filename_include': 'nominal',
                'event_dir': 'events1min',
                'bucket': 'odometryclassification'
            },
            'outputs': {
                'bucket': 'odometryclassification',
                'features_dir': 'features',
                'features_path': 'features1min.csv'
            }
        }
    },
    'artifacts': {
        'minio': {
            'access_key': 'mobilityodometry',
            'secret_key': '0abaZw+BOFtrAD+tLH/kYGoX7ljpNa38/BMKBakSYD8jufmvb18RWDhcqxo1hjPG504eAr9ejBmOqL7Je1peQQ==',
            'endpoint_url': 'https://raw-data-storage.mobility-odometry.smart-mobility.alstom.com',
            'region_name': 'us-east-1',
            'secure': True
        },
        'mlflow': {
            'endpoint': 'http://mlflow.mobility-odometry.smart-mobility.alstom.com',
            'username': 'mlflow',
            'password': 'jei7viphoo4Kae&j'
        },
        "openfaas": {
            "gateway": "https://openfaasgw.mobility-odometry.smart-mobility.alstom.com",
            "version": 1.0
        },
        "nifi": {
            "host": "http://nifi.default.svc.cluster.local:8080/nifi-api"
        }        
    },
    'training': {
        "anomaly": {
            'model': {
                'IsolationForest': True
            },
            'model_params': {
                'IsolationForest': {
                    'n_estimators': 1000,
                    'contamination': 0
                }
            },
            'data': {
                'bucket': 'odometryclassification',
                'features_dir': 'features/4005',
                'file_path': None,
                'index_col': 'Unnamed: 0',
                'vehicle_id': None,
                'test_ratio': 0.2,
                'metrics_nsteps': 10,
                'nominal_sample_frac': 1,
                'nominal_feature_name': 'is_nominal'
            }
        },
        "odometry": {
            'Dnn': {
                'optimizer': 'Adam',
                'learning_rate': 0.001,
                'loss_weight': 1.0,
                'loss': 'sparse_categorical_crossentropy',
                'epochs': 2,
                'verbose': 1
            },
            'data': {
                'bucket': 'odometryclassification',
                'features_dir': 'features',
                'feedback_file_path': None,
                'index_col': 'Unnamed: 0',
                'validation_split': 0.3,
                'seed': 2020,
                'label_columns': ["Axle Event", "Odo Algo Issue", "Speed Estimation"],
                'vehicle_id': None,
                'nominal_feature_name': 'is_nominal'
            }
        }
    }
}
