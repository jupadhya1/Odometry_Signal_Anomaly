import logging
import os
import copy
from typing import List, Union, Dict

from .utils.io import read_yaml, write_yaml


class FunctionDeployment():
    """
        Deploy any openfass funtion
    """    
    def __init__(self, name: str, gateway: str, username: str, password: str, version: float=1.0, relative_path: str='openfaas'):
        """Genearte a openfaas deployment object

        Parameters
        ----------
        name : str
            name of the openfaas function
        gateway : str
            openfaas gateway
        version : float, optional
            openfaas version, by default 1.0
        relative_path : str, optional
            openfaas funtion exists relative to which directory, by default 'openfaas'

        Raises
        ------
        ValueError
            if openfaas funtion is not given
        """               
        self.name = name
        self.gateway = gateway
        self.username = username
        self.password = password
        self.version = version
        self.relative_path = relative_path
        self.yaml_path = None
        self.deployment_yaml_path = None

        if name is None:
            raise ValueError(
                f"`name` could not be None, please provide fucntion name")

        self.check_if_exists()

    def _build_yaml(self, model_yaml: Dict, old_func_name:str, model_path: str, label_mapping_path: str = None, image_name: str = None, secrets: List[str] = None):
        new_func_name = f"{self.name}-{os.path.basename(model_path.rstrip('/'))}"
        old_function = copy.deepcopy(model_yaml['functions'][old_func_name])
        old_function["environment"]["MODEL_PATH"] = model_path

        if image_name is not None:
            old_function['image'] = image_name
        if label_mapping_path is not None:
            old_function["environment"]["LABEL_MAPPING_PATH"] = label_mapping_path
        if secrets is not None:
            old_function['secrets'] = secrets

        model_yaml['functions'][new_func_name] = old_function

    def build_yamls(self, model_path: Union[str, List[str]],
                    label_mapping_path: str = None,
                    image_name: str = None,
                    secrets: List[str] = None):
        """built deployment yaml

        Parameters
        ----------
        model_path : Union[str, List[str]]
            Moel path to set enviornment variable
        label_mapping_path : str, optional
            class label mapping. Defaults to None., by default None
        image_name : str, optional
            docker image name. Defaults to None., by default None
        secrets : List[str], optional
            secret name e.g. minio-secret. Defaults to None., by default None

        Returns
        -------
        [List]
            list of deployment yaml path
        """        
       
        model_yaml = read_yaml(self.yaml_path)
        old_func_name = list(model_yaml['functions'].keys())[0]
        model_paths = [model_path] if isinstance(model_path, str) else model_path
        
        model_yaml["version"] = self.version
        model_yaml['provider']['gateway'] = self.gateway

        for mp in model_paths:
            self._build_yaml(model_yaml, old_func_name, mp, label_mapping_path=label_mapping_path, image_name=image_name, secrets=secrets)

        self.deployment_yaml_path = os.path.join(os.path.dirname(
            self.yaml_path), f"{self.name}-new.yml")

        del model_yaml['functions'][self.name]
        write_yaml(model_yaml, self.deployment_yaml_path)

        logging.info(f"Final deployment yaml is {model_yaml}")
        logging.info(f"Created deployment yaml {os.path.basename(self.deployment_yaml_path)}")
        return self.deployment_yaml_path

    def deploy(self, old_yaml: bool = False):
        """deploy yaml as openfaas function

        Parameters
        ----------
        old_yaml : bool, optional
            choose if already yaml have to deploy. Defaults to False., by default False

        Returns
        -------
        [type]
            [description]

        Raises
        ------
        ValueError
            provide either of arguments
        """        

        if not old_yaml and self.deployment_yaml_path is None:
            raise ValueError(
                f"Either use `build_yamls` method to deploy new model or change argument `old_yaml` to `True` to deploy same model again.")
        if old_yaml and self.deployment_yaml_path is not None:
            raise ValueError(
                f"Either use old_yaml or build new yaml to deploy.")

        if old_yaml:
            exit_code = self._deploy(self.yaml_path)
        if self.deployment_yaml_path is not None:
            exit_code = self._deploy(self.deployment_yaml_path)
        return exit_code

    def _deploy(self, file_path):
        logging.info(f"Deploying {os.path.basename(file_path)}")
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        os.chdir(CURRENT_DIR)
        os.system(f"faas-cli login -u {self.username} -p {self.password} -g {self.gateway}")
        exit_code = os.system(f"faas-cli deploy -f {file_path}")
        if exit_code == 0:
            logging.info(f"Succesfully Deployed {os.path.basename(file_path)}")
        else:
            logging.error(f"{os.path.basename(file_path)} couldn't deployed")   
        return exit_code     

    def check_if_exists(self):
        """check if openfaas function temlate exist

        Returns
        -------
        [bool]
            True if function exists

        Raises
        ------
        ValueError
            if function name not provided]
        """              
        dir_name = os.path.dirname(os.path.abspath(__file__))
        fucntion_dir = os.path.join(dir_name, 'openfaas', self.name)
        if not os.path.isdir(fucntion_dir):
            raise ValueError(
                f"Function name `{self.name}` provided does not exist.")
        self.yaml_path = os.path.join(fucntion_dir, f"{self.name}.yml")
        return True
