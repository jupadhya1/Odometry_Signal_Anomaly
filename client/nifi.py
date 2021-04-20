import logging
from typing import Union, Dict, List
from random import randrange
import nipyapi


class Nifi():
    """A nifi client to create nifi process group
    """    
    def __init__(self, host, registry=None):        
        self.host = host
        self.registry = registry
        nipyapi.config.nifi_config.host = host
        logging.info(f"Starting with pg {nipyapi.canvas.get_root_pg_id()}")

    def create_process_group(self, name: str, parent_pg=None):
        """create nifi process group

        Parameters
        ----------
        name : str
            nifi process group name
        parent_pg : nipyapi.canvas, optional
            parent process group object, by default None

        Returns
        -------
        process group object
            nifi process group object
        """        
        if parent_pg is None:
            parent_pg = nipyapi.canvas.get_process_group(
                nipyapi.canvas.get_root_pg_id(), 'id')
        location = (randrange(100, 1400), randrange(200, 1000))
        process_group = nipyapi.canvas.create_process_group(
            parent_pg, name, location=location)
        return process_group

    def get_process_group(self, name: str):
        """get nifi process group object

        Parameters
        ----------
        name : str
            nifi process group name

        Returns
        -------
        process group object
            nifi process group object
        """        
        process_group = nipyapi.canvas.get_process_group(name)
        return process_group

    def get_process_group_id(self, name: str):
        """nifi process group id

        Parameters
        ----------
        name : str
            nifi process group name

        Returns
        -------
        str
            process group id name
        """        
        process_group = nipyapi.canvas.get_process_group(name)
        return process_group.id

    def create_processor(self, name: str,
                         process_group: str,
                         processor_type: str = 'GenerateFlowFile',
                         properties: Dict = None,
                         scheduling_period: str = '0s',
                         location: tuple = (400.0, 400.0),
                         relationships: List[str] = ['success']):
        """create nifi processor of a particular type

        Parameters
        ----------
        name : str
            nifi processor name
        process_group : str
            nifi process group name
        processor_type : str, optional
            nifi processor type, by default 'GenerateFlowFile'
        properties : Dict, optional
            nifi processor properties, by default None
        scheduling_period : str, optional
            schedule period for pooling request, by default '0s'
        location : tuple, optional
            location on nifi canvas for processor, by default (400.0, 400.0)
        relationships : List[str], optional
            relationships, by default ['success']

        Returns
        -------
        nifi processor object
            nifi processor
        """                         
        process_group = self.get_process_group(process_group) if isinstance(
            process_group, str) else process_group

        logging.info(
            f"Creating processor `{name}` as a new {processor_type} in process group `{process_group}`")
        processor = nipyapi.canvas.create_processor(
            parent_pg=process_group,
            processor=nipyapi.canvas.get_processor_type(processor_type),
            location=location,
            name=name,
            config=nipyapi.nifi.ProcessorConfigDTO(
                scheduling_strategy='TIMER_DRIVEN',
                scheduling_period=scheduling_period,
                auto_terminated_relationships=relationships,
                properties=properties
            )
        )
        logging.info(f"Processor {name} created !")

        return processor

    def update_processor(self, name, properties: Dict = None):
        processor = nipyapi.canvas.get_processor(name)
        nipyapi.canvas.update_processor(
            processor=processor,
            update=nipyapi.nifi.ProcessorConfigDTO(
                scheduling_period='1s'
            )
        )

    def total_pgs(self, pg_id: str = 'root'):
        """get total number of process groups

        Parameters
        ----------
        pg_id : str, optional
            process group id, by default 'root'

        Returns
        -------
        int
            total number of process groups
        """        
        pgs = nipyapi.canvas.list_all_process_groups(pg_id=pg_id)
        return len(pgs)

    def schedule_process_group(self, pg_id: str, scheduled: bool = True) -> bool:
        """schedule nifi process group whether to start or stop a process group

        Parameters
        ----------
        pg_id : str
             nifi process group id
        scheduled : bool, optional
            start or stop nifi process group, by default True

        Returns
        -------
        bool
            success or failed
        """        
        return nipyapi.canvas.schedule_process_group(pg_id, scheduled)

    def schedule_processor(self, processor, scheduled: bool = True) -> bool:
        """schedule nifi processor whether to start or stop a process group

        Parameters
        ----------
        pg_id : str
             nifi process group id
        scheduled : bool, optional
            start or stop nifi processor, by default True

        Returns
        -------
        bool
            success or failed
        """    
        return nipyapi.canvas.schedule_processor(processor, scheduled)

    def create_connection(self, source, target):
        """create connection between processors

        Parameters
        ----------
        source : nifi processor object
            nifi processor object
        target : nifi processor object
            nifi processor object
        """        
        nipyapi.canvas.create_connection(source, target)

    def create_and_run_pg(self, pg_name: str,
                          bucket: str,
                          prefix: str,
                          openfaas_url: str,
                          minio_creds: Dict = {},
                          openfaas_creds: Dict = {}):
        """create and schedule a nifi process group

        Parameters
        ----------
        pg_name : str
            process group name
        bucket : str
            minio bucket to list files from bucket
        prefix : str
            minio bucket object path 
        openfaas_url : str
            openfaas endpoint url
        minio_creds : Dict, optional
            minio credentials, by default {}
        openfaas_creds : Dict, optional
            openfaas credentials, by default {}
        """        
        pg = self.create_process_group(pg_name)
        s3_props = Nifi._s3_properties()
        http_props = Nifi._invoke_properties()

        s3_props['Bucket'] = bucket
        s3_props['prefix'] = prefix
        s3_props.update(minio_creds)
        http_props['Remote URL'] = openfaas_url
        http_props.update(openfaas_creds)

        lists3_proc = self.create_processor('listings3', pg,
                                            processor_type='ListS3',
                                            properties=s3_props,
                                            location=(1000.0, 400.0),
                                            relationships=['success'])
        http_proc = self.create_processor('invokehttp', pg,
                                          processor_type='InvokeHTTP',
                                          properties=http_props,
                                          location=(1200.0, 600.0),
                                          relationships=['Failure', 'No Retry', 'Original', 'Response', 'Retry'])
        self.create_connection(lists3_proc, http_proc)
        self.schedule_process_group(pg.id, scheduled=True)

    def check_if_pg_exists(self, name):
        """check if nifi process group exist

        Parameters
        ----------
        name : str
            process group name

        Returns
        -------
        bool
            true or false / exists or not
        """        
        pgs = nipyapi.canvas.list_all_process_groups(pg_id='root')
        pg_names = [pg.component.name for pg in pgs]
        if name in pg_names:
            return True
        return False

    @staticmethod
    def _s3_properties():
        props = {
            'Bucket': 'odometryclassification',
            'list-type': '2',
            'prefix': 'features',
            'Access Key': '',
            'Secret Key': '',
            'Endpoint Override URL': ''
        }
        return props

    @staticmethod
    def _invoke_properties():
        props = {
            'HTTP Method': 'POST',
            'Remote URL': '',
            'Basic Authentication Username': '',
            'Basic Authentication Password': ''
        }
        return props
