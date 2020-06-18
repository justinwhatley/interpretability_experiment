"""
Setup files
"""

def run_setup(config_path='../config.ini', overwrite_dataset=False, test_size=0.5):
    """
    Setup loader 
    """
    # Load config
    config_path = '../config.ini'
    config = load_config(config_path)
    
    # Start dask client
    dask_client = init_local_dask()
    
    # Create dataset manager
    dataset_manager = init_dataset_manager(config)
    dataset_manager.write_dataset(test_size=test_size, overwrite=overwrite_dataset)
    
    # Get dataloader
    dataloader = init_dataloader(dataset_manager)
    
    return config, dask_client, dataset_manager, dataloader
    


def load_config(config_path='config.ini'):
    """
    Loads common configuration parameters
    """
    from project.utils.setup.configuration_manager import Config
    return Config(config_path)


def init_local_dask(dashboard_address=':20100', memory_limit='4G'):
    """
    Set up local cluster
    """
    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(dashboard_address=dashboard_address, memory_limit=memory_limit)
    client = Client(cluster)
    print(client)
    return client


def init_dataset_manager(config):
    """
    Set up dataset manager to handle dataset manipulations 
    """
    from project.utils.preprocessing.dataset_manager import DatasetManager
    return DatasetManager(config)


def init_dataloader(dataset_manager):
    """
    Sets up dataloader to feed data gradually to learning algorithms
    """
    import project.experiments.dataset.dataloader as dl
    dataloader = dl.DataLoader(dataset_manager)
    return dataloader