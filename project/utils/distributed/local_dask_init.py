#TODO 
"""
Singleton initiation of Dask local instance
"""
from dask.distributed import Client, LocalCluster

class DaskLocal:
    class __DaskLocal:
        def __init__(self, dash_add):
            dash_add
            cluster = LocalCluster(dashboard_address=dash_add)
            print('Setting new client')
            client = Client(cluster)
            print(client)
            
        def __str__(self):
            return repr(self) + self.val
    
    client_instance = None
    def __init__(self, arg):
        if not DaskLocal.client_instance:
            DaskLocal.client_instance = DaskLocal.__DaskLocal(arg)
        else:
            DaskLocal.instance.val = arg
            
    def __getattr__(self, name):
        return getattr(self.instance, name)


    def _start_local_client(client = None, dash_add = ':20100'):
        try:
            if client:
                print('Restarting client')
                client.restart()
        except:
        #     cluster = LocalCluster(dashboard_address=':20100', memory_limit='4G')
            cluster = LocalCluster(dashboard_address=dash_add)
            print('Setting new client')
            client = Client(cluster)
            print(client)
        return client
