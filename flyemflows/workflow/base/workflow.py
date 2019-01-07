import os
import time
import socket
import getpass
import logging
import tempfile
from collections import defaultdict

import dask
from distributed import Client, LocalCluster
from distributed.utils import parse_bytes

import neuclease
from neuclease.util import Timer

from confiddler import convert_to_base_types

from ...util import extract_ip_from_link, construct_ganglia_link
from ...util.lsf import construct_rtm_url, get_job_submit_time

from .base_schema import BaseSchema
from .contexts import environment_context, LocalResourceManager, WorkerDaemons

logger = logging.getLogger(__name__)

# defines workflows that work over DVID
class Workflow(object):
    """
    Base class for all Workflows.

    TODO:
    - Possibly produce profiles of driver functions

    """
    
    @classmethod
    def schema(cls):
        if cls is Workflow:
            # The Workflow class itself is sometimes "executed" during unit tests,
            # to test generic workflow features (such as worker initialization)
            return BaseSchema
        else:
            # Subclasses must implement schema() themselves.
            raise NotImplementedError
    
    def __init__(self, config, num_workers):
        """Initialization of workflow object.

        Args:
            config (dict): loaded config data for workflow, as a dict
            num_workers: How many workers to launch for the job.
                         Note that this is not necessarily the same as the number of nodes (machines),
                         depending on the dask config.
        """
        self.config = config
        neuclease.dvid.DEFAULT_APPNAME = self.config['workflow-name']
        self.num_workers = num_workers
        
        # Initialized in run()
        self.cluster = None
        self.client = None


    def __del__(self):
        # If the cluster is still alive (a debugging feature),
        # kill it now.
        self._cleanup_dask()


    def execute(self):
        if type(self) is Workflow:
            # The Workflow class itself is sometimes "executed" during unit tests,
            # to test generic workflow features (such as worker initialization)
            pass
        else:
            # Subclasses must implement execute() themselves.
            raise NotImplementedError

    
    def run(self, kill_cluster=True):
        """
        Run the workflow by calling the subclass's execute() function
        (with some startup/shutdown steps before/after).
        """
        logger.info(f"Working dir: {os.getcwd()}")
        with Timer(f"Running {self.config['workflow-name']}", logger), \
             environment_context(self.config["environment-variables"]), \
             LocalResourceManager(self.config["resource-manager"]):
                self._write_driver_graph_urls()
                
                # See also: reinitialize_cluster()
                self._init_dask()
                
                try:
                    with WorkerDaemons(self):
                        self.execute()
                finally:
                    # See also: reinitialize_cluster()
                    if kill_cluster:
                        self._cleanup_dask()


    def total_cores(self):
        return sum( self.client.ncores().values() )


    def reinitialize_cluster(self):
        """
        Discard the current cluster (and client) and start up a new one.
        
        Workflows can call this in between 'batches' of work
        to get a new lease on cluster nodes from LSF.
        
        Note: Don't do this while there are tasks in-flight.
        """
        logger.info("Shutting down old cluster")
        self._kill_initialization_procs()
        self._cleanup_dask()
        
        logger.info("Initializing new cluster")
        self._init_dask()
        self._run_worker_initializations()


    def _write_driver_graph_urls(self):
        try:
            driver_jobid = os.environ['LSB_JOBID']
        except KeyError:
            pass
        else:
            driver_rtm_url = construct_rtm_url(driver_jobid)
            driver_host = socket.gethostname()
            logger.info(f"Driver host is: {driver_host}")
            logger.info(f"Driver RTM graphs: {driver_rtm_url}")

            start_timestamp = get_job_submit_time()
            ganglia_url = construct_ganglia_link(driver_host, start_timestamp)

            hostgraph_url_path = 'graph-links.txt'
            with open(hostgraph_url_path, 'a') as f:
                f.write(f"{socket.gethostname()} (driver)\n")
                f.write(f"  {ganglia_url}\n")
                f.write(f"  {driver_rtm_url}\n")
            

    def _init_dask(self, wait_for_workers=True):
        # Consider using client.register_worker_callbacks() to configure
        # - faulthandler (later)
        # - excepthook?
        # - (okay, maybe it's just best to put that stuff in __init__.py, like in DSS)
        user_config = convert_to_base_types(self.config['dask-config'])
        new_config = dask.config.update(dask.config.config, user_config)
        dask.config.set(new_config)

        if self.config["cluster-type"] == "lsf":
            from dask_jobqueue import LSFCluster #@UnresolvedImport

            ncpus = self.config["dask-config"]["jobqueue"]["lsf"]["ncpus"]
            if ncpus == -1:
                ncpus = self.config["dask-config"]["jobqueue"]["lsf"]["cores"]
                self.config["dask-config"]["jobqueue"]["lsf"]["ncpus"] = ncpus

            mem = self.config["dask-config"]["jobqueue"]["lsf"]["mem"]
            if not mem:
                memory = self.config["dask-config"]["jobqueue"]["lsf"]["memory"]
                mem = parse_bytes(memory)
                self.config["dask-config"]["jobqueue"]["lsf"]["mem"] = mem

            local_dir = self.config["dask-config"]["jobqueue"]["lsf"]["local-directory"]
            if not local_dir:
                user = getpass.getuser()
                local_dir = f"/scratch/{user}"
                self.config["dask-config"]["jobqueue"]["lsf"]["local-directory"] = local_dir
                
                # Set tmp dir, too.
                tempfile.tempdir = local_dir
                os.environ['TMPDIR'] = local_dir # Forked processes will use this for tempfile.tempdir

            # Reconfigure
            user_config = convert_to_base_types(self.config['dask-config'])
            new_config = dask.config.update(dask.config.config, user_config)
            dask.config.set(new_config)

            self.cluster = LSFCluster(ip='0.0.0.0')
            self.cluster.scale(self.num_workers)
        elif self.config["cluster-type"] == "local-cluster":
            self.cluster = LocalCluster(ip='0.0.0.0')
            self.cluster.scale(self.num_workers)
        elif self.config["cluster-type"] in ("synchronous", "processes"):
            cluster_type = self.config["cluster-type"]

            # synchronous/processes mode is for testing and debugging only
            assert self.config['dask-config'].get('scheduler', cluster_type) == cluster_type, \
                "Inconsistency between the dask-config and the scheduler you chose."

            if cluster_type == "synchronous":
                ncores = 1
            else:
                import multiprocessing
                ncores = multiprocessing.cpu_count()
            
            dask.config.set(scheduler=self.config["cluster-type"])
            class DebugClient:
                def ncores(self):
                    return {'driver': ncores}
                def close(self):
                    pass
                def scatter(self, data, *_):
                    return data
            self.client = DebugClient()
        else:
            assert False, "Unknown cluster type"

        if self.cluster:
            dashboard = self.cluster.dashboard_link
            logger.info(f"Dashboard running on {dashboard}")
            dashboard_ip = extract_ip_from_link(dashboard)
            dashboard = dashboard.replace(dashboard_ip, socket.gethostname())
            logger.info(f"              a.k.a. {dashboard}")
            
            self.client = Client(self.cluster, timeout='60s') # Note: Overrides config value: distributed.comm.timeouts.connect

            # Wait for the workers to spin up.
            with Timer(f"Waiting for {self.num_workers} workers to launch", logger):
                while ( wait_for_workers
                        and self.client.status == "running"
                        and len(self.cluster.scheduler.workers) < self.num_workers ):
                    time.sleep(0.1)

            if wait_for_workers and self.config["cluster-type"] == "lsf":
                self._write_worker_graph_urls('graph-links.txt')


    def _write_worker_graph_urls(self, graph_url_path):
        """
        Write (or append to) the file containing links to the Ganglia and RTM
        hostgraphs for the workers in our cluster.
        
        We emit the following URLs:
            - One Ganglia URL for the combined graphs of all workers
            - One Ganglia URL for each worker
            - One RTM URL for each job (grouped by host)
        """
        assert self.config["cluster-type"] == "lsf"
        job_submit_times = self.run_on_each_worker(get_job_submit_time, True, True)

        host_min_submit_times = {}
        for addr, timestamp in job_submit_times.items():
            host = addr[len('tcp://'):].split(':')[0]
            try:
                min_timestamp = host_min_submit_times[host]
                if timestamp < min_timestamp:
                    host_min_submit_times[host] = timestamp
            except KeyError:
                host_min_submit_times[host] = timestamp

        host_ganglia_links = { host: construct_ganglia_link(host, ts) for host,ts in host_min_submit_times.items() }

        all_hosts = list(host_min_submit_times.keys())
        min_timestamp = min(host_min_submit_times.values())
        combined_ganglia_link = construct_ganglia_link(all_hosts, min_timestamp)
        
        hostgraph_urls = self.run_on_each_worker(construct_rtm_url, False, True)

        # Some workers share the same parent LSF job,
        # and hence have the same hostgraph URL.
        # Don't show duplicate links, but do group the links by host
        # and indicate how many workers are hosted on each node.
        host_url_counts = defaultdict(lambda: defaultdict(lambda: 0))
        for addr, url in hostgraph_urls.items():
            host = addr[len('tcp://'):].split(':')[0]
            host_url_counts[host][url] += 1
        
        with open(graph_url_path, 'a') as f:
            f.write('-'*100 + '\n')
            f.write("Combined Ganglia Graphs:\n")
            f.write(f"  {combined_ganglia_link}\n")
            
            for host, url_counts in host_url_counts.items():
                total_workers = sum(url_counts.values())
                f.write(f"{host} ({total_workers} workers)\n")
                f.write(f"  {host_ganglia_links[host]}\n")
                for url in url_counts.keys():
                    f.write(f"  {url}\n")


    def _cleanup_dask(self):
        if self.client:
            self.client.close()
            self.client = None
        if self.cluster:
            self.cluster.close()
            self.cluster = None


    def run_on_each_worker(self, func, once_per_machine=False, return_hostnames=True):
        """
        Run the given function once per worker (or once per worker machine).
        Results are returned in a dict of { worker: result }
        
        Args:
            func:
                Must be picklable.
            
            once_per_machine:
                Ensure that the function is only run once per machine,
                even if your cluster is configured to run more than one
                worker on each node.
            
            return_hostnames:
                If True, result keys use hostnames instead of IPs.
        Returns:
            dict:
            { 'ip:port' : result } OR
            { 'hostname:port' : result }
        """
        if self.config["cluster-type"] in ("synchronous", "processes"):
            if return_hostnames:
                results = {f'tcp://{socket.gethostname()}': func()}
            else:
                results = {'tcp://127.0.0.1': func()}
            logger.info(f"Ran {func.__name__} on the driver only")
            return results
        
        all_worker_hostnames = self.client.run(socket.gethostname)
        if not once_per_machine:
            worker_hostnames = all_worker_hostnames

        if once_per_machine:
            machines = set()
            worker_hostnames = {}
            for address, name in all_worker_hostnames.items():
                ip = address.split('://')[1].split(':')[0]
                if ip not in machines:
                    machines.add(ip)
                    worker_hostnames[address] = name
        
        workers = list(worker_hostnames.keys())
        
        with Timer(f"Running {func.__name__} on {len(workers)} workers", logger):
            results = self.client.run(func, workers=workers)
        
        if not return_hostnames:
            return results
    
        final_results = {}
        for address, result in results.items():
            hostname = worker_hostnames[address]
            ip = extract_ip_from_link(address)
            final_results[address.replace(ip, hostname)] = result

        return final_results


