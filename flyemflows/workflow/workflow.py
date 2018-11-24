import os
import sys
import time
import socket
import getpass
import logging
import tempfile
import subprocess

import dask
from distributed import Client, LocalCluster

import neuclease
from neuclease.util import Timer

import confiddler.json as json
from dvid_resource_manager.server import DEFAULT_CONFIG as DEFAULT_RESOURCE_MANAGER_CONFIG

from ..util import kill_if_running, get_localhost_ip_address


logger = logging.getLogger(__name__)

# driver_ip_addr = '127.0.0.1'
driver_ip_addr = get_localhost_ip_address()

USER = getpass.getuser()

# defines workflows that work over DVID
class Workflow(object):
    """
    Base class for all Workflows.
    """
    LsfJobSchema = \
    {
        "description": "dask-jobqueue config settings for LSF jobs.\n"
                       "https://jobqueue.dask.org/en/latest/generated/dask_jobqueue.LSFCluster.html#dask-jobqueue-lsfcluster",
        "type": "object",
        "additionalProperties": True,
        "default": {},
        "properties": {
            "cores": {
                "description": "How many cores for each 'job', (typically an entire node's worth).",
                "type": "integer",
                "minimum": 1,
                "default": 16
            },
            "processes": {
                        "description": "How many processes per 'job'.\n"
                                       "https://jobqueue.dask.org/en/latest/configuration-setup.html#processes",
                        "type": "integer",
                        "minimum": 1,
                        "default": 1
            },
            
            "memory": {
                "description": "How much memory to allot to each 'job' (typically an entire node's worth, assuming the job reserved all CPUs).\n"
                               "Specified as a string with a suffix for units, e.g. 4GB",
                "type": "string",
                "default": "128GB"
            },
            "log-directory": {
                "description": "Where LSF worker logs (from stdout) will be stored.",
                "type": "string",
                "default": "worker-logs"
            },
            "local-directory": {
                "description": "Where dask should store temporary files when data spills to disk.\n"
                               "Note: Will also be used to configure Python's tempfile.tempdir",
                "type": "string",
                "default": ""
            },
            "walltime": {
                "description": "How much time to give the workers before killing them automatically.\n"
                               "Specified in HH:MM format.\n",
                "type": "string",
                "default": "24:00"
            },
            "death-timeout": {
                "description": "Seconds to wait for a scheduler before closing workers",
                "type": "integer",
                "default": 60
            }
        }
    }
    
    JobQueueSchema = \
    {
        "description": "dask-jobqueue config settings.",
        "type": "object",
        "additionalProperties": True,
        "default": {},
        "properties": {
            "lsf": LsfJobSchema
        }
    }
    
    DaskConfigSchema = \
    {
        "description": "Dask config values to override the defaults in ~/.config/dask/ or /etc/dask/.\n"
                       "See https://docs.dask.org/en/latest/configuration.html for details.",
        "type": "object",
        "additionalProperties": True,
        "default": {},
        "properties": {
            "jobqueue": JobQueueSchema
        }
    }

    ResourceManagerSchema = \
    {
        "type": "object",
        "default": {},
        "additionalProperties": False,
        "description": "Which resource manager server to use (if any) and how to configure it (if launching on the driver).",
        "properties": {
            "server": {
                "description": "If provided, workflows MAY use this resource server to coordinate competing requests from worker nodes. \n"
                               "Set to the IP address of the (already-running) resource server, or use the special word 'driver' \n"
                               "to automatically start a new resource server on the driver node.",
                "type": "string",
                "default": ""
            },
            "port": {
                "description": "Which port the resource server is running on.  (See description above.)",
                "type": "integer",
                "default": 0
            },
            "config": {
                "type": "object",
                "default": DEFAULT_RESOURCE_MANAGER_CONFIG,
                "additionalProperties": True
            }
        }
    }
    
    WorkerInitSchema = \
    {
        "type": "object",
        "default": {},
        "additionalProperties": False,
        "description": "The given script will be called once per worker node, before the workflow executes.",
        "properties": {
            "script-path": {
                "type": "string",
                "default": ""
            },
            "script-args": {
                "type": "array",
                "items": { "type": "string" },
                "default": []
            },
            "launch-delay": {
                "description": "By default, wait for the script to complete before continuing.\n"
                               "Otherwise, launch the script asynchronously and then pause for N seconds before continuing.",
                "type": "integer",
                "default": -1 # default: blocking execution
            },
            "log-dir": {
                "type": "string",
                "default": "/tmp"
            },
            "also-run-on-driver": {
                "description": "Also run this initialization script on the driver machine.\n",
                "type": "boolean",
                "default": False
            }
        }
    }

    BaseSchema = \
    {
        "type": "object",
        "description": "Workflow base config",
        "default": {},
        "additionalProperties": True,
        "required": ["workflow-name"],
        "properties": {
            "workflow-name": {
                "description": "The class name of the workflow which will be executed using this config.",
                "type": "string",
                "minLength": 1
            },
            "cluster-type": {
                "description": "Whether or not to use an LSF cluster or a local cluster.",
                "type": "string",
                "enum": ["lsf", "local-cluster", "synchronous"],
                "default": "local-cluster"
            },
            "dask-config": DaskConfigSchema,
            "resource-manager": ResourceManagerSchema,
            "worker-initialization": WorkerInitSchema
        }
    }
    
    @classmethod
    def schema(cls):
        raise NotImplementedError
    
    def __init__(self, config, num_workers):
        """Initialization of workflow object.

        Args:
            config (dict): loaded config data for workflow, as a dict
            schema (dict): json schema for workflow (already loaded as dict)
        """
        self.config = config
        neuclease.dvid.DEFAULT_APPNAME = self.config['workflow-name']
        self.num_workers = num_workers


    def execute(self):
        raise NotImplementedError

    
    def run(self):
        """
        Run the workflow by calling the subclass's execute() function
        (with some startup/shutdown steps before/after).
        """
        resource_server_proc = self._start_resource_server()
        self.cluster, self.client = self._init_dask()
        worker_init_pids, driver_init_pid = self._run_worker_initializations()
        
        with Timer(f"Running {self.config['workflow-name']}", logger):
            try:
                self.execute()
            finally:
                sys.stderr.flush()
                
                self._kill_initialization_procs(worker_init_pids, driver_init_pid)
                self._kill_resource_server(resource_server_proc)
    
                # Only the workflow calls cleanup_faulthandler, once all workers have exited
                # (All workers share the same output file for faulthandler.)
    
                # FIXME
                #cleanup_faulthandler()


    def _init_dask(self):
        # Consider using client.register_worker_callbacks() to configure
        # - faulthandler (later)
        # - excepthook?
        # - (okay, maybe it's just best to put that stuff in __init__.py, like in DSS)
        new_config = dask.config.update(dask.config.config, self.config['dask-config'])
        dask.config.set(new_config)

        cluster = None
        client = None

        if self.config["cluster-type"] == "lsf":
            from dask_jobqueue import LSFCluster #@UnresolvedImport

            local_dir = self.config["dask-config"]["jobqueue"]["lsf"]["local-directory"]
            if not local_dir:
                user = getpass.getuser()
                local_dir = f"/scratch/{user}"
                self.config["dask-config"]["jobqueue"]["lsf"]["local-directory"] = local_dir
                
                # Set tmp dir, too.
                tempfile.tempdir = local_dir
                os.environ['TMPDIR'] = local_dir # Forked processes will use this for tempfile.tempdir

            cluster = LSFCluster()
            cluster.scale(self.num_workers)
        elif self.config["cluster-type"] == "local-cluster":
            cluster = LocalCluster()
            cluster.scale(self.num_workers)
        elif self.config["cluster-type"] == "synchronous":
            # Synchronous mode is for testing and debugging only
            assert self.config['dask-config'].get('scheduler', 'synchronous') == 'synchronous'
            dask.config.set(scheduler='synchronous')
            class SynchronousClient:
                def ncores(self):
                    return {'driver': 1}
            client = SynchronousClient()
        else:
            assert False, "Unknown cluster type"

        if cluster:
            client = Client(cluster)

        return cluster, client


    def _start_resource_server(self):
        """
        Initialize the resource server config members and, if necessary,
        start the resource server process on the driver node.
        
        If the resource server is started locally, the "resource-server"
        setting is OVERWRITTEN in the config data with the driver IP.
        
        Returns:
            The resource server Popen object (if started), or None
        """
        server = self.config["resource-manager"]["server"]
        port = self.config["resource-manager"]["port"]

        if server == "":
            return None
        
        if port == 0:
            msg = f"You specified a resource server ({server}), but no port"
            raise RuntimeError(msg)
        
        if server != "driver":
            if self.config["resource-manager"]["config"]:
                msg = ("The resource manager config should only be specified when resource manager 'server' is set to 'driver'."
                       "(If the resource manager server is already running on a different machine, configure it there.)")
                raise RuntimeError(msg)
            return None

        if self.config["resource-manager"]["config"]:
            tmpdir = f"/tmp/{USER}"
            os.makedirs(tmpdir, exist_ok=True)
            server_config_path = f'{tmpdir}/driver-resource-server-config.json'
            with open(server_config_path, 'w') as f:
                json.dump(self.config["resource-manager"]["config"], f)
            config_arg = f'--config-file={server_config_path}'
        else:
            config_arg = ''
        
        # Overwrite workflow config data so workers see our IP address.
        self.config["resource-manager"]["server"] = server = driver_ip_addr

        logger.info(f"Starting resource manager on the driver ({driver_ip_addr})")
        
        python = sys.executable
        cmd = f"{python} {sys.prefix}/bin/dvid_resource_manager {port} {config_arg}"
        resource_server_process = subprocess.Popen(cmd, stderr=subprocess.STDOUT, shell=True)
        logger.info("Started resource manager")

        return resource_server_process


    def _kill_resource_server(self, resource_server_proc):
        if resource_server_proc:
            logger.info(f"Terminating resource manager (PID {resource_server_proc.pid})")
            resource_server_proc.terminate()
            try:
                resource_server_proc.wait(10.0)
            except subprocess.TimeoutExpired:
                kill_if_running(resource_server_proc.pid, 10.0)


    def run_on_each_worker(self, func, once_per_machine=True):
        """
        Run the given function once per worker (or once per worker machine).
        
        Args:
            func:
                Must be picklable.
            
            once_per_machine:
                Ensure that the function is only run once per machine,
                even if your cluster is configured to run more than one
                worker on each node.
        """
        if self.config["cluster-type"] == "synchronous":
            results = {'driver': func()}
            logger.info(f"Ran {func.__name__} on the driver only")
            return results
        
        all_worker_hostnames = self.client.run(socket.gethostname)
        if not once_per_machine:
            worker_hostnames = all_worker_hostnames

        if once_per_machine:
            machines = set()
            worker_hostnames = {}
            for address, name in worker_hostnames.items():
                ip = address.split('://')[1].split(':')[1]
                if ip not in machines:
                    machines.add(ip)
                    worker_hostnames[address] = name
        
        workers = list(worker_hostnames.keys())
        hostnames = list(worker_hostnames.values())
        results = self.client.run(func, workers=workers)
        logger.info(f"Ran {func.__name__} on {len(hostnames)} nodes: {hostnames}")
        return results

    
    def _run_worker_initializations(self):
        """
        Run an initialization script (e.g. a bash script) on each worker node.
        Returns:
            (worker_init_pids, driver_init_pid), where worker_init_pids is a
            dict of { hostname : PID } containing the PIDs of the init process
            IDs running on the workers.
        """
        from subprocess import STDOUT
        from os.path import basename, splitext

        try:
            init_options = self.config["worker-initialization"]
        except KeyError:
            # old workflows haven't been updated to inherit from the base workflow schema
            return ({}, None)
            
        if not init_options["script-path"]:
            return ({}, None)

        init_options["script-path"] = self.relpath_to_abspath(init_options["script-path"])
        init_options["log-dir"] = self.relpath_to_abspath(init_options["log-dir"])
        os.makedirs(init_options["log-dir"], exist_ok=True)
        
        def launch_init_script():
            script_name = splitext(basename(init_options["script-path"]))[0]
            log_dir = init_options["log-dir"]
            hostname = socket.gethostname()
            log_file = open(f'{log_dir}/{script_name}-{hostname}.log', 'w')

            try:
                script_args = [str(a) for a in init_options["script-args"]]
                p = subprocess.Popen( [init_options["script-path"], *script_args], stdout=log_file, stderr=STDOUT )
            except OSError as ex:
                if ex.errno == 8: # Exec format error
                    raise RuntimeError("OSError: [Errno 8] Exec format error\n"
                                       "Make sure your script begins with a shebang line, e.g. !#/bin/bash")
                raise

            if init_options["launch-delay"] == -1:
                p.wait()
                if p.returncode == 126:
                    raise RuntimeError("Permission Error: Worker initialization script is not executable: {}"
                                       .format(init_options["script-path"]))
                assert p.returncode == 0, \
                    "Worker initialization script ({}) failed with exit code: {}"\
                    .format(init_options["script-path"], p.returncode)
                return None

            time.sleep(init_options["launch-delay"])
            return p.pid
        
        worker_init_pids = self.run_on_each_worker(launch_init_script)

        driver_init_pid = None
        if init_options["also-run-on-driver"]:
            driver_init_pid = launch_init_script()
        
        return (worker_init_pids, driver_init_pid)


    def _kill_initialization_procs(self, worker_init_pids, driver_init_pid):
        """
        Kill any initialization processes (as launched from _run_worker_initializations)
        that might still running on the workers and/or the driver.
        
        If they don't respond to SIGTERM, they'll be force-killed with SIGKILL after 10 seconds.
        """
        def kill_init_proc():
            try:
                pid_to_kill = worker_init_pids[socket.gethostname()]
            except KeyError:
                return
            else:
                kill_if_running(pid_to_kill, 10.0)
        
        if any(worker_init_pids.values()):
            self.run_on_each_worker(kill_init_proc)
        else:
            logger.info("No worker init processes to kill")
            

        if driver_init_pid:
            kill_if_running(driver_init_pid, 10.0)
        else:
            logger.info("No driver init process to kill")

