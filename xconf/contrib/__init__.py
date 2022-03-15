from typing import Optional, Union
from urllib.parse import urlparse
import threading
import logging
log = logging.getLogger(__name__)

_LOCAL = threading.local()
_LOCAL.filesystems = {}

from .. import field, config, Command

@config
class CommonRayConfig:
    env_vars : Optional[dict[str, str]] = field(default_factory=dict, help="Environment variables to set for worker processes")

    def _get_ray_init_kwargs(self):
        env_vars = {
            'RAY_USER_SETUP_FUNCTION': 'xpipeline.commands.vapp_trap.init_worker',
            'MKL_NUM_THREADS': '1',
            'OMP_NUM_THREADS': '1',
            'NUMBA_NUM_THREADS': '1',
        }
        env_vars.update(self.env_vars)
        ray_init_kwargs = {'runtime_env': {'env_vars': env_vars,}}
        return ray_init_kwargs

    def init(self):
        import ray
        ray_init_kwargs = self._get_ray_init_kwargs()
        ray.init(**ray_init_kwargs)

@config
class LocalRayConfig(CommonRayConfig):
    cpus : Optional[int] = field(default=None, help="CPUs available to built-in Ray cluster (default is auto-detected)")
    gpus : Optional[int] = field(default=None, help="GPUs available to built-in Ray cluster (default is auto-detected)")
    resources : Optional[dict[str, float]] = field(default_factory=dict, help="Node resources available when running in standalone mode")

    def _get_ray_init_kwargs(self):
        ray_init_kwargs = super()._get_ray_init_kwargs()
        ray_init_kwargs['num_cpus'] = self.cpus
        ray_init_kwargs['num_gpus'] = self.gpus
        ray_init_kwargs['resources'] = self.resources
        return ray_init_kwargs


@config
class RemoteRayConfig(CommonRayConfig):
    url : str = field(help="URL to existing Ray cluster head node")
    def _get_ray_init_kwargs(self):
        ray_init_kwargs = super()._get_ray_init_kwargs()
        ray_init_kwargs['address'] = self.url
        return ray_init_kwargs

def get_fs(path):
    '''Obtain a concrete fsspec filesystem from a path using
    the protocol string (if any; defaults to 'file:///') and
    `_get_kwargs_from_urls` on the associated fsspec class. The same
    instance will be returned if the same kwargs are used multiple times
    in the same thread.

    Note: Won't work when kwargs are required but not encoded in the
    URL.
    '''
    import fsspec
    scheme = urlparse(path).scheme
    proto = scheme if scheme != '' else 'file'
    cls = fsspec.get_filesystem_class(proto)
    if hasattr(cls, '_get_kwargs_from_urls'):
        kwargs = cls._get_kwargs_from_urls(path)
    else:
        kwargs = {}
    key = (proto,) + tuple(kwargs.items())
    if not hasattr(_LOCAL, 'filesystems'):
        _LOCAL.filesystems = {}   # unclear why this is not init at import in dask workers
    if key not in _LOCAL.filesystems:
        fs = cls(**kwargs)
        _LOCAL.filesystems[key] = fs
    return _LOCAL.filesystems[key]


@config
class PathConfig:
    path : str = field(help="File path")

    def get_fs(self):
        return get_fs(self.path)

@config
class FileConfig(PathConfig):

    def open(self, mode='r'):
        fs = get_fs(self.path)
        return fs.open(self.path, mode)

TIME_COLNAME = 'time_total_sec'
INDEX_COLNAME = 'index'

@config
class BaseRayGrid(Command):
    destination : PathConfig = field(default=".", help="Directory for output files")
    output_filename : str = field(default="grid.fits", help="Output filename to write")
    output_extname : str = field(default="grid", help="Output table extension name")
    ray : Union[RemoteRayConfig, LocalRayConfig] = field(
        default=LocalRayConfig(),
        help="Ray distributed framework configuration"
    )
    only_indices : list[int] = field(default_factory=list, help="List of grid indices to process")
    dry_run : bool = field(default=False, help="Output the grid points that would be processed without processing them")
    recompute : bool = field(default=False, help="Whether to ignore existing checkpoints and recompute everything")
    shutdown_on_completion : bool = field(default=True, help="Whether ray.shutdown() is called to clean up the cluster after completing the grid")
    hide_progress_bar : bool = field(default=None, help="Whether to hide the progress bar, default chooses to hide only when stdout is not a tty")
    checkpoint_every_x : int = field(default=10, help="How often to write a grid checkpoint")

    def main(self):
        tbl = self.load_checkpoint_or_generate()
        pending_tbl = self.filter_grid(tbl)
        refs = self.launch_grid(pending_tbl)
        self.process_grid(tbl, refs)

    def load_checkpoint_or_generate(self):
        from astropy.io import fits
        empty_grid_tbl = self.generate_grid()
        output_path = f"{self.destination}/{self.output_filename}"
        destfs = self.destination.get_fs()
        if not destfs.exists(output_path):
            log.debug(f"No checkpoint at {output_path}, using generated grid")
            return empty_grid_tbl
        with destfs.open(self.output_filename) as fh:
            tbl = fits.open(fh)[self.output_extname].data
        log.debug(f"Loaded checkpoint from {output_path}[{self.output_extname}]")
        if not self.compare_grid_to_checkpoint(tbl, empty_grid_tbl):
            raise RuntimeError(f"Grid parameters changed but checkpoint file exists, aborting")
        return tbl

    def save(self, tbl):
        from astropy.io import fits
        with self.destination.get_fs().open(self.output_filename, 'wb') as fh:
            fits.HDUList([
                fits.PrimaryHDU(),
                fits.BinTableHDU(tbl, name=self.output_extname),
            ]).writeto(fh, overwrite=True)
        log.debug(f"Saved to {self.destination}/{self.output_filename}")
        return tbl

    def filter_grid(self, tbl):
        import numpy as np
        if len(self.only_indices):
            mask = np.zeros(len(tbl), dtype=bool)
            for idx in self.only_indices:
                mask |= tbl[INDEX_COLNAME] == idx
        mask &= tbl[TIME_COLNAME] == 0
        return tbl[mask]

    def process_grid(self, tbl, pending_refs):
        import ray
        import numpy as np
        from tqdm import tqdm
        # Wait for results, checkpointing as we go
        pending = pending_refs
        total = len(tbl)
        restored_from_checkpoint = np.count_nonzero(tbl[TIME_COLNAME] != 0)

        with tqdm(total=total, disable=self.hide_progress_bar, unit='points') as pbar:
            pbar.update(restored_from_checkpoint)
            while pending:
                complete, pending = ray.wait(pending, timeout=5, num_returns=min(self.checkpoint_every_x, len(pending)))
                for result in ray.get(complete):
                    idx = result[INDEX_COLNAME]
                    tbl[idx] = result
                if len(complete):
                    pbar.update(len(complete))
                    if self.checkpoint is not None:
                        self.save(tbl)
        self.save(tbl)

    def compare_grid_to_checkpoint(self, checkpoint_tbl, grid_tbl):
        raise NotImplementedError("Subclasses must implement compare_grid_to_checkpoint()")

    def generate_grid(self):
        raise NotImplementedError("Subclasses must implement generate_grid()")

    def launch_grid(self, pending_tbl) -> list:
        '''Launch Ray tasks for each grid point and collect object
        refs. The Ray remote function ref must return a copy of the
        grid row it's called with, updating 'time_total_sec' to
        indicate it's been processed.
        '''
        raise NotImplementedError("Subclasses must implement launch_grid()")
