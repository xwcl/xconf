import fsspec
import enum
from typing import Optional, Union
import typing
import logging

log = logging.getLogger(__name__)
# also re-exported as backwards compatibility aliases
from .. import field, config, Command
from ..fs import join, get_fs

@config
class PathConfig:
    path: str = field(help="File path")

    def get_fs(self) -> fsspec.AbstractFileSystem:
        return get_fs(self.path)

    def exists(self):
        return self.get_fs().exists(self.path)


@config
class DirectoryConfig(PathConfig):
    def exists(self, path=None):
        '''Return whether this directory (or, optionally, the path provided, rooted
        at this directory) exists'''
        if path is not None:
            test_path = self.join(path)
        else:
            test_path = self.path
        return self.get_fs().exists(test_path)

    def join(self, path):
        return join(self.path, path)

    def ensure_exists(self):
        destfs = self.get_fs()
        destfs.makedirs(self.path, exist_ok=True)

    def open_path(self, path, mode="rb"):
        return self.get_fs().open(join(self.path, path), mode=mode)


@config
class FileConfig(PathConfig):
    def open(self, mode="rb") -> fsspec.core.OpenFile:
        fs = get_fs(self.path)
        return fs.open(self.path, mode)

@config
class CommonRayConfig:
    setup_function_path: typing.ClassVar[typing.Optional[str]] = None
    env_vars: Optional[dict[str, str]] = field(
        default_factory=dict, help="Environment variables to set for worker processes"
    )
    threads_per_worker : int = field(
        default=1,
        help="Max. number of threads to use per worker (passed in OMP_NUM_THREADS, MKL_NUM_THREADS, and NUMBA_NUM_THREADS)"
    )

    def _get_ray_init_kwargs(self):
        num_threads = str(self.threads_per_worker)
        env_vars = {
            "MKL_NUM_THREADS": num_threads,
            "OMP_NUM_THREADS": num_threads,
            "NUMBA_NUM_THREADS": num_threads,
        }
        if self.setup_function_path is not None:
            env_vars["RAY_USER_SETUP_FUNCTION"] = self.setup_function_path
        env_vars.update(self.env_vars)
        ray_init_kwargs = {
            "runtime_env": {
                "env_vars": env_vars,
            }
        }
        return ray_init_kwargs

    def init(self):
        import ray

        ray_init_kwargs = self._get_ray_init_kwargs()
        ray.init(**ray_init_kwargs)


@config
class LocalRayConfig(CommonRayConfig):
    cpus: Optional[int] = field(
        default=None,
        help="CPUs available to built-in Ray cluster (default is auto-detected)",
    )
    gpus: Optional[int] = field(
        default=None,
        help="GPUs available to built-in Ray cluster (default is auto-detected)",
    )
    resources: Optional[dict[str, float]] = field(
        default_factory=dict,
        help="Node resources available when running in standalone mode",
    )

    def _get_ray_init_kwargs(self):
        ray_init_kwargs = super()._get_ray_init_kwargs()
        ray_init_kwargs["num_cpus"] = self.cpus
        ray_init_kwargs["num_gpus"] = self.gpus
        ray_init_kwargs["resources"] = self.resources
        return ray_init_kwargs


@config
class RemoteRayConfig(CommonRayConfig):
    url: str = field(help="URL to existing Ray cluster head node")

    def _get_ray_init_kwargs(self):
        ray_init_kwargs = super()._get_ray_init_kwargs()
        ray_init_kwargs["address"] = self.url
        return ray_init_kwargs


AnyRayConfig = Union[LocalRayConfig, RemoteRayConfig]

TIME_COLNAME = "time_total_sec"
INDEX_COLNAME = "index"


class GridFormat(enum.Enum):
    FITS = "fits"
    JSON = "json"


@config
class BaseRayGrid(Command):
    destination: DirectoryConfig = field(
        default_factory=lambda: DirectoryConfig(path="."), help="Directory for output files"
    )
    output_filename: str = field(default="grid", help="Output filename to write, sans extension")
    output_extname: str = field(default="grid", help="Output table extension name")
    format: GridFormat = field(
        default=GridFormat.FITS, help="Format for grid point output"
    )
    skip_check: bool = field(default=False, help="When an existing grid is provided, ignore the grid generation parameters and execute this computation for all remaining non-evaluated points in the existing file")
    point_filename_format: str = field(default="point_{:04}", help="Output filename for individual point, sans extension")
    ray: Union[RemoteRayConfig, LocalRayConfig] = field(
        default_factory=lambda: LocalRayConfig(), help="Ray distributed framework configuration"
    )
    only_indices: list[int] = field(
        default_factory=list, help="List of grid indices to process"
    )
    dry_run: bool = field(
        default=False,
        help="Output the grid points that would be processed without processing them",
    )
    recompute: bool = field(
        default=False,
        help="Whether to ignore existing checkpoints and recompute everything requested",
    )
    shutdown_on_completion: bool = field(
        default=True,
        help="Whether ray.shutdown() is called to clean up the cluster after completing the grid",
    )
    hide_progress_bar: bool = field(
        default=None,
        help="Whether to hide the progress bar, default chooses to hide only when stdout is not a tty",
    )
    checkpoint_every_x: int = field(
        default=10,
        help="How many results to collect before writing a grid checkpoint (approximately), set 0 to disable",
    )

    def main(self):
        self.ray.init()
        tbl = self.load_checkpoint_or_generate()
        if TIME_COLNAME not in tbl.dtype.fields:
            raise RuntimeError(f"Grid table doesn't include {TIME_COLNAME} column")
        if INDEX_COLNAME not in tbl.dtype.fields:
            raise RuntimeError(f"Grid table doesn't include {INDEX_COLNAME} column")
        pending_tbl = self.filter_grid(tbl)
        log.debug(f"{len(pending_tbl)} points to process")
        refs = self.launch_grid(pending_tbl)
        log.debug(f"Got {len(refs)} pending refs")
        self.process_grid(tbl, refs)

    def _get_output_path(self):
        return join(self.destination.path, self.output_filename + "." + self.format.value)

    def load_checkpoint_or_generate(self, *args):
        empty_grid_tbl = self.generate_grid(*args)
        if self.format is GridFormat.FITS:
            output_path = self._get_output_path()
            destfs = self.destination.get_fs()
            if not destfs.exists(output_path):
                log.debug(f"No checkpoint at {output_path}, using generated grid")
                return empty_grid_tbl
            with destfs.open(output_path) as fh:
                    from astropy.io import fits
                    import numpy as np
                    tbl = (
                        np.asarray(fits.open(fh)[self.output_extname].data)
                        .byteswap()
                        .newbyteorder()
                    )
                    log.debug(f"Loaded checkpoint from {output_path}[{self.output_extname}]")
        elif self.format is GridFormat.JSON:
            log.debug(f"Skipping checkpoint loading for JSON format")
            tbl = empty_grid_tbl
        else:
            raise ValueError(f"Unknown format value {self.format}")
        if not self.skip_check and not self.compare_grid_to_checkpoint(tbl, empty_grid_tbl):
            raise RuntimeError(
                f"Grid parameters changed but checkpoint file exists, aborting"
            )
        return tbl

    def _save_fits(self, tbl, output_path):
        from astropy.io import fits
        self.destination.ensure_exists()
        with self.destination.get_fs().open(output_path, "wb") as fh:
            fits.HDUList(
                [
                    fits.PrimaryHDU(),
                    fits.BinTableHDU(tbl, name=self.output_extname),
                ]
            ).writeto(fh, overwrite=True)
        pass

    def _save_json(self, tbl):
        import orjson
        dest_fs = self.destination.get_fs()
        completed_points = tbl[tbl[TIME_COLNAME] > 0]
        for row in completed_points:
            point_dict = {}
            for key in row.dtype.names:
                point_dict[key] = row[key]
            output_path = join(self.destination.path, self.point_filename_format.format(row[INDEX_COLNAME]))
            with dest_fs.open(output_path, "wb") as fh:
                fh.write(orjson.dumps(
                    point_dict,
                    option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY,
                ))
            log.debug(f"Wrote to {output_path}: {point_dict}")

    def save(self, tbl):
        output_path = self._get_output_path()
        if self.format is GridFormat.JSON:
            self._save_json(tbl)
        elif self.format is GridFormat.FITS:
            self._save_fits(tbl, output_path)
        else:
            raise ValueError(f"Unknown format value {self.format}")
        log.debug(f"Saved to {output_path}")
        return tbl

    def filter_grid(self, tbl):
        import numpy as np

        mask = np.ones(len(tbl), dtype=bool)
        if len(self.only_indices):
            indices_mask = np.zeros(len(tbl), dtype=bool)
            for idx in self.only_indices:
                indices_mask |= tbl[INDEX_COLNAME] == idx
            mask &= indices_mask
        if not self.recompute:
            mask &= tbl[TIME_COLNAME] == 0
        return tbl[mask]

    def process_grid(self, tbl, pending_refs):
        if len(pending_refs) == 0:
            log.info("All requested grid points already computed")
            return
        import ray
        import numpy as np
        from tqdm import tqdm

        # Wait for results, checkpointing as we go
        pending = pending_refs
        total = len(tbl)
        n_submitted = len(pending_refs)
        n_completed = np.count_nonzero(tbl[TIME_COLNAME] != 0)

        with tqdm(total=total, disable=self.hide_progress_bar, unit="points") as pbar:
            pbar.update(n_completed)
            while pending:
                complete, pending = ray.wait(
                    pending,
                    timeout=5,
                    num_returns=min(
                        self.checkpoint_every_x if self.checkpoint_every_x > 0 else 1,
                        len(pending),
                    ),
                )
                results_retired = 0
                for result in ray.get(complete):
                    idx = result[INDEX_COLNAME]
                    tbl[idx] = result
                    if len(result.shape) > 0:
                        results_retired += result.shape[0]
                    else:
                        results_retired += 1
                if len(complete):
                    pbar.update(results_retired)
                    n_completed += results_retired
                    if self.checkpoint_every_x > 0:
                        self.save(tbl)
                        log.debug(f"Saved {n_completed} of {total} ({n_submitted} submitted)")
        self.save(tbl)

    def compare_grid_to_checkpoint(self, checkpoint_tbl, grid_tbl) -> bool:
        raise NotImplementedError(
            "Subclasses must implement compare_grid_to_checkpoint()"
        )

    def generate_grid(self):
        raise NotImplementedError("Subclasses must implement generate_grid()")

    def launch_grid(self, pending_tbl) -> list:
        """Launch Ray tasks for each grid point and collect object
        refs. The Ray remote function ref must return a copy of the
        grid row it's called with, updating 'time_total_sec' to
        indicate it's been processed.
        """
        raise NotImplementedError("Subclasses must implement launch_grid()")

