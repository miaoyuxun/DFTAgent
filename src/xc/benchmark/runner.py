"""Benchmark runner: evaluate XCFunctional against GMTKN55 subsets.

Supports molecule-level parallelism via ProcessPoolExecutor, and can
optionally submit whole-subset evaluations to Slurm via ``srun``.
"""

from __future__ import annotations

import os
import json
import time
import sys
import tempfile
import logging
import math
import subprocess
import atexit
import hashlib
import queue
import re
import signal
import threading
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np

from ..xc_functional import XCFunctional
from ..slurm_env import clean_slurm_submit_env
from .result import BenchmarkResult, SubsetResult
from .gmtkn55 import parse_res_file, load_molecule_geometry

logger = logging.getLogger(__name__)

SEARCH_SUBSETS = ["SIE4x4", "BH76", "G21IP", "G21EA"]
VALIDATION_SUBSETS = ["W4-11", "S66", "ISO34", "S22", "G2RC"]

HARTREE_TO_KCAL = 627.509474


@dataclass
class MoleculeEvalResult:
    ok: bool
    energy: float | None = None
    error: str = ""
    walltime_s: float = 0.0
    started_at: float = 0.0
    ended_at: float = 0.0


@dataclass
class SlurmLaunchConfig:
    enabled: bool = False
    launcher: str = "srun"
    nodes: int = 1
    cpus_per_task: int = 8
    memory: str = "10G"
    partition: str = "dcu_htc"
    output_dir: Path | None = None
    extra_args: list[str] | None = None



@dataclass
class SlurmDockerQueueConfig:
    """Options for the Slurm+Docker file-queue benchmark backend."""

    image: str = ""
    container_repo_root: str = "/workspace/DFTAgent"
    time_limit: str = "24:00:00"
    total_cpus: int = 128
    cpus_per_worker: int = 32
    worker_count: int = 0
    nodes: int = 0
    ntasks_per_node: int = 0
    cpus_per_node: int = 64
    submit_mode: str = "allocation"
    array_max_running: int = 0
    queue_dispatch_mode: str = "candidate_dedicated"
    thread_policy: str = "atom_adaptive"
    atom_thread_bins: str = "2:1,6:2,10:4,999:8"
    max_threads_per_sample: int = 8
    molecules_per_task: int = 1
    output_dir: Path | None = None
    docker_extra_args: list[str] | None = None

class SlurmAllocation:
    """Hold one program-lifetime Slurm allocation and submit work into it."""

    _READY_RE = re.compile(r"DFTAGENT_SALLOC_READY:(\d+)")
    _GRANTED_RE = re.compile(r"Granted job allocation (\d+)")
    _PENDING_RE = re.compile(r"Pending job allocation (\d+)")
    _QUEUED_RE = re.compile(r"\bjob (\d+) queued\b")
    _ALLOCATED_RE = re.compile(r"\bjob (\d+) has been allocated resources\b")

    def __init__(
        self,
        *,
        partition: str = "dcu_htc",
        nodes: int = 1,
        ntasks_per_node: int = 1,
        cpus_per_task: int = 8,
        memory: str = "10G",
        salloc_launcher: str = "salloc",
        extra_args: list[str] | None = None,
        startup_timeout_s: float | None = None,
        hold_seconds: int = 24 * 60 * 60,
        time_limit: str = "24:00:00",
        job_id_file: Path | str | None = None,
        job_name: str | None = None,
        register_cleanup_handlers: bool = True,
    ):
        self.partition = partition
        self.nodes = nodes
        self.ntasks_per_node = ntasks_per_node
        self.cpus_per_task = cpus_per_task
        self.memory = memory
        self.salloc_launcher = salloc_launcher
        self.extra_args = list(extra_args or [])
        self.startup_timeout_s = startup_timeout_s
        self.hold_seconds = hold_seconds
        self.time_limit = time_limit
        self.job_id_file = Path(job_id_file) if job_id_file else None
        self.job_name = job_name or f"DFTAgent_salloc_{os.getpid()}"
        self.register_cleanup_handlers = bool(register_cleanup_handlers)
        self.job_id: str | None = None
        self._known_job_ids: set[str] = set()
        self._job_lock = threading.Lock()
        self.process: subprocess.Popen | None = None
        self._output_queue: queue.Queue[str] = queue.Queue()
        self._registered_atexit = False
        self._registered_signals = False
        self._previous_signal_handlers: dict[int, Any] = {}
        self._closed = False

    def start(self) -> str:
        """Start ``salloc`` if needed and return the allocation job id."""
        if self.job_id and self.process and self.process.poll() is None:
            return self.job_id
        self._closed = False
        cmd = [
            self.salloc_launcher,
            "-p",
            self.partition,
            "-N",
            str(self.nodes),
            f"--ntasks-per-node={self.ntasks_per_node}",
            f"--cpus-per-task={self.cpus_per_task}",
            f"--mem={self.memory}",
            f"--time={self.time_limit}",
            "-J",
            self.job_name,
            *self.extra_args,
            "bash",
            "-lc",
            f"echo DFTAGENT_SALLOC_READY:${{SLURM_JOB_ID:-}}; sleep {self.hold_seconds}",
        ]
        logger.info("Requesting Slurm allocation: %s", " ".join(cmd[:-3]))
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._pump_output, daemon=True).start()
        if self.register_cleanup_handlers:
            self._register_cleanup_handlers()
        deadline = (
            time.monotonic() + float(self.startup_timeout_s)
            if self.startup_timeout_s is not None and self.startup_timeout_s > 0
            else None
        )
        recent_lines: list[str] = []
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                self.close()
                raise TimeoutError(
                    f"Timed out after {self.startup_timeout_s:.0f}s waiting for salloc allocation"
                )
            if self.process.poll() is not None:
                raise RuntimeError(
                    "salloc exited before granting an allocation: " + " | ".join(recent_lines[-8:])
                )
            try:
                line = self._output_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            recent_lines.append(line)
            self._remember_job_ids_from_line(line)
            logger.info("salloc: %s", line)
            ready = self._READY_RE.search(line) or self._GRANTED_RE.search(line)
            if ready:
                self.job_id = ready.group(1)
                self._remember_job_id(self.job_id)
                logger.info("Slurm allocation ready: job_id=%s", self.job_id)
                return self.job_id

    def close(self, *, cancel_jobs: bool = True, wait_timeout_s: float = 2.0) -> None:
        """Release the allocation. Safe to call multiple times."""
        if self._closed:
            return
        self._closed = True
        with self._job_lock:
            job_ids = set(self._known_job_ids)
            if self.job_id:
                job_ids.add(self.job_id)
        self.job_id = None
        cancelled = self._scancel_job_ids(job_ids) if cancel_jobs else set()
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=max(0.1, float(wait_timeout_s)))
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
        with self._job_lock:
            late_job_ids = set(self._known_job_ids) - cancelled
        if cancel_jobs:
            self._scancel_job_ids(late_job_ids)
        self.process = None

    def _pump_output(self) -> None:
        if not self.process or self.process.stdout is None:
            return
        for line in self.process.stdout:
            clean = line.strip()
            if clean:
                self._remember_job_ids_from_line(clean)
                self._output_queue.put(clean)

    def _remember_job_ids_from_line(self, line: str) -> None:
        for pattern in (
            self._READY_RE,
            self._GRANTED_RE,
            self._PENDING_RE,
            self._QUEUED_RE,
            self._ALLOCATED_RE,
        ):
            match = pattern.search(line)
            if match:
                self._remember_job_id(match.group(1))

    def _remember_job_id(self, job_id: str | None) -> None:
        if not job_id:
            return
        job_id = str(job_id)
        with self._job_lock:
            if job_id in self._known_job_ids:
                return
            self._known_job_ids.add(job_id)
            if self.job_id_file is not None:
                try:
                    self.job_id_file.parent.mkdir(parents=True, exist_ok=True)
                    with self.job_id_file.open("a", encoding="utf-8") as handle:
                        handle.write(job_id + "\n")
                except Exception as exc:
                    logger.debug("Unable to persist Slurm allocation job id %s: %s", job_id, exc)

    def _scancel_job_ids(self, job_ids: set[str]) -> set[str]:
        candidates = sorted({str(job_id) for job_id in job_ids if str(job_id).strip()})
        if not candidates:
            return set()
        logger.info("Releasing Slurm allocation job_ids=%s", ",".join(candidates))
        try:
            completed = subprocess.run(
                ["scancel", *candidates],
                text=True,
                capture_output=True,
                timeout=15,
            )
            if completed.returncode != 0:
                logger.warning(
                    "scancel returned %s for job_ids=%s stderr=%s",
                    completed.returncode,
                    ",".join(candidates),
                    (completed.stderr or "").strip()[:500],
                )
            return set(candidates)
        except Exception as exc:
            logger.warning("Unable to scancel Slurm allocations %s: %s", ",".join(candidates), exc)
            return set()

    def _register_cleanup_handlers(self) -> None:
        if self._registered_atexit:
            pass
        else:
            atexit.register(self.close)
            self._registered_atexit = True
        if self._registered_signals:
            return
        signals = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGHUP"):
            signals.append(signal.SIGHUP)
        for sig in signals:
            try:
                previous = signal.getsignal(sig)
                self._previous_signal_handlers[int(sig)] = previous

                def _handler(signum, frame, *, _self=self):
                    logger.warning("Received signal %s; closing Slurm allocation resources", signum)
                    _self.close()
                    prior = _self._previous_signal_handlers.get(int(signum))
                    if callable(prior):
                        prior(signum, frame)
                    elif prior == signal.SIG_IGN:
                        return
                    elif signum == signal.SIGINT:
                        raise KeyboardInterrupt
                    raise SystemExit(128 + int(signum))

                signal.signal(sig, _handler)
            except Exception:
                continue
        self._registered_signals = True


class SlurmAllocationPool:
    """Hold several independent 8-CPU Slurm allocations and lend them to tasks."""

    def __init__(
        self,
        *,
        count: int = 1,
        partition: str = "dcu_htc",
        nodes: int = 1,
        ntasks_per_node: int = 1,
        cpus_per_task: int = 8,
        memory: str = "10G",
        salloc_launcher: str = "salloc",
        extra_args: list[str] | None = None,
        startup_timeout_s: float | None = None,
        hold_seconds: int = 24 * 60 * 60,
        time_limit: str = "24:00:00",
        job_id_file: Path | str | None = None,
        job_name: str | None = None,
    ):
        self.count = max(1, int(count))
        self.startup_timeout_s = startup_timeout_s
        self.job_id_file = Path(job_id_file) if job_id_file else None
        self.job_name = job_name or f"DFTAgent_salloc_{os.getpid()}"
        self.allocations = [
            SlurmAllocation(
                partition=partition,
                nodes=nodes,
                ntasks_per_node=ntasks_per_node,
                cpus_per_task=cpus_per_task,
                memory=memory,
                salloc_launcher=salloc_launcher,
                extra_args=list(extra_args or []),
                startup_timeout_s=startup_timeout_s,
                hold_seconds=hold_seconds,
                time_limit=time_limit,
                job_id_file=self.job_id_file,
                job_name=self.job_name,
                register_cleanup_handlers=False,
            )
            for _ in range(self.count)
        ]
        self._ready: list[SlurmAllocation] = []
        self._available: queue.Queue[SlurmAllocation] = queue.Queue()  # legacy; group acquire uses _available_items
        self._available_items: list[SlurmAllocation] = []
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._started = False
        self._closed = False
        self._launcher_failures = 0
        self._launcher_threads: list[threading.Thread] = []
        self._registered_atexit = False
        self._registered_signals = False
        self._previous_signal_handlers: dict[int, Any] = {}

    def start(self) -> str:
        """Start all requested allocations and wait until at least one is ready."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot start Slurm allocation pool after it was closed")
            if self._started and self._ready:
                return str(self._ready[0].job_id)
            if not self._started:
                self._started = True
                self._register_cleanup_handlers()
                logger.info(
                    "Requesting Slurm allocation pool: count=%d wait_policy=%s",
                    self.count,
                    "indefinite-until-first-ready",
                )
                for alloc in self.allocations:
                    thread = threading.Thread(
                        target=self._launch_allocation,
                        args=(alloc,),
                        daemon=True,
                    )
                    self._launcher_threads.append(thread)
                    thread.start()
            while not self._ready:
                if self._closed:
                    raise RuntimeError("Slurm allocation pool closed before any allocation became ready")
                if self._launcher_failures >= self.count:
                    raise RuntimeError("All Slurm allocation requests failed before any allocation became ready")
                self._condition.wait(timeout=1.0)
            return str(self._ready[0].job_id)

    def _launch_allocation(self, alloc: SlurmAllocation) -> None:
        try:
            alloc.start()
        except Exception as exc:
            logger.warning("Slurm allocation request failed: %s", exc)
            alloc.close()
            with self._lock:
                self._launcher_failures += 1
                self._condition.notify_all()
            return
        close_after_start = False
        with self._lock:
            if self._closed:
                close_after_start = True
            elif alloc not in self._ready:
                self._ready.append(alloc)
                self._available_items.append(alloc)
                self._available.put(alloc)
                logger.info(
                    "Slurm allocation pool acquired: acquired=%d/%d job_ids=%s",
                    len(self._ready),
                    self.count,
                    ",".join(str(item.job_id) for item in self._ready),
                )
            self._condition.notify_all()
        if close_after_start:
            alloc.close()
            return

    def acquire(self) -> SlurmAllocation:
        return self.acquire_many(1)[0]

    def acquire_many(self, count: int) -> list[SlurmAllocation]:
        self.start()
        requested = max(1, int(count))
        with self._condition:
            while not self._closed and len(self._available_items) < requested:
                self._condition.wait(timeout=1.0)
            if self._closed:
                raise RuntimeError("Cannot acquire Slurm allocation after pool was closed")
            acquired = self._available_items[:requested]
            del self._available_items[:requested]
            logger.debug(
                "Slurm allocation pool loaned group: requested=%d job_ids=%s available_after=%d",
                requested,
                ",".join(str(item.job_id) for item in acquired),
                len(self._available_items),
            )
            return acquired

    def release(self, allocation: SlurmAllocation) -> None:
        self.release_many([allocation])

    def release_many(self, allocations: list[SlurmAllocation]) -> None:
        valid = [alloc for alloc in allocations if alloc.job_id]
        if not valid:
            return
        with self._condition:
            if self._closed:
                return
            self._available_items.extend(valid)
            for alloc in valid:
                self._available.put(alloc)
            logger.debug(
                "Slurm allocation pool returned group: count=%d job_ids=%s available_after=%d",
                len(valid),
                ",".join(str(item.job_id) for item in valid),
                len(self._available_items),
            )
            self._condition.notify_all()

    def job_ids(self) -> list[str]:
        self.start()
        with self._lock:
            return [str(alloc.job_id) for alloc in self._ready if alloc.job_id]

    def size(self) -> int:
        return len(self.job_ids())

    def requested_size(self) -> int:
        return self.count

    def ready_size(self) -> int:
        self.start()
        with self._lock:
            return len(self._ready)

    def parallelism(
        self,
        *,
        settle_seconds: float = 15.0,
        max_allocations: int | None = None,
    ) -> int:
        """Return how many allocations should be used for the next sharded job.

        ``start()`` waits indefinitely until at least one allocation is ready.
        After that, wait only until either the requested per-evaluation cap is
        ready or a short settling window expires. Pending allocations do not
        block the subset forever; later evaluations can use them once ready.
        """
        self.start()
        cap = self.count if max_allocations is None else max(1, min(self.count, int(max_allocations)))
        deadline = time.monotonic() + max(0.0, float(settle_seconds))
        with self._condition:
            while (
                len(self._ready) < cap
                and len(self._ready) + self._launcher_failures < self.count
                and time.monotonic() < deadline
                and not self._closed
            ):
                remaining = deadline - time.monotonic()
                self._condition.wait(timeout=max(0.0, min(1.0, remaining)))
            return max(1, min(cap, len(self._ready)))

    def close(self) -> None:
        if self._closed:
            return
        with self._lock:
            self._closed = True
            ready_ids = [str(alloc.job_id) for alloc in self._ready if alloc.job_id]
            known_ids = sorted({
                str(job_id)
                for alloc in self.allocations
                for job_id in getattr(alloc, "_known_job_ids", set())
            })
            self._condition.notify_all()
        logger.info(
            "Closing Slurm allocation pool: requested=%d ready=%d ready_job_ids=%s known_job_ids=%s",
            self.count,
            len(ready_ids),
            ",".join(ready_ids) or "none",
            ",".join(known_ids) or "none",
        )
        if known_ids:
            # Cancel all allocations in one Slurm request before touching the
            # local salloc processes.  This makes SIGHUP/terminal close cleanup
            # much less likely to leave later allocations behind.
            self.allocations[0]._scancel_job_ids(set(known_ids))
        for alloc in self.allocations:
            alloc.close(cancel_jobs=False, wait_timeout_s=0.5)
        current = threading.current_thread()
        for thread in list(self._launcher_threads):
            if thread is current or not thread.is_alive():
                continue
            thread.join(timeout=2.0)

    def _register_cleanup_handlers(self) -> None:
        if not self._registered_atexit:
            atexit.register(self.close)
            self._registered_atexit = True
        if self._registered_signals:
            return
        signals = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGHUP"):
            signals.append(signal.SIGHUP)
        for sig in signals:
            try:
                previous = signal.getsignal(sig)
                self._previous_signal_handlers[int(sig)] = previous

                def _handler(signum, frame, *, _self=self):
                    logger.warning("Received signal %s; closing Slurm allocation resources", signum)
                    _self.close()
                    prior = _self._previous_signal_handlers.get(int(signum))
                    if callable(prior):
                        prior(signum, frame)
                    elif prior == signal.SIG_IGN:
                        return
                    elif signum == signal.SIGINT:
                        raise KeyboardInterrupt
                    raise SystemExit(128 + int(signum))

                signal.signal(sig, _handler)
            except Exception:
                continue
        self._registered_signals = True


def _worker_init(n_threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)


def _worker_compute_molecule(args: tuple) -> MoleculeEvalResult:
    """Compute single-molecule energy in a worker process.

    Args: (subset_dir, mol_name, xc_dict, basis, conv_tol, max_cycle[, density_output_dir])
    Returns: MoleculeEvalResult with energy or a detailed failure message.
    """
    started_at = time.time()
    if len(args) >= 7:
        subset_dir, mol_name, xc_dict, basis, conv_tol, max_cycle, density_output_dir = args[:7]
    else:
        subset_dir, mol_name, xc_dict, basis, conv_tol, max_cycle = args
        density_output_dir = None
    from pyscf import gto, dft
    from ..pyscf_registry import register_with_pyscf
    xc = XCFunctional.from_dict(xc_dict)

    geom = load_molecule_geometry(Path(subset_dir), mol_name)
    if geom is None:
        ended_at = time.time()
        return MoleculeEvalResult(
            ok=False,
            error=f"missing_geometry: no struc.xyz or coord under {Path(subset_dir) / mol_name}",
            walltime_s=ended_at - started_at,
            started_at=started_at,
            ended_at=ended_at,
        )
    atoms, coords = geom

    mol_dir = Path(subset_dir) / mol_name
    charge = 0
    spin = 0
    chrg_file = mol_dir / ".CHRG"
    if chrg_file.exists():
        try:
            charge = int(chrg_file.read_text().strip())
        except (ValueError, IndexError):
            pass
    uhf_file = mol_dir / ".UHF"
    if uhf_file.exists():
        try:
            spin = int(uhf_file.read_text().strip())
        except (ValueError, IndexError):
            pass

    try:
        mol = gto.Mole()
        mol.atom = list(zip(atoms, coords))
        mol.basis = basis
        mol.charge = charge
        mol.spin = spin
        mol.verbose = 0
        mol.unit = "Angstrom"
        mol.build()

        if mol.spin > 0:
            mf = dft.UKS(mol)
        else:
            mf = dft.RKS(mol)

        register_with_pyscf(mf, xc)
        mf.conv_tol = conv_tol
        mf.max_cycle = max_cycle
        mf.kernel()

        if not mf.converged:
            ended_at = time.time()
            return MoleculeEvalResult(
                ok=False,
                error=f"scf_not_converged: max_cycle={max_cycle}, xc={xc.name}, molecule={mol_name}",
                walltime_s=ended_at - started_at,
                started_at=started_at,
                ended_at=ended_at,
            )
        if density_output_dir:
            try:
                density_dir = Path(density_output_dir)
                density_dir.mkdir(parents=True, exist_ok=True)
                safe_mol = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(mol_name))[:160]
                dm = mf.make_rdm1()
                np.savez_compressed(density_dir / f"{safe_mol}.dm.npz", dm=np.asarray(dm))
                meta = {
                    "mol_name": mol_name,
                    "energy": float(mf.e_tot),
                    "converged": bool(mf.converged),
                    "basis": basis,
                    "charge": int(mol.charge),
                    "spin": int(mol.spin),
                    "xc_name": xc.name,
                    "density_file": str(density_dir / f"{safe_mol}.dm.npz"),
                }
                (density_dir / f"{safe_mol}.meta.json").write_text(
                    json.dumps(meta, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception as exc:
                logger.warning("Unable to save density for %s/%s: %s", subset_dir, mol_name, exc)
        ended_at = time.time()
        return MoleculeEvalResult(
            ok=True,
            energy=float(mf.e_tot),
            walltime_s=ended_at - started_at,
            started_at=started_at,
            ended_at=ended_at,
        )
    except Exception as exc:
        ended_at = time.time()
        return MoleculeEvalResult(
            ok=False,
            error=f"{exc.__class__.__name__}: {exc}",
            walltime_s=ended_at - started_at,
            started_at=started_at,
            ended_at=ended_at,
        )


def _compute_reaction_energy_from_cache(
    reaction: dict, mol_energies: dict[str, float | None]
) -> float | None:
    energy = 0.0
    for group, stoich in zip(reaction['species_groups'], reaction['stoich_coeffs']):
        for mol_name in group:
            e = mol_energies.get(mol_name)
            if e is None:
                return None
            energy += stoich * e
    return energy * HARTREE_TO_KCAL


def _molecule_names_from_reactions(reactions: list[dict]) -> list[str]:
    mol_names: set[str] = set()
    for rxn in reactions:
        for group in rxn["species_groups"]:
            for mol_name in group:
                mol_names.add(mol_name)
    return sorted(mol_names)


def compute_molecule_batch(
    *,
    subset_dir: Path | str,
    mol_names: list[str],
    xc: XCFunctional,
    basis: str,
    scf_conv_tol: float,
    scf_max_cycle: int,
    n_workers: int,
    n_threads: int,
    density_output_dir: Path | str | None = None,
    profile: bool = False,
) -> dict[str, Any]:
    """Compute a shard of molecule energies for one subset."""
    subset_dir = Path(subset_dir)
    xc_dict = xc.to_dict()
    t0 = time.time()
    mol_energies: dict[str, float | None] = {}
    mol_errors: dict[str, str] = {}
    molecule_timings: list[dict[str, Any]] = []
    names = list(mol_names)
    if n_workers > 1 and len(names) > 1:
        args_list = [
            (str(subset_dir), mol_name, xc_dict, basis, scf_conv_tol, scf_max_cycle, str(density_output_dir) if density_output_dir else None)
            for mol_name in names
        ]
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(n_threads,),
        ) as pool:
            results = pool.map(_worker_compute_molecule, args_list)
            for mol_name, outcome in zip(names, results):
                if outcome.ok:
                    mol_energies[mol_name] = outcome.energy
                else:
                    mol_energies[mol_name] = None
                    mol_errors[mol_name] = outcome.error
                if profile:
                    molecule_timings.append({
                        "mol_name": mol_name,
                        "ok": bool(outcome.ok),
                        "walltime_s": float(outcome.walltime_s),
                        "started_at": float(outcome.started_at),
                        "ended_at": float(outcome.ended_at),
                        "error": outcome.error,
                    })
    else:
        os.environ["OMP_NUM_THREADS"] = str(n_threads)
        os.environ["MKL_NUM_THREADS"] = str(n_threads)
        for mol_name in names:
            outcome = _worker_compute_molecule(
                (str(subset_dir), mol_name, xc_dict, basis, scf_conv_tol, scf_max_cycle, str(density_output_dir) if density_output_dir else None)
            )
            if outcome.ok:
                mol_energies[mol_name] = outcome.energy
            else:
                mol_energies[mol_name] = None
                mol_errors[mol_name] = outcome.error
            if profile:
                molecule_timings.append({
                    "mol_name": mol_name,
                    "ok": bool(outcome.ok),
                    "walltime_s": float(outcome.walltime_s),
                    "started_at": float(outcome.started_at),
                    "ended_at": float(outcome.ended_at),
                    "error": outcome.error,
                })
    t1 = time.time()
    payload = {
        "mol_energies": mol_energies,
        "mol_errors": mol_errors,
        "walltime_s": t1 - t0,
        "started_at": t0,
        "ended_at": t1,
        "n_molecules": len(names),
    }
    if profile:
        payload["molecule_timings"] = molecule_timings
        payload["n_workers"] = int(n_workers)
        payload["n_threads"] = int(n_threads)
    return payload


def _subset_result_from_molecule_energies(
    *,
    subset_name: str,
    reactions: list[dict],
    mol_energies: dict[str, float | None],
    mol_errors: dict[str, str],
    walltime_s: float,
    log_label: str = "",
) -> tuple[SubsetResult, list[dict[str, Any]]]:
    errors = []
    n_failures = 0
    failure_details: list[str] = []
    reaction_rows: list[dict[str, Any]] = []

    for idx, rxn in enumerate(reactions, start=1):
        e_calc = _compute_reaction_energy_from_cache(rxn, mol_energies)
        if e_calc is None:
            n_failures += 1
            failed_molecules = []
            for group in rxn["species_groups"]:
                for mol_name in group:
                    if mol_energies.get(mol_name) is None:
                        failed_molecules.append(
                            f"{mol_name}: {mol_errors.get(mol_name, 'missing_energy')}"
                        )
            if not failed_molecules:
                failed_molecules.append("unknown_reaction_failure")
            failure_text = f"reaction_{idx:03d} failed: " + " | ".join(failed_molecules[:6])
            failure_details.append(failure_text)
            reaction_rows.append(
                {
                    "reaction_index": idx,
                    "subset": subset_name,
                    "species_groups": rxn["species_groups"],
                    "stoich_coeffs": rxn["stoich_coeffs"],
                    "reference_kcal": rxn["ref_value"],
                    "calculated_kcal": None,
                    "abs_error_kcal": None,
                    "status": "failed",
                    "failure_detail": failure_text,
                }
            )
            continue
        e_ref = rxn["ref_value"]
        abs_error = abs(e_calc - e_ref)
        errors.append(abs_error)
        reaction_rows.append(
            {
                "reaction_index": idx,
                "subset": subset_name,
                "species_groups": rxn["species_groups"],
                "stoich_coeffs": rxn["stoich_coeffs"],
                "reference_kcal": e_ref,
                "calculated_kcal": e_calc,
                "abs_error_kcal": abs_error,
                "status": "ok",
                "failure_detail": "",
            }
        )

    if errors:
        mae = float(np.mean(errors))
        rms = float(np.sqrt(np.mean(np.array(errors) ** 2)))
        max_err = float(np.max(errors))
    else:
        mae, rms, max_err = 0.0, 0.0, 0.0

    subset_result = SubsetResult(
        subset=subset_name,
        mae_kcal=mae,
        rms_kcal=rms,
        max_error_kcal=max_err,
        n_reactions=len(errors),
        n_failures=n_failures,
        walltime_s=walltime_s,
        per_reaction_errors=errors,
        failure_details=failure_details,
    )
    if subset_result.n_failures > 0 and log_label:
        logger.warning(
            "Subset %s for %s had %d failures; sample=%s",
            subset_name,
            log_label,
            subset_result.n_failures,
            subset_result.failure_details[:2],
        )
    return subset_result, reaction_rows


def compute_subset_result(
    *,
    gmtkn55_dir: Path | str,
    subset_name: str,
    xc: XCFunctional,
    basis: str,
    scf_conv_tol: float,
    scf_max_cycle: int,
    n_workers: int,
    n_threads: int,
    log_label: str = "",
) -> SubsetResult:
    subset_result, _reaction_rows = compute_subset_diagnostics(
        gmtkn55_dir=gmtkn55_dir,
        subset_name=subset_name,
        xc=xc,
        basis=basis,
        scf_conv_tol=scf_conv_tol,
        scf_max_cycle=scf_max_cycle,
        n_workers=n_workers,
        n_threads=n_threads,
        log_label=log_label,
    )
    return subset_result


def compute_subset_diagnostics(
    *,
    gmtkn55_dir: Path | str,
    subset_name: str,
    xc: XCFunctional,
    basis: str,
    scf_conv_tol: float,
    scf_max_cycle: int,
    n_workers: int,
    n_threads: int,
    log_label: str = "",
) -> tuple[SubsetResult, list[dict[str, Any]]]:
    subset_dir = Path(gmtkn55_dir) / subset_name
    res_path = subset_dir / ".res"

    if not res_path.exists():
        missing_msg = f"missing_res_file: {res_path}"
        if log_label:
            logger.warning("Subset %s failed for %s: %s", subset_name, log_label, missing_msg)
        return SubsetResult(
            subset=subset_name, mae_kcal=0, rms_kcal=0, max_error_kcal=0,
            n_reactions=0, n_failures=1, walltime_s=0,
            per_reaction_errors=[],
            failure_details=[f"fatal:{missing_msg}"],
        ), []

    reactions = parse_res_file(res_path)
    if not reactions:
        parse_msg = f"no_reactions_parsed: {res_path}"
        if log_label:
            logger.warning("Subset %s failed for %s: %s", subset_name, log_label, parse_msg)
        return SubsetResult(
            subset=subset_name, mae_kcal=0, rms_kcal=0, max_error_kcal=0,
            n_reactions=0, n_failures=1, walltime_s=0,
            per_reaction_errors=[],
            failure_details=[f"fatal:{parse_msg}"],
        ), []
    t0 = time.time()
    batch = compute_molecule_batch(
        subset_dir=subset_dir,
        mol_names=_molecule_names_from_reactions(reactions),
        xc=xc,
        basis=basis,
        scf_conv_tol=scf_conv_tol,
        scf_max_cycle=scf_max_cycle,
        n_workers=n_workers,
        n_threads=n_threads,
    )
    return _subset_result_from_molecule_energies(
        subset_name=subset_name,
        reactions=reactions,
        mol_energies=batch["mol_energies"],
        mol_errors=batch["mol_errors"],
        walltime_s=time.time() - t0,
        log_label=log_label,
    )


class BenchmarkRunner:
    def __init__(
        self,
        gmtkn55_dir: Path | str,
        basis: str = "def2-TZVP",
        subsets: list[str] | None = None,
        scf_max_cycle: int = 200,
        scf_conv_tol: float = 1e-8,
        n_workers: int = 4,
        n_threads: int = 4,
        cache_dir: Path | str | None = None,
        execution_backend: str = "local",
        slurm_launcher: str = "srun",
        slurm_nodes: int = 1,
        slurm_cpus_per_task: int = 8,
        slurm_mem: str = "10G",
        slurm_partition: str = "dcu_htc",
        slurm_output_dir: Path | str | None = None,
        slurm_extra_args: list[str] | None = None,
        slurm_allocation: SlurmAllocation | SlurmAllocationPool | None = None,
        aggregate_mode: str = "reaction_mean",
        slurm_allocations_per_evaluation: int | None = None,
        slurm_docker_image: str | None = None,
        slurm_docker_container_repo_root: str = "/workspace/DFTAgent",
        slurm_docker_time: str = "24:00:00",
        slurm_docker_total_cpus: int = 128,
        slurm_docker_cpus_per_worker: int = 32,
        slurm_docker_worker_count: int = 0,
        slurm_docker_nodes: int = 0,
        slurm_docker_ntasks_per_node: int = 0,
        slurm_docker_cpus_per_node: int = 64,
        slurm_docker_submit_mode: str = "allocation",
        slurm_docker_array_max_running: int = 0,
        slurm_docker_queue_dispatch_mode: str = "candidate_dedicated",
        slurm_docker_thread_policy: str = "atom_adaptive",
        slurm_docker_atom_thread_bins: str = "2:1,6:2,10:4,999:8",
        slurm_docker_max_threads_per_sample: int = 8,
        slurm_docker_molecules_per_task: int = 1,
        slurm_docker_output_dir: Path | str | None = None,
        slurm_docker_extra_args: list[str] | None = None,
        vc_prefix: str = "",
        vc_container_repo_root: str = "/code/remote/DFTAgent",
        vc_python_executable: str = "python3",
        vc_conda_sh: str = "",
        vc_conda_env: str = "",
        vc_pre_cmd: str = "",
        vc_molecules_per_task: int = 8,
        vc_concurrent_subset_jobs: int = 32,
        vc_tasks_per_submit: int = 1,
        vc_poll_seconds: float = 10.0,
        vc_wait_timeout_seconds: float = 0.0,
        vc_output_dir: Path | str | None = None,
    ):
        self.gmtkn55_dir = Path(gmtkn55_dir)
        self.basis = basis
        self.subsets = subsets or SEARCH_SUBSETS
        self.scf_max_cycle = scf_max_cycle
        self.scf_conv_tol = scf_conv_tol
        self.n_workers = n_workers
        self.n_threads = n_threads
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if aggregate_mode not in {"reaction_mean", "subset_mean"}:
            raise ValueError(f"Unsupported benchmark aggregate mode: {aggregate_mode}")
        self.aggregate_mode = aggregate_mode
        self.execution_backend = execution_backend
        if self.execution_backend not in {"local", "srun", "slurm_docker_queue", "vc", "persistent_vc"}:
            raise ValueError(f"Unsupported benchmark execution backend: {self.execution_backend}")
        self.slurm = SlurmLaunchConfig(
            enabled=execution_backend != "local",
            launcher=slurm_launcher,
            nodes=slurm_nodes,
            cpus_per_task=slurm_cpus_per_task,
            memory=slurm_mem,
            partition=slurm_partition,
            output_dir=Path(slurm_output_dir) if slurm_output_dir else None,
            extra_args=list(slurm_extra_args or []),
        )
        queue_image = str(slurm_docker_image or os.environ.get("DFTAGENT_IMAGE", "")).strip()
        self.slurm_docker_queue = SlurmDockerQueueConfig(
            image=queue_image,
            container_repo_root=str(slurm_docker_container_repo_root),
            time_limit=str(slurm_docker_time),
            total_cpus=max(1, int(slurm_docker_total_cpus)),
            cpus_per_worker=max(1, int(slurm_docker_cpus_per_worker)),
            worker_count=max(0, int(slurm_docker_worker_count)),
            nodes=max(0, int(slurm_docker_nodes)),
            ntasks_per_node=max(0, int(slurm_docker_ntasks_per_node)),
            cpus_per_node=max(1, int(slurm_docker_cpus_per_node)),
            submit_mode=str(slurm_docker_submit_mode),
            array_max_running=max(0, int(slurm_docker_array_max_running)),
            queue_dispatch_mode=str(slurm_docker_queue_dispatch_mode),
            thread_policy=str(slurm_docker_thread_policy),
            atom_thread_bins=str(slurm_docker_atom_thread_bins),
            max_threads_per_sample=max(1, int(slurm_docker_max_threads_per_sample)),
            molecules_per_task=max(1, int(slurm_docker_molecules_per_task)),
            output_dir=Path(slurm_docker_output_dir) if slurm_docker_output_dir else None,
            docker_extra_args=list(slurm_docker_extra_args or []),
        )
        self._slurm_docker_queue_batch_index = 0
        self.vc_manifest_runner = None
        if self.execution_backend in {"vc", "persistent_vc"}:
            from .vc_manifest_runner import VCManifestBenchmarkRunner, VCManifestRunnerConfig

            resolved_vc_prefix = str(vc_prefix or os.environ.get("DFTAGENT_VC_PREFIX", "")).strip()
            persistent_queue_dir = None
            if self.execution_backend == "persistent_vc":
                queue_value = str(os.environ.get("DFTAGENT_PERSISTENT_VC_QUEUE", "")).strip()
                if not queue_value:
                    raise ValueError(
                        "persistent_vc requires DFTAGENT_PERSISTENT_VC_QUEUE"
                    )
                persistent_queue_dir = Path(queue_value)
            vc_root = Path(vc_output_dir) if vc_output_dir else (
                self.cache_dir / "vc_manifest" if self.cache_dir else Path("results/vc_manifest")
            )
            self.vc_manifest_runner = VCManifestBenchmarkRunner(
                VCManifestRunnerConfig(
                    output_dir=vc_root,
                    vc_prefix=resolved_vc_prefix if self.execution_backend == "vc" else "",
                    persistent_queue_dir=persistent_queue_dir,
                    container_repo_root=str(vc_container_repo_root),
                    python_executable=str(vc_python_executable),
                    conda_sh=str(vc_conda_sh),
                    conda_env=str(vc_conda_env),
                    pre_cmd=str(vc_pre_cmd),
                    gmtkn55_dir=self.gmtkn55_dir,
                    subsets=tuple(self.subsets),
                    basis=self.basis,
                    scf_max_cycle=self.scf_max_cycle,
                    scf_conv_tol=self.scf_conv_tol,
                    n_workers=self.n_workers,
                    n_threads=self.n_threads,
                    molecules_per_task=int(vc_molecules_per_task),
                    aggregate_mode=self.aggregate_mode,
                    concurrent_subset_jobs=int(vc_concurrent_subset_jobs),
                    vc_tasks_per_submit=int(vc_tasks_per_submit),
                    poll_seconds=float(vc_poll_seconds),
                    wait_timeout_seconds=float(vc_wait_timeout_seconds),
                ),
                repo_root=self._repo_root(),
            )
        if self.execution_backend == "srun" and self.n_workers * self.n_threads > self.slurm.cpus_per_task:
            old_workers, old_threads = self.n_workers, self.n_threads
            if self.n_workers >= self.slurm.cpus_per_task:
                self.n_workers = max(1, self.slurm.cpus_per_task)
                self.n_threads = 1
            else:
                self.n_threads = max(1, self.slurm.cpus_per_task // max(1, self.n_workers))
            logger.warning(
                "BenchmarkRunner Slurm CPU request clamped from n_workers*n_threads=%d "
                "to %d*%d<=cpus_per_task=%d",
                old_workers * old_threads,
                self.n_workers,
                self.n_threads,
                self.slurm.cpus_per_task,
            )
        self.slurm_allocation = slurm_allocation
        self.slurm_allocations_per_evaluation = (
            max(1, int(slurm_allocations_per_evaluation))
            if slurm_allocations_per_evaluation is not None
            else None
        )
        self._cache_index: dict[str, dict[str, Path]] | None = None
        self._cache_lock = threading.RLock()
        if self.execution_backend == "srun" and self.slurm_allocation is None:
            self.slurm_allocation = SlurmAllocation(
                partition=slurm_partition,
                nodes=slurm_nodes,
                cpus_per_task=slurm_cpus_per_task,
                memory=slurm_mem,
                extra_args=list(slurm_extra_args or []),
            )

    def evaluate(self, xc: XCFunctional) -> BenchmarkResult:
        if self.execution_backend in {"slurm_docker_queue", "vc", "persistent_vc"}:
            return self.evaluate_many([xc], label_prefix="single")[0]
        logger.debug("Benchmark start for %s", self._xc_log_label(xc))
        cached = self._load_cached_result(xc)
        if cached is not None:
            return cached
        result = BenchmarkResult(
            xc_name=xc.name,
            timestamp=datetime.now().isoformat(),
        )

        for subset_name in self.subsets:
            sr = self._run_subset(subset_name, xc)
            result.subsets[subset_name] = sr
            if self._is_fatal_subset_failure(sr):
                logger.error(
                    "Benchmark aborted for %s at subset %s due to fatal dataset/config error: %s",
                    self._xc_log_label(xc),
                    subset_name,
                    sr.failure_details[:1],
                )
                self._compute_overall(result)
                return result
            # Long validation panels can be interrupted; persist each completed
            # subset immediately so expensive work is not lost on timeout.
            if self._should_cache_subset_result(sr):
                partial = BenchmarkResult(
                    xc_name=result.xc_name,
                    subsets={subset_name: sr},
                    timestamp=result.timestamp,
                )
                self._compute_overall(partial)
                self._write_cached_result(partial, xc=xc)
            else:
                logger.debug(
                    "Skipping benchmark cache update for %s subset=%s due to incomplete evaluation",
                    self._xc_log_label(xc),
                    subset_name,
                )

        self._compute_overall(result)
        cacheable_subsets = {
            subset_name: sr
            for subset_name, sr in result.subsets.items()
            if self._should_cache_subset_result(sr)
        }
        if cacheable_subsets and len(cacheable_subsets) == len(result.subsets):
            cacheable_result = BenchmarkResult(
                xc_name=result.xc_name,
                subsets=cacheable_subsets,
                timestamp=result.timestamp,
            )
            self._compute_overall(cacheable_result)
            self._write_cached_result(cacheable_result, xc=xc)
        elif cacheable_subsets:
            logger.debug(
                "Skipping final benchmark cache update for %s because not all subsets completed successfully",
                self._xc_log_label(xc),
            )
        logger.debug(
            "Benchmark complete for %s: overall_mae=%.3f failures=%d",
            self._xc_log_label(xc),
            result.overall_mae,
            result.total_failures,
        )
        return result

    def evaluate_many(self, xcs: list[XCFunctional], *, label_prefix: str = "batch") -> list[BenchmarkResult]:
        """Evaluate several XC functionals, batching them when the backend supports it."""
        if not xcs:
            return []
        if self.execution_backend not in {"slurm_docker_queue", "vc", "persistent_vc"}:
            return [self.evaluate(xc) for xc in xcs]

        results: list[BenchmarkResult | None] = [None] * len(xcs)
        pending: list[tuple[int, XCFunctional]] = []
        for index, xc in enumerate(xcs):
            cached = self._load_cached_result(xc)
            if cached is not None:
                results[index] = cached
            else:
                pending.append((index, xc))
        if pending:
            if self.execution_backend in {"vc", "persistent_vc"}:
                if self.vc_manifest_runner is None:
                    raise RuntimeError("VC benchmark runner was not initialised")
                pending_results = self.vc_manifest_runner.evaluate_many(
                    [xc for _index, xc in pending],
                    label_prefix=label_prefix,
                )
                for (index, xc), result in zip(pending, pending_results, strict=True):
                    results[index] = result
                    if result.subsets and all(
                        self._should_cache_subset_result(sr) for sr in result.subsets.values()
                    ):
                        self._write_cached_result(result, xc=xc)
            else:
                batch_size = self._slurm_docker_queue_candidate_batch_size()
                for offset in range(0, len(pending), batch_size):
                    chunk = pending[offset : offset + batch_size]
                    batch_results = self._evaluate_many_via_slurm_docker_queue(
                        chunk,
                        label_prefix=f"{label_prefix}_chunk{offset // batch_size + 1:02d}",
                    )
                    for index, result in batch_results.items():
                        results[index] = result
        return [result if result is not None else self._fatal_benchmark_result(xcs[index], "missing_queue_batch_result") for index, result in enumerate(results)]

    def _slurm_docker_queue_candidate_batch_size(self) -> int:
        config = self.slurm_docker_queue
        if str(config.queue_dispatch_mode) != "candidate_dedicated":
            return max(1, len(self.subsets))
        workers = int(config.worker_count or 0)
        if workers <= 0:
            workers = max(1, math.ceil(int(config.total_cpus) / max(1, int(config.cpus_per_worker))))
        return max(1, workers)

    def _evaluate_many_via_slurm_docker_queue(
        self,
        pending: list[tuple[int, XCFunctional]],
        *,
        label_prefix: str,
    ) -> dict[int, BenchmarkResult]:
        config = self.slurm_docker_queue
        if not config.image:
            raise RuntimeError("slurm_docker_queue backend requires --slurm-docker-image or DFTAGENT_IMAGE")
        repo_root = self._repo_root()
        script = repo_root / "benchmark" / "scripts" / "evaluate_manifest_slurm_docker_queue.py"
        if not script.exists():
            raise FileNotFoundError(f"Missing docker queue evaluator: {script}")
        self._slurm_docker_queue_batch_index += 1
        safe_prefix = self._sanitize_label(label_prefix)[:64] or "batch"
        root = config.output_dir or self.slurm.output_dir or (self.cache_dir / "slurm_docker_queue" if self.cache_dir else Path("slurm_docker_queue"))
        batch_dir = Path(root) / f"{safe_prefix}_{self._slurm_docker_queue_batch_index:05d}_{int(time.time())}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = batch_dir / "candidate_manifest.json"
        manifest = []
        index_by_label: dict[str, int] = {}
        for manifest_index, (original_index, xc) in enumerate(pending, start=1):
            label = f"{safe_prefix}_{manifest_index:03d}_{self._short_label_for_xc(xc)}"
            index_by_label[label] = original_index
            manifest.append(
                {
                    "rank": manifest_index,
                    "label": label,
                    "source": "basin_agent",
                    "source_id": xc.name,
                    "raw_reward": xc.metadata.get("source_raw_reward"),
                    "xc_dict": xc.to_dict(),
                }
            )
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        common_cmd = [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(batch_dir),
            "--gmtkn55-dir",
            str(self.gmtkn55_dir),
            "--subsets",
            *list(self.subsets),
            "--basis",
            str(self.basis),
            "--scf-max-cycle",
            str(self.scf_max_cycle),
            "--scf-conv-tol",
            str(self.scf_conv_tol),
            "--aggregate-mode",
            str(self.aggregate_mode),
            "--molecules-per-task",
            str(config.molecules_per_task),
            "--image",
            str(config.image),
            "--container-repo-root",
            str(config.container_repo_root),
            "--partition",
            str(self.slurm.partition),
            "--time",
            str(config.time_limit),
            "--mem",
            str(self.slurm.memory),
            "--total-cpus",
            str(config.total_cpus),
            "--cpus-per-worker",
            str(config.cpus_per_worker),
            "--worker-count",
            str(config.worker_count),
            "--nodes",
            str(config.nodes),
            "--ntasks-per-node",
            str(config.ntasks_per_node),
            "--cpus-per-node",
            str(config.cpus_per_node),
            "--slurm-submit-mode",
            str(config.submit_mode),
            "--array-max-running",
            str(config.array_max_running),
            "--queue-dispatch-mode",
            str(config.queue_dispatch_mode),
            "--thread-policy",
            str(config.thread_policy),
            "--n-threads",
            str(self.n_threads),
            "--atom-thread-bins",
            str(config.atom_thread_bins),
            "--max-threads-per-sample",
            str(config.max_threads_per_sample),
            "--log-level",
            "INFO",
        ]
        extra_sbatch: list[str] = []
        slurm_nodelist = ""
        for extra in list(self.slurm.extra_args or []):
            item = str(extra)
            if item.startswith("--nodelist="):
                slurm_nodelist = item.split("=", 1)[1]
            elif item in {"--nodelist", "-w"}:
                logger.warning("Ignoring incomplete Slurm nodelist extra arg %r; use --slurm-nodelist", item)
            else:
                extra_sbatch.append(item)
        if slurm_nodelist:
            common_cmd.extend(["--slurm-nodelist", slurm_nodelist])
        for extra in extra_sbatch:
            common_cmd.append(f"--extra-sbatch={extra}")
        for extra in list(config.docker_extra_args or []):
            common_cmd.append(f"--docker-extra-args={extra}")

        logger.info(
            "Slurm+Docker queue batch start label=%s candidates=%d subsets=%s output=%s workers=%d cpus_per_worker=%d",
            label_prefix,
            len(pending),
            ",".join(self.subsets),
            batch_dir,
            config.worker_count or max(1, math.ceil(config.total_cpus / max(1, config.cpus_per_worker))),
            config.cpus_per_worker,
        )
        self._run_queue_command([*common_cmd, "--prepare-only"], step="prepare", batch_dir=batch_dir)
        self._run_queue_command(["sbatch", "--wait", str(batch_dir / "run_slurm_docker_queue.sbatch")], step="sbatch", batch_dir=batch_dir)
        self._run_queue_command([*common_cmd, "--aggregate-only"], step="aggregate", batch_dir=batch_dir)
        results_path = batch_dir / "slurm_docker_queue_benchmark_results.json"
        if not results_path.exists():
            raise RuntimeError(f"Slurm+Docker queue did not produce {results_path}")
        payloads = json.loads(results_path.read_text(encoding="utf-8"))
        mapped: dict[int, BenchmarkResult] = {}
        for payload in payloads:
            candidate = payload.get("candidate") or {}
            label = str(candidate.get("label") or "")
            original_index = index_by_label.get(label)
            if original_index is None:
                continue
            result = self._benchmark_result_from_queue_payload(payload, fallback_name=str(candidate.get("source_id") or label))
            mapped[original_index] = result
            cache_xc = pending[[idx for idx, _xc in pending].index(original_index)][1]
            cacheable = all(self._should_cache_subset_result(sr) for sr in result.subsets.values())
            if cacheable:
                self._write_cached_result(result, xc=cache_xc)
        logger.info(
            "Slurm+Docker queue batch end label=%s completed=%d/%d output=%s",
            label_prefix,
            len(mapped),
            len(pending),
            batch_dir,
        )
        return mapped

    def _run_queue_command(self, cmd: list[str], *, step: str, batch_dir: Path) -> None:
        env = clean_slurm_submit_env() if cmd and Path(cmd[0]).name == "sbatch" else dict(os.environ)
        completed = subprocess.run(
            cmd,
            cwd=self._repo_root(),
            env=env,
            text=True,
            capture_output=True,
        )
        (batch_dir / f"{step}.stdout.log").write_text(completed.stdout or "", encoding="utf-8")
        (batch_dir / f"{step}.stderr.log").write_text(completed.stderr or "", encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"Slurm+Docker queue {step} failed rc={completed.returncode}; "
                f"stdout={str(completed.stdout)[-1000:]}; stderr={str(completed.stderr)[-1000:]}"
            )

    def _benchmark_result_from_queue_payload(self, payload: dict[str, Any], *, fallback_name: str) -> BenchmarkResult:
        raw = payload.get("benchmark_result") or {}
        result = BenchmarkResult(
            xc_name=str(raw.get("xc_name") or fallback_name),
            timestamp=str(raw.get("timestamp") or datetime.now().isoformat()),
        )
        for subset_name, subset_payload in (raw.get("subsets") or {}).items():
            if not isinstance(subset_payload, dict):
                continue
            result.subsets[str(subset_name)] = SubsetResult.from_dict({"subset": subset_name, **subset_payload})
        self._compute_overall(result)
        return result

    def _fatal_benchmark_result(self, xc: XCFunctional, reason: str) -> BenchmarkResult:
        result = BenchmarkResult(xc_name=xc.name, timestamp=datetime.now().isoformat())
        for subset_name in self.subsets:
            result.subsets[subset_name] = SubsetResult(
                subset=subset_name,
                mae_kcal=0.0,
                rms_kcal=0.0,
                max_error_kcal=0.0,
                n_reactions=0,
                n_failures=1,
                walltime_s=0.0,
                per_reaction_errors=[],
                failure_details=[f"fatal:{reason}"],
            )
        self._compute_overall(result)
        return result

    def _load_cached_result(self, xc: XCFunctional) -> BenchmarkResult | None:
        """Return a complete precomputed result when every requested subset is available."""
        if self.cache_dir is None:
            return None
        path = None
        for candidate_path in self._cache_paths_for_xc(xc):
            if candidate_path.exists() and self._cache_path_matches_xc(candidate_path, xc):
                path = candidate_path
                break
        if path is None:
            path = self._lookup_cached_result_by_signature(xc)
        if path is None:
            return None
        try:
            raw = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            logger.warning("Unable to read benchmark cache for %s", xc.name)
            return None
        if any(subset not in raw for subset in self.subsets):
            return None
        result = BenchmarkResult(xc_name=xc.name, timestamp=datetime.now().isoformat())
        for subset in self.subsets:
            payload = raw.get(subset) or {}
            if payload.get("mae_kcal") is None:
                return None
            result.subsets[subset] = SubsetResult.from_dict(
                {
                    **payload,
                    "subset": subset,
                    "per_reaction_errors": [],
                }
            )
        self._compute_overall(result)
        logger.debug("Loaded cached benchmark result for %s from %s subsets=%s", xc.name, path, self.subsets)
        return result

    def _cache_path_matches_xc(self, path: Path, xc: XCFunctional) -> bool:
        """Reject same-name caches whose embedded XC is not this candidate."""
        try:
            raw = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        meta = raw.get("__meta__") if isinstance(raw, dict) else None
        if not isinstance(meta, dict):
            # Legacy same-name caches remain usable; signature-based recovery
            # should migrate them before ambiguous edited candidates are run.
            return path.stem == xc.name
        structural = str(meta.get("structural_signature", "") or "")
        effective = str(meta.get("effective_signature", "") or "")
        matches = (
            structural == xc.structural_signature()
            or effective == xc.effective_signature()
        )
        if not matches:
            logger.debug(
                "Skipping same-name benchmark cache with mismatched signature: candidate=%s path=%s",
                xc.name,
                path,
            )
        return matches

    def _compute_overall(self, result: BenchmarkResult) -> None:
        result.compute_overall(aggregate_mode=self.aggregate_mode)

    def _write_cached_result(self, result: BenchmarkResult, *, xc: XCFunctional | None = None) -> None:
        """Merge freshly computed subset summaries into the on-disk benchmark cache."""
        if self.cache_dir is None:
            return
        with self._cache_lock:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path = self._cache_path_for_xc(xc, fallback_name=result.xc_name)
            try:
                raw = json.loads(path.read_text()) if path.exists() else {}
            except (OSError, json.JSONDecodeError):
                raw = {}
            for subset, sr in result.subsets.items():
                raw[subset] = sr.to_dict(include_reaction_errors=False)
            if xc is not None:
                raw["__meta__"] = self._cache_metadata_for_xc(xc)
            path.write_text(json.dumps(raw, indent=2))
            self._cache_index = None
        logger.debug(
            "Updated benchmark cache for %s subsets=%s",
            self._xc_log_label(xc, fallback_name=result.xc_name),
            list(result.subsets),
        )

    def _run_subset(self, subset_name: str, xc: XCFunctional) -> SubsetResult:
        if self.execution_backend == "srun":
            return self._run_subset_via_slurm(subset_name, xc)
        return compute_subset_result(
            gmtkn55_dir=self.gmtkn55_dir,
            subset_name=subset_name,
            xc=xc,
            basis=self.basis,
            scf_conv_tol=self.scf_conv_tol,
            scf_max_cycle=self.scf_max_cycle,
            n_workers=self.n_workers,
            n_threads=self.n_threads,
            log_label=self._xc_log_label(xc),
        )

    def _run_subset_via_slurm(self, subset_name: str, xc: XCFunctional) -> SubsetResult:
        if self._slurm_pool_size() > 1:
            return self._run_subset_via_slurm_sharded(subset_name, xc)
        return self._run_subset_via_slurm_single(subset_name, xc)

    def _run_subset_via_slurm_single(self, subset_name: str, xc: XCFunctional) -> SubsetResult:
        allocation_id = self._ensure_slurm_allocation()
        output_dir = self.slurm.output_dir or (self.cache_dir / "slurm_logs" if self.cache_dir else Path("slurm_logs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_label = self._short_label_for_xc(xc)
        job_stub = f"{safe_label}__{subset_name}"
        task_dir_base = output_dir / "tasks"
        task_dir_base.mkdir(parents=True, exist_ok=True)
        repo_root = self._repo_root()
        with tempfile.TemporaryDirectory(prefix=f"{job_stub}_", dir=str(task_dir_base)) as tmpdir:
            tmp_path = Path(tmpdir)
            task_path = tmp_path / "task.json"
            result_path = tmp_path / "result.json"
            slurm_log_path = output_dir / f"{job_stub}.out"
            task_payload = {
                "gmtkn55_dir": str(self.gmtkn55_dir),
                "subset_name": subset_name,
                "xc_dict": xc.to_dict(),
                "basis": self.basis,
                "scf_conv_tol": self.scf_conv_tol,
                "scf_max_cycle": self.scf_max_cycle,
                "n_workers": self.n_workers,
                "n_threads": self.n_threads,
                "log_label": self._xc_log_label(xc),
            }
            task_path.write_text(json.dumps(task_payload, indent=2))
            cmd = [
                self.slurm.launcher,
                f"--jobid={allocation_id}",
                "--ntasks=1",
                "-N",
                str(self.slurm.nodes),
                "-c",
                str(self.slurm.cpus_per_task),
                "-J",
                self._sanitize_label(job_stub)[:128],
                "-o",
                str(slurm_log_path),
                *list(self.slurm.extra_args or []),
                sys.executable,
                "-m",
                "xc.benchmark.slurm_worker",
                "--task-file",
                str(task_path),
                "--output-file",
                str(result_path),
            ]
            env = os.environ.copy()
            src_dir = str(repo_root / "src")
            existing_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = src_dir if not existing_pythonpath else src_dir + os.pathsep + existing_pythonpath
            env.setdefault("DFTAGENT_COLOR", "0")
            logger.debug(
                "Submitting Slurm benchmark for %s subset=%s allocation=%s via %s",
                self._xc_log_label(xc),
                subset_name,
                allocation_id,
                " ".join(cmd[:10]),
            )
            completed = subprocess.run(
                cmd,
                cwd=repo_root,
                env=env,
                text=True,
                capture_output=True,
            )
            if completed.returncode != 0:
                details = [
                    f"slurm_launch_failed: returncode={completed.returncode}",
                    f"slurm_log: {slurm_log_path}",
                ]
                if completed.stderr.strip():
                    details.append(f"stderr: {completed.stderr.strip()[:500]}")
                logger.error(
                    "Slurm benchmark failed for %s subset=%s: %s",
                    self._xc_log_label(xc),
                    subset_name,
                    details,
                )
                return SubsetResult(
                    subset=subset_name,
                    mae_kcal=0.0,
                    rms_kcal=0.0,
                    max_error_kcal=0.0,
                    n_reactions=0,
                    n_failures=1,
                    walltime_s=0.0,
                    per_reaction_errors=[],
                    failure_details=details,
                )
            if not result_path.exists():
                details = [
                    "slurm_missing_result_file",
                    f"expected_result: {result_path}",
                    f"slurm_log: {slurm_log_path}",
                ]
                logger.error(
                    "Slurm benchmark produced no result for %s subset=%s",
                    self._xc_log_label(xc),
                    subset_name,
                )
                return SubsetResult(
                    subset=subset_name,
                    mae_kcal=0.0,
                    rms_kcal=0.0,
                    max_error_kcal=0.0,
                    n_reactions=0,
                    n_failures=1,
                    walltime_s=0.0,
                    per_reaction_errors=[],
                    failure_details=details,
                )
            payload = json.loads(result_path.read_text())
            return SubsetResult.from_dict(payload)

    def _run_subset_via_slurm_sharded(self, subset_name: str, xc: XCFunctional) -> SubsetResult:
        subset_dir = self.gmtkn55_dir / subset_name
        res_path = subset_dir / ".res"
        if not res_path.exists():
            return SubsetResult(
                subset=subset_name,
                mae_kcal=0.0,
                rms_kcal=0.0,
                max_error_kcal=0.0,
                n_reactions=0,
                n_failures=1,
                walltime_s=0.0,
                per_reaction_errors=[],
                failure_details=[f"fatal:missing_res_file: {res_path}"],
            )
        reactions = parse_res_file(res_path)
        if not reactions:
            return SubsetResult(
                subset=subset_name,
                mae_kcal=0.0,
                rms_kcal=0.0,
                max_error_kcal=0.0,
                n_reactions=0,
                n_failures=1,
                walltime_s=0.0,
                per_reaction_errors=[],
                failure_details=[f"fatal:no_reactions_parsed: {res_path}"],
            )

        mol_names = _molecule_names_from_reactions(reactions)
        slurm_parallelism = self._slurm_parallelism()
        shard_count = min(max(1, slurm_parallelism), len(mol_names))
        if shard_count <= 1:
            return self._run_subset_via_slurm_single(subset_name, xc)

        allocations = self._acquire_slurm_allocations(shard_count)
        output_dir = self.slurm.output_dir or (self.cache_dir / "slurm_logs" if self.cache_dir else Path("slurm_logs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        task_dir_base = output_dir / "tasks"
        task_dir_base.mkdir(parents=True, exist_ok=True)
        safe_label = self._short_label_for_xc(xc)
        repo_root = self._repo_root()
        t0 = time.time()
        mol_energies: dict[str, float | None] = {}
        mol_errors: dict[str, str] = {}
        processes: list[dict[str, Any]] = []

        try:
            shards = [mol_names[idx::shard_count] for idx in range(shard_count)]
            logger.info(
                "Submitting sharded Slurm benchmark for %s subset=%s shards=%d molecules=%d allocations=%s",
                self._xc_log_label(xc),
                subset_name,
                shard_count,
                len(mol_names),
                ",".join(str(getattr(item, "job_id", "") or item) for item in allocations),
            )
            with tempfile.TemporaryDirectory(prefix=f"{safe_label}__{subset_name}_sharded_", dir=str(task_dir_base)) as tmpdir:
                tmp_path = Path(tmpdir)
                env = os.environ.copy()
                src_dir = str(repo_root / "src")
                existing_pythonpath = env.get("PYTHONPATH", "")
                env["PYTHONPATH"] = src_dir if not existing_pythonpath else src_dir + os.pathsep + existing_pythonpath
                env.setdefault("DFTAGENT_COLOR", "0")

                # Launch all shard srun commands first. This gives Slurm the parallelism,
                # while keeping Python-side resource ownership deterministic and single-threaded.
                for shard_index, (allocation, shard_mols) in enumerate(zip(allocations, shards)):
                    if not shard_mols:
                        continue
                    allocation_id = str(getattr(allocation, "job_id", "") or allocation)
                    shard_stub = f"{safe_label}__{subset_name}__shard{shard_index + 1:02d}of{shard_count:02d}"
                    shard_dir = tmp_path / f"shard_{shard_index:02d}"
                    shard_dir.mkdir(parents=True, exist_ok=True)
                    task_path = shard_dir / "task.json"
                    result_path = shard_dir / "result.json"
                    slurm_log_path = output_dir / f"{shard_stub}.out"
                    task_payload = {
                        "mode": "molecule_batch",
                        "gmtkn55_dir": str(self.gmtkn55_dir),
                        "subset_name": subset_name,
                        "mol_names": shard_mols,
                        "xc_dict": xc.to_dict(),
                        "basis": self.basis,
                        "scf_conv_tol": self.scf_conv_tol,
                        "scf_max_cycle": self.scf_max_cycle,
                        "n_workers": min(self.n_workers, max(1, len(shard_mols))),
                        "n_threads": self.n_threads,
                        "log_label": self._xc_log_label(xc),
                    }
                    task_path.write_text(json.dumps(task_payload, indent=2))
                    cmd = [
                        self.slurm.launcher,
                        f"--jobid={allocation_id}",
                        "--ntasks=1",
                        "-N",
                        str(self.slurm.nodes),
                        "-c",
                        str(self.slurm.cpus_per_task),
                        "-J",
                        self._sanitize_label(shard_stub)[:128],
                        "-o",
                        str(slurm_log_path),
                        *list(self.slurm.extra_args or []),
                        sys.executable,
                        "-m",
                        "xc.benchmark.slurm_worker",
                        "--task-file",
                        str(task_path),
                        "--output-file",
                        str(result_path),
                    ]
                    logger.debug(
                        "Submitting Slurm benchmark shard for %s subset=%s shard=%d/%d allocation=%s mols=%d",
                        self._xc_log_label(xc),
                        subset_name,
                        shard_index + 1,
                        shard_count,
                        allocation_id,
                        len(shard_mols),
                    )
                    proc = subprocess.Popen(
                        cmd,
                        cwd=repo_root,
                        env=env,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    processes.append({
                        "process": proc,
                        "mols": shard_mols,
                        "result_path": result_path,
                        "slurm_log_path": slurm_log_path,
                    })

                for item in processes:
                    proc = item["process"]
                    stdout, stderr = proc.communicate()
                    shard_mols = item["mols"]
                    result_path = item["result_path"]
                    slurm_log_path = item["slurm_log_path"]
                    if proc.returncode != 0:
                        error = (
                            f"slurm_shard_failed: returncode={proc.returncode}; "
                            f"slurm_log={slurm_log_path}; stderr={str(stderr).strip()[:500]}"
                        )
                        for name in shard_mols:
                            mol_energies[name] = None
                            mol_errors[name] = error
                        continue
                    if not result_path.exists():
                        error = f"slurm_shard_missing_result_file: expected_result={result_path}; slurm_log={slurm_log_path}"
                        for name in shard_mols:
                            mol_energies[name] = None
                            mol_errors[name] = error
                        continue
                    payload = json.loads(result_path.read_text())
                    mol_energies.update(payload.get("mol_energies", {}))
                    mol_errors.update(payload.get("mol_errors", {}))
        finally:
            self._release_slurm_allocations(allocations)

        return _subset_result_from_molecule_energies(
            subset_name=subset_name,
            reactions=reactions,
            mol_energies=mol_energies,
            mol_errors=mol_errors,
            walltime_s=time.time() - t0,
            log_label=self._xc_log_label(xc),
        )[0]
    def _ensure_slurm_allocation(self) -> str:
        if self.slurm_allocation is None:
            raise RuntimeError("Srun backend requires a SlurmAllocation")
        if hasattr(self.slurm_allocation, "start"):
            return str(self.slurm_allocation.start())
        job_id = getattr(self.slurm_allocation, "job_id", None)
        if job_id:
            return str(job_id)
        raise RuntimeError("Invalid Slurm allocation object: missing start() or job_id")

    def _slurm_pool_size(self) -> int:
        return self._slurm_parallelism()

    def _slurm_parallelism(self) -> int:
        if self.slurm_allocation is None:
            return 0
        if hasattr(self.slurm_allocation, "parallelism"):
            return int(self.slurm_allocation.parallelism(
                max_allocations=self.slurm_allocations_per_evaluation
            ))
        if hasattr(self.slurm_allocation, "size"):
            return int(self.slurm_allocation.size())
        return 1

    def _slurm_ready_size(self) -> int:
        if self.slurm_allocation is None:
            return 0
        if hasattr(self.slurm_allocation, "ready_size"):
            return int(self.slurm_allocation.ready_size())
        if hasattr(self.slurm_allocation, "size"):
            return int(self.slurm_allocation.size())
        return 1

    def _slurm_requested_size(self) -> int:
        if self.slurm_allocation is None:
            return 0
        if hasattr(self.slurm_allocation, "requested_size"):
            return int(self.slurm_allocation.requested_size())
        return 1

    def _acquire_slurm_allocations(self, count: int) -> list[Any]:
        if self.slurm_allocation is None:
            raise RuntimeError("Srun backend requires a SlurmAllocation")
        requested = max(1, int(count))
        if hasattr(self.slurm_allocation, "acquire_many"):
            return list(self.slurm_allocation.acquire_many(requested))
        return [self._acquire_slurm_allocation()]

    def _release_slurm_allocations(self, allocations: list[Any]) -> None:
        if not allocations:
            return
        if self.slurm_allocation is not None and hasattr(self.slurm_allocation, "release_many"):
            self.slurm_allocation.release_many(allocations)
            return
        for allocation in allocations:
            self._release_slurm_allocation(allocation)
    def _acquire_slurm_allocation(self) -> Any:
        if self.slurm_allocation is None:
            raise RuntimeError("Srun backend requires a SlurmAllocation")
        if hasattr(self.slurm_allocation, "acquire"):
            return self.slurm_allocation.acquire()
        self._ensure_slurm_allocation()
        return self.slurm_allocation

    def _release_slurm_allocation(self, allocation: Any) -> None:
        if self.slurm_allocation is not None and hasattr(self.slurm_allocation, "release"):
            self.slurm_allocation.release(allocation)

    def close(self) -> None:
        if self.slurm_allocation is not None and hasattr(self.slurm_allocation, "close"):
            self.slurm_allocation.close()

    def _compute_molecule_energy(
        self, subset_dir: Path, mol_name: str, xc: XCFunctional
    ) -> float | None:
        outcome = _worker_compute_molecule(
            (str(subset_dir), mol_name, xc.to_dict(), self.basis,
             self.scf_conv_tol, self.scf_max_cycle)
        )
        return outcome.energy if outcome.ok else None

    def _compute_reaction_energy(
        self, subset_dir: Path, reaction: dict, xc: XCFunctional
    ) -> float | None:
        energy = 0.0
        for group, stoich in zip(reaction['species_groups'], reaction['stoich_coeffs']):
            for mol_name in group:
                e = self._compute_molecule_energy(subset_dir, mol_name, xc)
                if e is None:
                    return None
                energy += stoich * e
        return energy * HARTREE_TO_KCAL

    def _xc_log_label(self, xc: XCFunctional | None, *, fallback_name: str = "") -> str:
        if xc is None:
            return fallback_name or "unknown_xc"
        proposal = str(xc.metadata.get("proposal_id", "")).strip()
        skill = str(xc.metadata.get("skill", "")).strip()
        label = xc.name
        if proposal:
            label += f" [{proposal}]"
        if skill:
            label += f"<{skill}>"
        return label

    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    def _sanitize_label(self, value: str) -> str:
        keep = []
        for char in value:
            if char.isalnum() or char in {"-", "_", "."}:
                keep.append(char)
            else:
                keep.append("_")
        return "".join(keep).strip("_") or "benchmark_job"

    def _cache_paths_for_xc(self, xc: XCFunctional) -> list[Path]:
        if self.cache_dir is None:
            return []
        primary = self._cache_path_for_xc(xc)
        paths = [primary]
        legacy = self.cache_dir / f"{xc.name}.json"
        if legacy != primary and len(legacy.name.encode("utf-8")) < 240:
            paths.append(legacy)
        structural_sig = xc.structural_signature()
        sig_path = self.cache_dir / f"sig_{structural_sig}.json"
        if sig_path not in paths:
            paths.append(sig_path)
        effective_sig = xc.effective_signature()
        eff_path = self.cache_dir / f"eff_{effective_sig}.json"
        if eff_path not in paths:
            paths.append(eff_path)
        for existing in self.cache_dir.glob(f"*__{structural_sig}.json"):
            if existing not in paths:
                paths.append(existing)
        return paths

    def _cache_path_for_xc(self, xc: XCFunctional | None, *, fallback_name: str = "") -> Path:
        if self.cache_dir is None:
            raise RuntimeError("cache_dir is not configured")
        name = xc.name if xc is not None else (fallback_name or "unknown_xc")
        safe_stem = self._sanitize_cache_stem(name)
        if len(safe_stem.encode("utf-8")) <= 120 and safe_stem == name:
            return self.cache_dir / f"{safe_stem}.json"
        signature = (
            xc.structural_signature()
            if xc is not None
            else hashlib.sha256(name.encode()).hexdigest()[:16]
        )
        return self.cache_dir / f"{safe_stem[:80].rstrip('_')}__{signature}.json"

    def _sanitize_cache_stem(self, value: str) -> str:
        keep = []
        for char in value:
            if char.isalnum() or char in {"-", "_", ".", "="}:
                keep.append(char)
            else:
                keep.append("_")
        cleaned = re.sub(r"_+", "_", "".join(keep)).strip("_")
        return cleaned or "benchmark_result"

    def _short_label_for_xc(self, xc: XCFunctional, *, max_len: int = 96) -> str:
        label = self._sanitize_label(self._xc_log_label(xc).replace(" ", "_"))
        if len(label.encode("utf-8")) <= max_len:
            return label
        return f"{label[: max_len - 18].rstrip('_')}__{xc.structural_signature()}"

    def _cache_metadata_for_xc(self, xc: XCFunctional) -> dict[str, Any]:
        effective_spec = xc.effective_spec()
        return {
            "cache_schema": "benchmark-cache-v2",
            "xc_name": xc.name,
            "structural_signature": xc.structural_signature(),
            "effective_signature": xc.effective_signature(),
            "effective_spec": effective_spec,
            "effective_spec_hash": hashlib.sha256(
                json.dumps(effective_spec, sort_keys=True).encode()
            ).hexdigest()[:16],
            "xc_dict": xc.to_dict(),
        }

    def _lookup_cached_result_by_signature(self, xc: XCFunctional) -> Path | None:
        if self.cache_dir is None:
            return None
        index = self._build_cache_index()
        structural_sig = xc.structural_signature()
        effective_sig = xc.effective_signature()
        path = index.get("structural", {}).get(structural_sig)
        if path is not None:
            logger.debug(
                "Benchmark cache signature hit for %s: structural_sig=%s path=%s",
                xc.name,
                structural_sig,
                path,
            )
            return path
        path = index.get("effective", {}).get(effective_sig)
        if path is not None:
            logger.debug(
                "Benchmark cache signature hit for %s: effective_sig=%s path=%s",
                xc.name,
                effective_sig,
                path,
            )
            return path
        return None

    def _build_cache_index(self) -> dict[str, dict[str, Path]]:
        with self._cache_lock:
            if self._cache_index is not None:
                return self._cache_index
            index: dict[str, dict[str, Path]] = {"structural": {}, "effective": {}}
            if self.cache_dir is None or not self.cache_dir.exists():
                self._cache_index = index
                return index
            for path in sorted(self.cache_dir.glob("*.json")):
                structural_from_name = self._signature_from_cache_filename(path.name)
                if structural_from_name:
                    index["structural"].setdefault(structural_from_name, path)
                try:
                    raw = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                meta = raw.get("__meta__") if isinstance(raw, dict) else None
                if not isinstance(meta, dict):
                    continue
                structural = str(meta.get("structural_signature", "") or "")
                effective = str(meta.get("effective_signature", "") or "")
                if structural:
                    index["structural"].setdefault(structural, path)
                if effective:
                    index["effective"].setdefault(effective, path)
            self._cache_index = index
            return index

    def _signature_from_cache_filename(self, name: str) -> str:
        stem = name[:-5] if name.endswith(".json") else name
        match = re.search(r"__(?P<sig>[0-9a-f]{16})$", stem)
        if match:
            return match.group("sig")
        match = re.match(r"sig_(?P<sig>[0-9a-f]{16})$", stem)
        if match:
            return match.group("sig")
        return ""

    def validate_requested_subsets(self) -> None:
        if self.execution_backend == "srun" and self.n_workers * self.n_threads > self.slurm.cpus_per_task:
            logger.warning(
                "BenchmarkRunner configured for Slurm with n_workers*n_threads=%d exceeding cpus_per_task=%d",
                self.n_workers * self.n_threads,
                self.slurm.cpus_per_task,
            )
        missing: list[str] = []
        unparsable: list[str] = []
        for subset_name in self.subsets:
            subset_dir = self.gmtkn55_dir / subset_name
            res_path = subset_dir / ".res"
            if not res_path.exists():
                missing.append(str(res_path))
                continue
            try:
                reactions = parse_res_file(res_path)
            except Exception as exc:
                unparsable.append(f"{res_path} ({exc.__class__.__name__}: {exc})")
                continue
            if not reactions:
                unparsable.append(str(res_path))
        if missing or unparsable:
            parts: list[str] = []
            if missing:
                parts.append("missing .res files: " + ", ".join(missing))
            if unparsable:
                parts.append("unusable .res files: " + ", ".join(unparsable))
            raise FileNotFoundError(
                f"BenchmarkRunner dataset check failed for subsets {self.subsets}: " + "; ".join(parts)
            )

    def _should_cache_subset_result(self, sr: SubsetResult) -> bool:
        return sr.n_reactions > 0

    def _is_fatal_subset_failure(self, sr: SubsetResult) -> bool:
        return any(detail.startswith("fatal:") for detail in sr.failure_details)
