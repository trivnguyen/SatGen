
import os
import pickle
import sys
sys.path.append('/mnt/home/tnguyen/projects/florah/florah-tree')

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import time
import torch
import ml_collections
from tqdm import tqdm
from absl import flags, logging
from ml_collections import config_flags

# satgen modules
import config as cfg
import cosmo as co
import pytorch_utils
import init
from profiles import Dekel, NFW
import aux

# florah modules
import datasets

DEFAULT_METADATA_DIR = "/mnt/ceph/users/tnguyen/florah-tree/metadata"


def generate_tree(lgM0: float,
                  lgMres: float,
                  halo_response: str,
                  z0: float = 0.0,
                  seed: int = None):
    """
    Generate a single merger tree.

    Args:
        tree_id: Tree identifier
        lgM0_range: (lgM0_lo, lgM0_hi) - mass range for root halo
        lgMres: Log10 of resolution mass
        halo_response: 'NIHAO' or 'APOSTLE'
        z0: Initial redshift
        seed: Random seed (if None, uses tree_id)

    Returns:
        dict: Tree data and metadata
    """
    cfg.M0 = 10.**lgM0
    cfg.z0 = z0
    cfg.Mres = 10.**lgMres
    cfg.Mmin = 0.04 * cfg.Mres

    k = 0               # The level, k, of the branch being considered
    ik = 0              # How many level-k branches have been finished
    Nk = 1              # Total number of level-k branches
    Nbranch = 1         # Total number of branches in the current tree

    Mak = [cfg.M0]      # Accretion masses of level-k branches
    zak = [cfg.z0]      # Accretion redshifts of level-k branches
    idk = [0]           # Branch ids of level-k branches
    ipk = [-1]          # Parent ids of level-k branches (-1: no parent)

    Mak_tmp = []
    zak_tmp = []
    idk_tmp = []
    ipk_tmp = []

    mass = np.zeros((cfg.Nmax, cfg.Nz)) - 99.
    order = np.zeros((cfg.Nmax, cfg.Nz), np.int8) - 99
    ParentID = np.zeros((cfg.Nmax, cfg.Nz), np.int16) - 99

    VirialRadius = np.zeros((cfg.Nmax, cfg.Nz), np.float32) - 99.
    concentration = np.zeros((cfg.Nmax, cfg.Nz), np.float32) - 99.
    DekelConcentration = np.zeros((cfg.Nmax, cfg.Nz), np.float32) - 99.
    DekelSlope = np.zeros((cfg.Nmax, cfg.Nz), np.float32) - 99.

    StellarMass = np.zeros((cfg.Nmax, cfg.Nz)) - 99.
    StellarSize = np.zeros((cfg.Nmax, cfg.Nz), np.float32) - 99.

    coordinates = np.zeros((cfg.Nmax, cfg.Nz, 6), np.float32)

    # Loop over branches, until the full tree is completed.
    # Starting from the main branch, draw progenitor(s) using the
    # Parkinson+08 algorithm. When there are two progenitors, the less
    # massive one is the root of a new branch. We draw branches level by
    # level, i.e., When a new branch occurs, we record its root, but keep
    # finishing the current branch and all the branches of the same level
    # as the current branch, before moving on to the next-level branches.
    while True:
        M = [Mak[ik]]   # Mass history of current branch in fine timestep
        z = [zak[ik]]   # The redshifts of the mass history
        cfg.M0 = Mak[ik]  # Descendant mass
        cfg.z0 = zak[ik]  # Descendant redshift
        id = idk[ik]    # Branch id
        ip = ipk[ik]    # Parent id

        while cfg.M0 > cfg.Mmin:
            if cfg.M0 > cfg.Mres:
                zleaf = cfg.z0  # Update leaf redshift

            co.UpdateGlobalVariables(**cfg.cosmo)
            M1, M2, Np = co.DrawProgenitors(**cfg.cosmo)

            # Update descendant halo mass and descendant redshift
            cfg.M0 = M1
            cfg.z0 = cfg.zW_interp(cfg.W0 + cfg.dW)
            if cfg.z0 > cfg.zmax:
                break

            # Register next-level branches
            if Np > 1 and cfg.M0 > cfg.Mres:
                Mak_tmp.append(M2)
                zak_tmp.append(cfg.z0)
                idk_tmp.append(Nbranch)
                ipk_tmp.append(id)
                Nbranch += 1

            # Record the mass history at the original time resolution
            M.append(cfg.M0)
            z.append(cfg.z0)

        # Now that a branch is fully grown, do some book-keeping

        # Convert mass-history list to array
        M = np.array(M)
        z = np.array(z)

        # Downsample the fine-step mass history, M(z), onto the
        # coarser output timesteps, cfg.zsample
        Msample, zsample = aux.downsample(M, z, cfg.zsample)
        iz = aux.FindClosestIndices(cfg.zsample, zsample)
        izleaf = aux.FindNearestIndex(cfg.zsample, zleaf)

        # Compute halo structure throughout time on the coarse grid, up
        # to the leaf point
        t = co.t(z, cfg.h, cfg.Om, cfg.OL)
        c, a, c2, Rv = [], [], [], []
        for i in iz:
            if i > (izleaf + 1):
                break  # Only compute structure below leaf
            msk = z >= cfg.zsample[i]
            if True not in msk:
                break  # Safety check
            ci, ai, Msi, c2i, c2DMOi = init.Dekel_fromMAH(
                M[msk], t[msk], cfg.zsample[i], HaloResponse=halo_response)
            Rvi = init.Rvir(M[msk][0], Delta=200., z=cfg.zsample[i])
            c.append(ci)
            a.append(ai)
            c2.append(c2i)
            Rv.append(Rvi)
            if i == iz[0]:
                Ms = Msi

        # Safety check: dealing with rare cases where the
        # branch's root z[0] is close to the maximum redshift -- when
        # this happens, the mass history has only one element, and
        # z[0] can be slightly above cfg.zsample[i] for the very
        # first iteration, leaving the lists c, a, c2, Rv never updated
        if len(c) == 0:
            ci, ai, Msi, c2i, _  = init.Dekel_fromMAH(M, t, z[0], HaloResponse=halo_response)
            c.append(ci)
            a.append(ai)
            c2.append(c2i)
            Rv.append(init.Rvir(M[0], Delta=200., z=z[0]))
            Ms = Msi

        c = np.array(c)
        a = np.array(a)
        c2 = np.array(c2)
        Rv = np.array(Rv)
        Nc = len(c2)  # Length of a branch over which c2 is computed

        # Compute stellar size at the root of the branch, i.e., at the
        # accretion epoch (z[0])
        Re = init.Reff(Rv[0], c2[0])

        # Use the redshift id and parent-branch id to access the parent
        # branch's information at our current branch's accretion epoch,
        # in order to initialize the orbit
        if ip == -1:  # If the branch is the main branch
            xv = np.zeros(6)
        else:
            Mp = mass[ip, iz[0]]
            cp = DekelConcentration[ip, iz[0]]
            ap = DekelSlope[ip, iz[0]]
            hp = Dekel(Mp, cp, ap, Delta=200., z=zsample[0])
            eps = 1. / np.pi * np.arccos(1. - 2. * np.random.random())
            xv = init.orbit(hp, xc=1., eps=eps)

        # Update the arrays for output
        mass[id, iz] = Msample
        order[id, iz] = k
        ParentID[id, iz] = ip

        VirialRadius[id, iz[0]:iz[0] + Nc] = Rv
        concentration[id, iz[0]:iz[0] + Nc] = c2
        DekelConcentration[id, iz[0]:iz[0] + Nc] = c
        DekelSlope[id, iz[0]:iz[0] + Nc] = a

        StellarMass[id, iz[0]] = Ms
        StellarSize[id, iz[0]] = Re

        coordinates[id, iz[0], :] = xv

        # Check if all the level-k branches have been dealt with: if so,
        # i.e., if ik == Nk, proceed to the next level.
        ik += 1
        if ik == Nk:  # All level-k branches are done!
            Mak = Mak_tmp
            zak = zak_tmp
            idk = idk_tmp
            ipk = ipk_tmp
            Nk = len(Mak)
            ik = 0
            Mak_tmp = []
            zak_tmp = []
            idk_tmp = []
            ipk_tmp = []
            if Nk == 0:
                break  # Jump out of "while True" if no next-level branch
            k += 1  # Update level

    # Trim and output
    mass = mass[:id + 1, :]
    order = order[:id + 1, :]
    ParentID = ParentID[:id + 1, :]
    VirialRadius = VirialRadius[:id + 1, :]
    concentration = concentration[:id + 1, :]
    DekelConcentration = DekelConcentration[:id + 1, :]
    DekelSlope = DekelSlope[:id + 1, :]
    StellarMass = StellarMass[:id + 1, :]
    StellarSize = StellarSize[:id + 1, :]
    coordinates = coordinates[:id + 1, :, :]

    return dict(
        redshift=cfg.zsample,
        CosmicTime=cfg.tsample,
        mass=mass,
        order=order,
        ParentID=ParentID,
        VirialRadius=VirialRadius,
        concentration=concentration,
        DekelConcentration=DekelConcentration,
        DekelSlope=DekelSlope,
        StellarMass=StellarMass,
        StellarSize=StellarSize,
        coordinates=coordinates
    )

def main(main_config):
    """ Main function to generate and save merger trees. """

    # Read the simulation data
    # get the root features from the root data
    sim_data = datasets.read_dataset(
        dataset_name=main_config.data.name,
        dataset_root=main_config.data.root,
        index_start=main_config.data.index_file_start,
        max_num_files=main_config.data.num_files,
    )
    sim_data = sim_data[:main_config.num_max_trees]
    num_sim = len(sim_data)

    # get the root mass
    lgM0_arr = np.array([tree.x[0][0].numpy() for tree in sim_data])

    # Divide into multiple jobs, calculating index ranges
    job_size = len(lgM_arr) // main_config.num_job
    start = job_size * main_config.job_id
    end = job_size * (main_config.job_id + 1)
    if main_config.job_id == main_config.num_job - 1:
        end = len(lgM0_arr)
    lgM0_arr = lgM0_arr[start:end]

    # Start generating EPS merger trees
    tree_list = []

    for lgM0 in tqdm(lgM0_arr):
        data = generate_tree(
            lgM0=lgM0 / config.h,
            lgMres=main_config.lgMres / config.h,
            halo_response=main_config.halo_response,
        )
        pyg_data = pytorch_utils.satgen_to_pyg(data, sort=True)
        pyg_data = pytorch_utils.remove_nodes(
            pyg_data,
            mass_cut=main_config.mass_cut,
            concentration_cut=main_config.concentration_cut,
            max_num_prog=main_config.max_num_prog,
        )

        tree_list.append(pyg_data)

    # Write to file
    os.makedirs(main_config.outdir, exist_ok=True)
    outfile = os.path.join(main_config.outdir, f'halos.{main_config.job_id}.pkl')

    print(outfile)
    with open(outfile, 'wb') as f:
        pickle.dump(tree_list, f)

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file(
        "config",
        None,
        "File path to the training or sampling hyperparameter configuration.",
        lock_config=True,
    )
    # Parse flags
    FLAGS(sys.argv)

    # Start training run
    main(main_config=FLAGS.config)
