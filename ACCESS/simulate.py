from cryospax import RelionParticleDataset, RelionParticleParameterFile
from cryospax._simulate_particles import _simulate_images, _configure_simulation_fn
import cryojax_eo as cxeo
from tqdm import tqdm
import numpy as np
import argparse
import tomllib
from pathlib import Path
import logging
import cryojax.simulator as cxs
import equinox as eqx
from cryojax.jax_util import filter_bmap
from typing import Literal
from cryojax.ndimage import CircularCosineMask
from jaxtyping import Float, Array
import jax
from cryojax_eo.simulator._image_simulation import simulate_image_with_white_gaussian_noise
import jax.numpy as jnp
from datetime import datetime

from rotation import UniformPoseParameterFile, GaussianPerturbedPoseParameterFile
from cryojax.ndimage import make_frequency_grid


def get_expt(
        star_path,
        loads_envelop,
        mrc_folder_path,
        n_images_in_parallel,
        expt_dir,
):
    expt_dir.mkdir(parents=True, exist_ok=True)
    expt_param_file = RelionParticleParameterFile(
            path_to_starfile=star_path,
            options=dict(
                loads_envelope=loads_envelop,
                broadcasts_image_config=True,
            ),
        )

    relion_dataset = RelionParticleDataset(
        parameter_file=expt_param_file,
        path_to_relion_project=mrc_folder_path,
    )
    dataloader = cxeo.dataset.create_dataloader(
        relion_dataset,
        batch_size=n_images_in_parallel,
        shuffle=False,
    )
    expt_images = []
    for batch in tqdm(dataloader, desc="batches", leave=False):
        expt_images.append(batch["particle_stack"]["images"])

    expt_images = np.concat(expt_images, axis=0).astype(np.float32)
    #subtract mean is not included in David's
    expt_images = expt_images - np.mean(expt_images, axis=(-2, -1), keepdims=True)

    n_expt_images = expt_images.shape[0]
    logging.debug(f"expt images have shape {expt_images.shape}")
    expt_path = expt_dir / f"expt_images{n_expt_images}.npy"
    np.save(file=expt_path, arr=expt_images)

    ctf_grids = []
    for item in tqdm(expt_param_file, desc="Evaluating CTFs", leave=False):
        tt = item["transfer_theory"]
        image_config = item["image_config"]
        wavelength = image_config.wavelength_in_angstroms
        print(image_config)

        # FIX: Generate a true reciprocal-space frequency grid starting from corners
        freq_grid = make_frequency_grid(
            shape=image_config.shape, 
            grid_spacing=image_config.pixel_size,
            outputs_rfftfreqs=False
        )

        ctf_2d = tt.ctf(
            frequency_grid_in_angstroms=freq_grid,
            wavelength_in_angstroms=wavelength,
            amplitude_contrast_ratio=tt.amplitude_contrast_ratio,
            phase_shift=tt.phase_shift,
            outputs_exp=False,  
        )
        
        ctf_2d_centered = jnp.fft.fftshift(ctf_2d, axes=(-2, -1))
        ctf_grids.append(ctf_2d_centered)

    ctf_path = expt_dir / f"expt_ctfs{n_expt_images}.npy"
    np.save(file=ctf_path, arr=ctf_grids)


# from _run_ensemble_reweighting.py
@eqx.filter_jit
def _gmm_volume_to_voxel_grid(
    gmm_volume: cxs.GaussianMixtureVolume, image_config: cxs.BasicImageConfig
) -> Float[Array, " z_dim y_dim x_dim"]:
    box_size = image_config.shape[0]
    render_fn = cxs.GaussianMixtureRenderFn(
        (box_size, box_size, box_size), image_config.pixel_size
    )
    return render_fn(gmm_volume)


def get_sim(
        star_path,
        loads_envelop,
        pdb_path,
        sim_dir: Path,
        data_sign: Literal["dark-on-light", "light-on-dark"],
        atom_selection,
        noise_snr_range: list[float],
        images_per_file,
        batch_size,
        rotmode: Literal["star", "uniform", "gaussian"],
        n_rot_samples: int,
        rot_sigma_radians: float,
        seed: int = 0,

):
    #FOR NOW, NO ESTIMATE POSE SINCE PREVIOUS REWEIGHTING RESULTS DID NOT USE THAT EITHER
    sim_dir.mkdir(parents=True, exist_ok=True)
    expt_param_file =  RelionParticleParameterFile(
            path_to_starfile=star_path,
            options=dict(
                loads_envelope=loads_envelop,
                broadcasts_image_config=True,
            ),
        )
    key = jax.random.key(seed=seed)
    key_rot, key_snr, key_noise = jax.random.split(key, 3)
    if rotmode == 'uniform':
        expt_param_file = UniformPoseParameterFile(
            expt_param_file,
            n_rot_samples,
            key_rot,
        )
        print('used uniform sampling for rotation')
    elif rotmode == 'gaussian':
        expt_param_file = GaussianPerturbedPoseParameterFile(
            expt_param_file,
            n_rot_samples,
            key_rot,
            rot_sigma_radians,
        )
        print('used gaussian sampling for rotation')
    elif rotmode == 'star':
        print('used original star file poses')
    else:
        raise ValueError(f'unknown rotation mode as {rotmode}')

    image_sign = -1.0 if data_sign == "dark-on-light" else 1.0
    image_config = expt_param_file[0]["image_config"]
    # from _run_ensemble_reweighting.py
    mask = CircularCosineMask(
        coordinate_grid=expt_param_file[0]["image_config"].get_coordinate_grid(
            physical=False
        ),
        radius=image_config.shape[0] // 2,
        rolloff_width=1.0,
    )

    if Path(pdb_path).suffix in [".pdb", ".cif"]:
        gmm_volume = cxs.load_tabulated_volume(
            pdb_path,
            output_type=cxs.GaussianMixtureVolume,
            tabulation="peng",
            include_b_factors=True,
            selection_string=atom_selection,
            # pdb_options=dict(center=False),
        )
        voxel_grid = _gmm_volume_to_voxel_grid(gmm_volume, image_config)


    # for truncating noise,
    # since max_volume_repr_resolution default None and never specified in yaml, exclude this

    voxel_volume = cxs.FourierVoxelGridVolume.from_real_voxel_grid(voxel_grid)



    n_images = len(expt_param_file)

    constant_args = ((voxel_volume,), mask, image_sign)

    keys_per_image = jax.random.split(key_noise, n_images)
    ensemble_indices_per_image = jnp.zeros((n_images,), dtype=jnp.int32)
    snr_per_image = jax.random.uniform(
        key_snr,
        (len(expt_param_file),),
        minval=noise_snr_range[0],
        maxval=noise_snr_range[1],
    )
    per_particle_args = (
        keys_per_image,
        ensemble_indices_per_image,
        snr_per_image,
    )

    simulate_particle_stack_to_npy(
            parameter_file=expt_param_file,              # Pass parameter file directly
            simulate_fn=simulate_image_with_white_gaussian_noise,
            output_dir=sim_dir,                           # Directory where .npy files drop
            constant_args=constant_args,
            per_particle_args=per_particle_args,
            images_per_file=images_per_file,
            batch_size=batch_size,
        )
    
    
def simulate_particle_stack_to_npy(
    parameter_file,
    simulate_fn,
    output_dir: Path,
    constant_args=None,
    per_particle_args=None,
    batch_size: int | None = None,
    images_per_file: int | None = None,
):
    """Simulates particles and saves them directly as .npy files chunk by chunk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_particles = len(parameter_file)
    images_per_file = n_particles if images_per_file is None else images_per_file
    
    # Configure the JAX batch function
    simulate_batch_fn = _configure_simulation_fn(
        simulate_fn,
        batch_size,
        images_per_file,
    )
    
    n_iterations, remainder = (
        n_particles // images_per_file,
        n_particles % images_per_file,
    )
    
    for file_index in range(n_iterations):
        dataset_index = np.arange(
            file_index * images_per_file, (file_index + 1) * images_per_file, dtype=int
        )
        images, _ = _simulate_images(
            dataset_index,
            parameter_file,
            simulate_batch_fn,
            constant_args,
            per_particle_args,
        )
        
        #different from cryospax.simulate_particle_stack, save images as npy after use. 
        np_images = np.array(images)
        np.save(output_dir / f"sim_images{file_index}{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.npy", np_images)


    if remainder > 0:
        simulate_batch_fn = _configure_simulation_fn(simulate_fn, batch_size, remainder)
        dataset_index = np.arange(n_particles - remainder, n_particles, dtype=int)
        
        images, _ = _simulate_images(
            dataset_index,
            parameter_file,
            simulate_batch_fn,
            constant_args,
            per_particle_args,
        )
        
        np_images = np.array(images)
        np.save(output_dir / f"sim_images{file_index}{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.npy", np_images)



def _merge_config(
        args: argparse.Namespace,
        parser: argparse.ArgumentParser,
        # subparser_list: list[argparse.ArgumentParser] 
):
    if args.config is not None:
        with args.config.open("rb") as f:
            cfg = tomllib.load(f)
            parser.set_defaults(**cfg)

        #use this if there are subparsers, see example in 7880 project
        """
        for subparser in subparser_list:
            subparser.set_defaults(**cfg)
        """
        args = parser.parse_args()
    return args

def _parse_args():
    parser = argparse.ArgumentParser(prog="cryoweight.run")
    parser.add_argument("--mrc_folder_path", "-mrc", type=Path, default=None, 
                        help="the folder path of MRC (experimental) files")
    parser.add_argument("--pdb_path", "-pdb", type=Path, default=None,
                        help="the paths of PDB files, separated by spaces")
    parser.add_argument("--map_paths", "-map", type=Path, nargs="+", default=None,
                        help="the paths of .map (volume) files, separated by spaces")
    parser.add_argument("--star_path", "-star", type=Path, default=None,
                        help="the path of STAR files, separated by spaces")
    parser.add_argument("--config", "-cfg", type=Path, default=None,
                        help="the config of all customized parameters")
    parser.add_argument("--sim_dir", type=Path, default=None,
                        help="the directory where sim images are saved.")
    parser.add_argument("--expt_dir", type=Path, default=None,
                        help="the directory where expt images are saved.")
    parser.add_argument("--cc_dir", type=Path, nargs="+", default=None,
                        help="the directory where cc results are saved.")
    parser.add_argument("--no_b_factor", "-nob", action="store_false", dest="use_b_factor",
                        help="whether to use b factor in image generation, default is True (use b factor).")
    parser.add_argument("--manual_b_factor", "-b", default=None, type=float,
                        help="if specified, use this b factor for all atoms.")
    parser.add_argument("--sim_batch_size", "-sbatch", type=int, default=16,
                        help="the batch size for generating synthetic images, \
                        default = 16")
    parser.add_argument("--trans_stdev", "-vstd", type=float, default=20,
                        help="the translation standard deviation of the experimental images from simulated images\
                        default = 20")
    parser.add_argument("--cc_batch_size", "-cbatch", type=int, default=16,
                        help="the batch size for generating synthetic images, \
                        default = 16")
    parser.add_argument("--max_workers", "-workers", type=int, default=4,
                        help="the max workers during cc calculation (multitasking), \
                        default = 4")
    parser.add_argument("--n_chunks", "-chunks", type=int, default=None,
                        help="Optional: Max number of random .npy files to process per PDB folder. Uses all if not called.")
    parser.add_argument("--shard-index", "-shard", type=int, default=0,
                        help="shard index for distributed execution, starting from 0.")
    parser.add_argument("--total_shards", "-totshards", type=int, default=1, 
                        help="total number of shards for distributed execution")
    parser.add_argument("--expt_batch_size", "-ebatch", type=int, default=8,
                    help="the batch size for experimental images in reweighting, \
                    default = 8")
    parser.add_argument("--rot_batch_size", "-rbatch", type=int, default=16,
                        help="the batch size for (rotation of) simulated images in reweighting, \
                        default = 16")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--rotmode", choices=["star", "uniform", "gaussian"], default="star",
                        help="the rotation mode for sampling extra poses per particle, \
                        default = 'star', i.e., according to star file and no extra")
    parser.add_argument("--n_rot_samples", "-rot", type=int, default=2000,
                    help="the number of rotations per particle for simulating images, ignored when --rotmode = 'star'\
                    default = 2000")
    parser.add_argument("--rot_sigma_radians", "-sigma", type=float, default=0.1,
                    help="the sigma in gaussian sampling of rotations for simulating images, ignored when --rotmode is not 'gaussian'\
                    default = 0.1")
    
    
    
    args = parser.parse_args()
    args = _merge_config(args, parser)
    return args








    