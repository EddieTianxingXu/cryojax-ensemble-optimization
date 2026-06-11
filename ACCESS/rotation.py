import numpy as np
import jax 
from cryojax.rotations import SO3
import equinox as eqx
from cryojax.simulator import EulerAnglePose
import jax.numpy as jnp

def _vmap_sample_uniform(
        key,
        n,
) -> SO3:
    keys = jax.random.split(key, n)
    return jax.vmap(SO3.sample_uniform)(keys)

def _vmap_sample_tangent(
        key,
        n,
        sigma
):
    return jax.random.normal(key, (n, 3) * sigma)

def _so3_to_zyz(R: SO3):
    #SO3 to euler angle poses (zyz) in degrees
    rot_m = R.as_matrix()
    theta = jnp.arccos(jnp.clip(rot_m[2, 2], -1.0, 1.0))
    sin_t = jnp.sin(theta)
    gimbal = jnp.abs(sin_t) < 1e-7

    phi = jnp.where(gimbal, 0.0, jnp.arctan2(rot_m[2, 1],  rot_m[2, 0]))
    psi = jnp.where(gimbal, jnp.arctan2(-rot_m[0, 1], rot_m[0, 0]), jnp.arctan2(rot_m[1, 2], -rot_m[0, 2]))

    return jnp.rad2deg(phi), jnp.rad2deg(theta), jnp.rad2deg(psi)

def _zyz_to_so3(pose: EulerAnglePose) -> SO3:
    #euler angles (zyz) in degrees to SO3
    phi = jnp.deg2rad(pose.phi_angle)
    psi = jnp.deg2rad(pose.psi_angle)
    theta = jnp.deg2rad(pose.theta_angle)

    return SO3.from_z_radians(phi).compose(SO3.from_z_radians(theta)).compose(SO3.from_z_radians(psi))

def _get_rotation(item: dict) -> SO3:
    return _zyz_to_so3(item['pose'])

def _set_rotation(
        item,
        phi, 
        theta,
        psi,
        ) -> dict:


    new_pose = eqx.tree_at(
        lambda pose: (pose.phi_angle, pose.theta_angle, pose.psi_angle ),
        item['pose'],
        (phi, theta, psi)
    )

    return eqx.tree_at(lambda t: t['pose'], item, new_pose)

class UniformPoseParameterFile:
    """Wraps a parameter file, replacing each particle's pose with a
    uniformly sampled SO(3) rotation.

    The wrapped file has length  len(base) * n_samples_per_particle.
    Layout: all n_samples for particle 0, then particle 1, …
    i.e. index  i  →  base particle  i // n_samples,  sample  i % n_samples.
    """

    def __init__(
            self,
            base_file,
            n_samples_per_particle: int,
            key: jax.Array,
    ):
        self._base = base_file
        self._n = n_samples_per_particle
        n_total = len(base_file) * n_samples_per_particle

        all_rotations = _vmap_sample_uniform(key, n_total)

        self._wxyz = np.array(all_rotations.wxyz)
    
    def __len__(self):
        return len(self._base) * self._n
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    
    def __getitem__(self, idx: int) -> dict:
        is_scalar = np.ndim(idx) == 0
        idx_arr = np.atleast_1d(np.asarray(idx))
        base_idx = int(idx // self._n) if is_scalar else (idx_arr // self._n)
        item = self._base[base_idx]
        wxyz = jnp.asarray(self._wxyz[idx_arr])

        phi, theta, psi = jax.vmap(
            lambda w: _so3_to_zyz(SO3(wxyz=w))
        )(wxyz)  

        if is_scalar:
            phi, theta, psi = phi[0], theta[0], psi[0]

        return _set_rotation(
            item,
            phi,
            theta,
            psi
        )
    
class GaussianPerturbedPoseParameterFile:
    """Wraps a parameter file, replacing each particle's pose with a
    rotation drawn from a Gaussian around the existing STAR orientation.

    Perturbation is done in the Lie algebra (tangent space):
        R_new = R_star  ∘  exp(ξ),   ξ ~ N(0, sigma^2 · I₃)
    where sigma is in radians.

    sigma ≈ 0.05 rad (~3°) is a mild perturbation;
    sigma ≈ 0.30 rad (~17°) covers a broader neighbourhood.

    The wrapped file has length  len(base) * n_samples_per_particle,
    with the same layout as UniformPoseParameterFile.
    """

    def __init__(
            self,
            base_file,
            n_samples_per_particle: int,
            key: jax.Array,
            sigma_radians: float
    ):
        self._base = base_file
        self._n = n_samples_per_particle
        n_total = len(base_file) * n_samples_per_particle

        tangents = _vmap_sample_tangent(key, n_total, sigma_radians)

        self._tangents = np.array(tangents)

        base_rotations = [_get_rotation(f) for f in base_file]
        self._base_wxyz = np.array([r.wxyz for r in base_rotations])
        
    def __len__(self):
        return len(self._base) * self._n
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx: int) -> dict:
        is_scalar = np.ndim(idx) == 0
        idx_arr = np.atleast_1d(np.asarray(idx))

        base_idx = int(idx // self._n) if is_scalar else (idx_arr // self._n)
        item = self._base[base_idx]

        base_wxyz = jnp.asarray(self._base_wxyz[idx_arr // self._n])

        tangents = jnp.asarray(self._tangents[idx_arr])

        def perturb(w, t):
            return SO3(wxyz=w).compose(SO3.exp(t))
        
        perturbed = jax.vmap(perturb)(base_wxyz, tangents) 

        phi, theta, psi = jax.vmap(
            lambda w: _so3_to_zyz(SO3(wxyz=w))
        )(perturbed.wxyz)  

        if is_scalar:
            phi, theta, psi = phi[0], theta[0], psi[0]

        return _set_rotation(
            item,
            phi,
            theta,
            psi
        )