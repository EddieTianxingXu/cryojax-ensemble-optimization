import pytest
import numpy as np
import jax
import jax.numpy as jnp
from cryojax.rotations import SO3
from cryojax.simulator import EulerAnglePose
from tqdm import tqdm
from rotation import GaussianPerturbedPoseParameterFile, UniformPoseParameterFile
import copy

# Minimal Mock for base_file that mimics a dataset of particle dicts
class MockBaseFile:
    def __init__(self, poses):
        self.poses = poses
    def __len__(self):
        return len(self.poses)
    def __getitem__(self, idx):
        # Return a deep copy so mutating item['pose'] doesn't corrupt the base dataset
        return copy.deepcopy({'pose': self.poses[int(idx)]})

@pytest.fixture
def identity_pose():
    return EulerAnglePose(phi_angle=0.0, theta_angle=0.0, psi_angle=0.0)

@pytest.fixture
def random_pose():
    return EulerAnglePose(phi_angle=45.0, theta_angle=60.0, psi_angle=120.0)


def test_uniform_pose_parameter_file(identity_pose):
    """Tests if UniformPoseParameterFile fills SO(3) space uniformly."""
    key = jax.random.PRNGKey(42)
    n_samples = 1000  # Large N for statistical power
    base_file = MockBaseFile([identity_pose])
    
    wrapper = UniformPoseParameterFile(base_file, n_samples_per_particle=n_samples, key=key)
    
    # Extract all generated matrices
    matrices = []
    for i in tqdm(range(len(wrapper))):
        item = wrapper[i]
        # Reconstruct the SO3 object from the ZYZ degrees to verify the round-trip
        phi = jnp.deg2rad(item['pose'].phi_angle)
        theta = jnp.deg2rad(item['pose'].theta_angle)
        psi = jnp.deg2rad(item['pose'].psi_angle)
        R = SO3.from_z_radians(phi).compose(SO3.from_y_radians(theta)).compose(SO3.from_z_radians(psi))
        matrices.append(np.array(R.as_matrix()))
        
    matrices = np.array(matrices)
    mean_matrix = np.mean(matrices, axis=0)
    
    # Statistical expectation of a uniform distribution over SO(3) is the zero matrix
    # With N=10000, standard error is roughly 1/sqrt(10000) = 0.01. atol=0.05 is safe.
    np.testing.assert_allclose(mean_matrix, 0.0, atol=0.05, 
                               err_msg="The sampled rotations are not uniformly distributed.")


@pytest.mark.parametrize("sigma_degrees", [3.0, 10.0])
def test_gaussian_perturbed_pose_parameter_file(random_pose, sigma_degrees):
    """Tests if GaussianPerturbedPoseParameterFile perturbs poses by the correct sigma."""
    key = jax.random.PRNGKey(24)
    n_samples = 5000
    sigma_radians = np.deg2rad(sigma_degrees)
    
    base_file = MockBaseFile([random_pose])
    wrapper = GaussianPerturbedPoseParameterFile(
        base_file, 
        n_samples_per_particle=n_samples, 
        key=key, 
        sigma_radians=sigma_radians
    )
    
    # Get the original base rotation matrix
    phi_base = jnp.deg2rad(random_pose.phi_angle)
    theta_base = jnp.deg2rad(random_pose.theta_angle)
    psi_base = jnp.deg2rad(random_pose.psi_angle)
    R_base = SO3.from_z_radians(phi_base).compose(SO3.from_y_radians(theta_base)).compose(SO3.from_z_radians(psi_base))
    R_base_mat = np.array(R_base.as_matrix())
    
    squared_angles = []
    
    for i in tqdm(range(len(wrapper))):
        item = wrapper[i]
        # Reconstruct perturbed matrix
        phi = jnp.deg2rad(item['pose'].phi_angle)
        theta = jnp.deg2rad(item['pose'].theta_angle)
        psi = jnp.deg2rad(item['pose'].psi_angle)
        R_perturbed = SO3.from_z_radians(phi).compose(SO3.from_y_radians(theta)).compose(SO3.from_z_radians(psi))
        R_pert_mat = np.array(R_perturbed.as_matrix())
        
        # Calculate the relative rotation matrix: delta_R = R_base^T * R_perturbed
        delta_R = np.dot(R_base_mat.T, R_pert_mat)
        
        # Extract the rotation angle theta from the trace: trace(R) = 1 + 2*cos(theta)
        trace = np.trace(delta_R)
        cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
        theta_val = np.arccos(cos_theta)
        
        squared_angles.append(theta_val ** 2)
        
    # For a 3D Gaussian perturbation vector, E[theta^2] should equal 3 * sigma^2
    empirical_mean_squared_angle = np.mean(squared_angles)
    expected_mean_squared_angle = 3 * (sigma_radians ** 2)
    
    # Allow a small relative tolerance due to sample variance
    np.testing.assert_allclose(empirical_mean_squared_angle, expected_mean_squared_angle, rtol=0.1,
                               err_msg=f"The perturbation variance does not match sigma={sigma_degrees}°")