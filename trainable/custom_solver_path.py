from sionna.rt.solver_paths import SolverPaths
from sionna.rt.solver_paths import PathsTmpData
from sionna.rt.paths import Paths
from sionna.utils.tensors import expand_to_rank, insert_dims, flatten_dims, split_dim
from sionna.rt.utils import (
    dot,
    phi_hat,
    theta_hat,
    theta_phi_from_unit_vec,
    normalize,
    component_transform,
    compute_field_unit_vectors,
    cross,
    sign,
    acos_diff,
)
from sionna.rt.solver_base import SolverBase
from sionna import PI

import tensorflow as tf


class CustomSolverPaths(SolverPaths):
    def _spec_transition_matrices_interaction(self, paths, paths_tmp, scattering):
        """
        Compute the transfer matrices, delays, angles of departures, and angles
        of arrivals, of paths from a set of valid reflection paths.

        Input
        ------

        paths : :class:`~sionna.rt.Paths`
            Paths to update

        paths_tmp : :class:`~sionna.rt.PathsTmpData`
            Addtional quantities required for paths computation

        scattering : bool
            Set to `True` if computing the scattered paths.

        Output
        -------
        mat_t : [num_targets, num_sources, max_num_paths, 2, 2], tf.complex
                Specular transition matrix for every path.
        """

        vertices = paths.vertices
        targets = paths.targets
        sources = paths.sources
        objects = paths.objects
        theta_t = paths.theta_t
        phi_t = paths.phi_t
        theta_r = paths.theta_r
        phi_r = paths.phi_r

        normals = paths_tmp.normals
        k_i = paths_tmp.k_i
        k_r = paths_tmp.k_r
        if scattering:
            # For scattering, only the distance up to the last intersection
            # point is considered for path loss.
            # [num_targets, num_sources, max_num_paths]
            total_distance = paths_tmp.scat_src_2_last_int_dist
        else:
            # [num_targets, num_sources, max_num_paths]
            total_distance = paths_tmp.total_distance

        # Maximum depth
        max_depth = tf.shape(vertices)[0]
        # Number of targets
        num_targets = tf.shape(targets)[0]
        # Number of sources
        num_sources = tf.shape(sources)[0]
        # Maximum number of paths
        max_num_paths = tf.shape(objects)[3]

        # Flag that indicates if a ray is valid
        # [max_depth, num_targets, num_sources, max_num_paths]
        valid_ray = tf.not_equal(objects, -1)
        # Pad to enable detection of the last valid reflection for scattering
        # [max_depth+1, num_targets, num_sources, max_num_paths]
        valid_ray = tf.pad(
            valid_ray, [[0, 1], [0, 0], [0, 0], [0, 0]], constant_values=False
        )

        # Compute e_i_s, e_i_p, e_r_s, e_r_p at each reflection point
        # all : [max_depth, num_targets, num_sources, max_num_paths,3]
        # pylint: disable=unbalanced-tuple-unpacking
        e_i_s, e_i_p, e_r_s, e_r_p = compute_field_unit_vectors(
            k_i[:max_depth], k_r, normals, SolverBase.EPSILON
        )

        # Compute cos(theta) at each reflection point
        # [max_depth, num_targets, num_sources, max_num_paths]
        cos_theta = -dot(k_i[:max_depth], normals, clip=True)

        # geo_feature = tf.stack([k_i[:max_depth], normals], -2)
        # geo_feature = k_i[:max_depth] * normals
        geo_feature = tf.stack([k_i[:max_depth], normals], -2)

        theta_ = acos_diff(tf.abs(cos_theta)) / PI
        theta_ = tf.expand_dims(theta_, axis=-1)

        # [max_depth, num_targets, num_sources, max_num_paths]
        transfer_mtx = self._scene.gen_spec_trans_mtx_NN(objects, vertices, theta_, 0)
        r_s, r_p = transfer_mtx[..., 0, 0], transfer_mtx[..., 1, 1]

        # Compute the field transfer matrix.
        # It is initialized with the identity matrix of size 2 (S and P
        # polarization components)
        # [num_targets, num_sources, max_num_paths, 2, 2]
        mat_t = tf.eye(
            num_rows=2,
            batch_shape=[num_targets, num_sources, max_num_paths],
            dtype=self._dtype,
        )
        # Initialize last field unit vector with outgoing ones
        # [num_targets, num_sources, max_num_paths, 3]
        last_e_r_s = theta_hat(theta_t, phi_t)
        last_e_r_p = phi_hat(phi_t)
        for depth in tf.range(0, max_depth):

            # Is this a valid reflection?
            # [num_targets, num_sources, max_num_paths]
            valid = valid_ray[depth]

            # Is the next reflection valid?
            # [num_targets, num_sources, max_num_paths]
            next_valid = valid_ray[depth + 1]
            # Expand for broadcasting
            # [num_targets, num_sources, max_num_paths, 1, 1]
            next_valid = insert_dims(next_valid, 2)

            # Early stopping if no active rays
            if not tf.reduce_any(valid):
                break

            # Add dimension for broadcasting with coordinates
            # [num_targets, num_sources, max_num_paths, 1]
            valid_ = tf.expand_dims(valid, axis=-1)

            # Change of basis matrix
            # [num_targets, num_sources, max_num_paths, 2, 2]
            mat_cob = component_transform(
                last_e_r_s, last_e_r_p, e_i_s[depth], e_i_p[depth]
            )
            mat_cob = tf.complex(mat_cob, tf.zeros_like(mat_cob))
            # Only apply transform if valid reflection
            # [num_targets, num_sources, max_num_paths, 1, 1]
            valid__ = tf.expand_dims(valid_, axis=-1)
            # [num_targets, num_sources, max_num_paths, 2, 2]
            e = tf.where(valid__, tf.linalg.matmul(mat_cob, mat_t), mat_t)
            # Only update ongoing direction for next iteration if this
            # reflection is valid and if this is not the last step
            last_e_r_s = tf.where(valid_, e_r_s[depth], last_e_r_s)
            last_e_r_p = tf.where(valid_, e_r_p[depth], last_e_r_p)

            # If scattering, then the reduction coefficient is not applied
            # to the last interaction as the outgoing ray is diffusely
            # reflected and not specularly reflected
            if scattering:
                # [num_targets, num_sources, max_num_paths, 1, 1]
                valid_interaction_mask = tf.logical_and(valid__, next_valid)
            else:
                valid_interaction_mask = valid__

            # Fresnel coefficients
            # [num_targets, num_sources, max_num_paths, 2]
            r = tf.stack([r_s[depth], r_p[depth]], -1)
            # Add a dimension to broadcast with mat_t
            # [num_targets, num_sources, max_num_paths, 2, 1]
            r = tf.expand_dims(r, axis=-1)
            # Set the coefficients to one if non-valid reflection
            # [num_targets, num_sources, max_num_paths, 2]
            r = tf.where(valid_interaction_mask, r, tf.ones_like(r))
            # Apply Fresnel coefficient
            # [num_targets, num_sources, max_num_paths, 2, 2]
            mat_t = r * e

        # Move to the targets frame
        # This is not done for scattering as we stop the last interaction point
        if not scattering:
            # Transformation matrix
            # [num_targets, num_sources, max_num_paths, 2, 2]
            mat_cob = component_transform(
                last_e_r_s, last_e_r_p, theta_hat(theta_r, phi_r), phi_hat(phi_r)
            )
            mat_cob = tf.complex(mat_cob, tf.zeros_like(mat_cob))
            # Apply transformation
            # [num_targets, num_sources, max_num_paths, 2, 2]
            mat_t = tf.linalg.matmul(mat_cob, mat_t)

        # Divide by total distance to account for propagation loss
        # [num_targets, num_sources, max_num_paths, 1, 1]
        total_distance = expand_to_rank(total_distance, tf.rank(mat_t), axis=3)
        total_distance = tf.complex(total_distance, tf.zeros_like(total_distance))
        # [num_targets, num_sources, max_num_paths, 2, 2]
        mat_t = tf.math.divide_no_nan(mat_t, total_distance)

        # Set invalid paths to 0 and stores the transition matrices
        # Expand masks to broadcast with the field components
        # [num_targets, num_sources, max_num_paths, 1, 1]
        mask_ = expand_to_rank(paths.mask, 5, axis=3)
        # Zeroing coefficients corresponding to non-valid paths
        # [num_targets, num_sources, max_num_paths, 2, 2]
        mat_t = tf.where(mask_, mat_t, tf.zeros_like(mat_t))

        return mat_t

    def _diffraction_transition_matrices_interaction(self, paths, paths_tmp):
        """
        Compute the transition matrices for diffracted rays.

        Input
        ------
        paths : :class:`~sionna.rt.Paths`
            Paths to update

        paths_tmp : :class:`~sionna.rt.PathsTmpData`
            Addtional quantities required for paths computation

        Output
        ------
        paths : :class:`~sionna.rt.Paths`
            Updated paths
        """

        mask = paths.mask
        targets = paths.targets
        sources = paths.sources
        theta_t = paths.theta_t
        phi_t = paths.phi_t
        theta_r = paths.theta_r
        phi_r = paths.phi_r

        normals = paths_tmp.normals

        wavelength = self._scene.wavelength
        k = 2.0 * PI / wavelength

        # [num_targets, num_sources, max_num_paths, 3]
        diff_points = paths.vertices[0]
        # [num_targets, num_sources, max_num_paths]
        wedges_indices = paths.objects[0]

        # On CPU, indexing with -1 does not work. Hence we replace -1 by 0.
        # This makes no difference on the resulting paths as such paths
        # are not flagged as active.
        # [num_targets, num_sources, max_num_paths]
        valid_wedges_idx = tf.where(wedges_indices == -1, 0, wedges_indices)

        # Normals
        # [num_targets, num_sources, max_num_paths, 2, 3]
        normals = tf.gather(self._wedges_normals, valid_wedges_idx, axis=0)

        # Compute the wedges angle
        # [num_targets, num_sources, max_num_paths]
        cos_wedges_angle = dot(normals[..., 0, :], normals[..., 1, :], clip=True)
        wedges_angle = PI - tf.math.acos(cos_wedges_angle)
        n = (2.0 * PI - wedges_angle) / PI

        # [num_targets, num_sources, max_num_paths, 3]
        e_hat = tf.gather(self._wedges_e_hat, valid_wedges_idx)

        # Reshape sources and targets
        # [1, num_sources, 1, 3]
        sources = tf.reshape(sources, [1, -1, 1, 3])
        # [num_targets, 1, 1, 3]
        targets = tf.reshape(targets, [-1, 1, 1, 3])

        # Extract surface normals
        # [num_targets, num_sources, max_num_paths, 3]
        n_0_hat = normals[..., 0, :]

        # [num_targets, num_sources, max_num_paths, 2]
        objects_indices = tf.gather(self._wedges_objects, valid_wedges_idx, axis=0)

        # Compute s_prime_hat, s_hat, s_prime, s
        # s_prime_hat : [num_targets, num_sources, max_num_paths, 3]
        # s_prime : [num_targets, num_sources, max_num_paths]
        s_prime_hat, s_prime = normalize(diff_points - sources)
        # s_hat : [num_targets, num_sources, max_num_paths, 3]
        # s : [num_targets, num_sources, max_num_paths]
        s_hat, s = normalize(targets - diff_points)

        # Compute phi_prime_hat, beta_0_prime_hat, phi_hat, beta_0_hat
        # [num_targets, num_sources, max_num_paths, 3]
        phi_prime_hat, _ = normalize(cross(s_prime_hat, e_hat))
        # [num_targets, num_sources, max_num_paths, 3]
        beta_0_prime_hat = cross(phi_prime_hat, s_prime_hat)

        # [num_targets, num_sources, max_num_paths, 3]
        phi_hat_, _ = normalize(-cross(s_hat, e_hat))
        beta_0_hat = cross(phi_hat_, s_hat)

        # Compute tangent vector t_0_hat
        # [num_targets, num_sources, max_num_paths, 3]
        t_0_hat = cross(n_0_hat, e_hat)

        # Compute s_t_prime_hat and s_t_hat
        # [num_targets, num_sources, max_num_paths, 3]
        s_t_prime_hat, _ = normalize(
            s_prime_hat - dot(s_prime_hat, e_hat, keepdim=True) * e_hat
        )
        # [num_targets, num_sources, max_num_paths, 3]
        s_t_hat, _ = normalize(s_hat - dot(s_hat, e_hat, keepdim=True) * e_hat)

        # Compute phi_prime and phi
        # [num_targets, num_sources, max_num_paths]
        phi_prime = PI - (PI - acos_diff(-dot(s_t_prime_hat, t_0_hat))) * sign(
            -dot(s_t_prime_hat, n_0_hat)
        )
        # [num_targets, num_sources, max_num_paths]
        phi = PI - (PI - acos_diff(dot(s_t_hat, t_0_hat))) * sign(dot(s_t_hat, n_0_hat))

        # Compute field component vectors for reflections at both surfaces
        # [num_targets, num_sources, max_num_paths, 3]
        e_i_s_0, e_i_p_0, e_r_s_0, e_r_p_0 = compute_field_unit_vectors(
            s_prime_hat,
            s_hat,
            n_0_hat,  # *sign(-dot(s_t_prime_hat, n_0_hat, keepdim=True)),
            SolverBase.EPSILON,
        )

        cos_beta_prime = dot(s_prime_hat, e_hat)
        cos_beta = dot(s_hat, e_hat)

        cos_beta_prime = tf.where(cos_beta_prime >= 0, cos_beta_prime, -cos_beta_prime)
        cos_beta = tf.where(cos_beta_prime >= 0, cos_beta, -cos_beta)

        beta_prime = acos_diff(cos_beta_prime) / (2.0 * PI)
        beta = acos_diff(cos_beta) / (2.0 * PI)

        phi_prime_ = phi_prime / (2.0 * PI)
        phi_ = phi / (2.0 * PI)

        n_ = n / 2.0

        s_prime_ = s_prime / (tf.math.sqrt(30.0**2 * 2.0 + 3.2**2))
        s_ = s / (tf.math.sqrt(30.0**2 * 2.0 + 3.2**2))

        geo_feature_dis = tf.stack(
            [phi_prime_, phi_, beta_prime, beta, n_, s_prime_, s_], -1
        )

        # [max_depth, num_targets, num_sources, max_num_paths]
        mat_no_scale = self._scene.gen_spec_trans_mtx_NN(
            objects_indices[..., 0], diff_points, geo_feature_dis, 1
        )

        # Compute matrices R_0, R_n
        # [num_targets, num_sources, max_num_paths, 2, 2]
        w_i_0 = component_transform(phi_prime_hat, beta_0_prime_hat, e_i_s_0, e_i_p_0)
        w_i_0 = tf.complex(w_i_0, tf.zeros_like(w_i_0))
        # [num_targets, num_sources, max_num_paths, 2, 2]
        w_r_0 = component_transform(e_r_s_0, e_r_p_0, phi_hat_, beta_0_hat)
        w_r_0 = tf.complex(w_r_0, tf.zeros_like(w_r_0))

        d_mul = 1 / tf.cast(tf.sqrt(2 * PI * k), self._dtype)

        # [num_targets, num_sources, max_num_paths]
        spreading_factor = tf.sqrt(1.0 / (s * s_prime * (s_prime + s)))
        spreading_factor = tf.complex(spreading_factor, tf.zeros_like(spreading_factor))
        mul = d_mul * spreading_factor
        mul = tf.expand_dims(mul, -1)
        mul = tf.expand_dims(mul, -1)

        mat_t = mat_no_scale * mul

        # Convert from/to GCS
        theta_t = paths.theta_t
        phi_t = paths.phi_t
        theta_r = paths.theta_r
        phi_r = paths.phi_r

        mat_from_gcs = component_transform(
            theta_hat(theta_t, phi_t), phi_hat(phi_t), phi_prime_hat, beta_0_prime_hat
        )
        mat_from_gcs = tf.complex(mat_from_gcs, tf.zeros_like(mat_from_gcs))

        mat_to_gcs = component_transform(
            phi_hat_, beta_0_hat, theta_hat(theta_r, phi_r), phi_hat(phi_r)
        )
        mat_to_gcs = tf.complex(mat_to_gcs, tf.zeros_like(mat_to_gcs))

        mat_t = tf.linalg.matmul(mat_t, mat_from_gcs)
        mat_t = tf.linalg.matmul(mat_to_gcs, mat_t)

        # Set invalid paths to 0
        # Expand masks to broadcast with the field components
        # [num_targets, num_sources, max_num_paths, 1, 1]
        mask_ = expand_to_rank(mask, 5, axis=3)
        # Zeroing coefficients corresponding to non-valid paths
        # [num_targets, num_sources, max_num_paths, 2]
        mat_t = tf.where(mask_, mat_t, tf.zeros_like(mat_t))

        return mat_t

    def _compute_paths_coefficients_interaction(
        self, rx_rot_mat, tx_rot_mat, paths, paths_tmp, num_samples, scat_keep_prob
    ):
        r"""
        Computes the paths coefficients.

        Input
        ------
        rx_rot_mat : [num_rx, 3, 3], tf.float
            Matrices for rotating according to the receivers orientations

        tx_rot_mat : [num_tx, 3, 3], tf.float
            Matrices for rotating according to the receivers orientations

        paths : :class:`~sionna.rt.Paths`
            Paths to update

        paths_tmp : :class:`~sionna.rt.PathsTmpData`
            Updated addtional quantities required for paths computation

        num_samples : int
            Number of random rays to trace in order to generate candidates.
            A large sample count may exhaust GPU memory.

        scat_keep_prob : float
            Probability with which to keep scattered paths.
            This is helpful to reduce the number of scattered paths computed,
            which might be prohibitively high in some setup.
            Must be in the range (0,1).

        Output
        ------
        paths : :class:`~sionna.rt.Paths`
            Updated paths
        """

        # [num_rx, num_tx, max_num_paths, 2, 2]
        theta_t = paths.theta_t
        phi_t = paths.phi_t
        theta_r = paths.theta_r
        phi_r = paths.phi_r
        types = paths.types

        mat_t = paths_tmp.mat_t
        k_tx = paths_tmp.k_tx
        k_rx = paths_tmp.k_rx

        # Apply multiplication by wavelength/4pi
        # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths,2, 2]
        cst = tf.cast(self._scene.wavelength / (4.0 * PI), self._dtype)
        a = cst * mat_t

        # Get dimensions that are needed later on
        num_rx = a.shape[0]
        rx_array_size = a.shape[1]
        num_tx = a.shape[2]
        tx_array_size = a.shape[3]

        # Expand dimension for broadcasting with receivers/transmitters,
        # antenna dimensions, and paths dimensions
        # [1, 1, num_tx, 1, 1, 3, 3]
        tx_rot_mat = insert_dims(insert_dims(tx_rot_mat, 2, 0), 2, 3)
        # [num_rx, 1, 1, 1, 1, 3, 3]
        rx_rot_mat = insert_dims(rx_rot_mat, 4, 1)

        if self._scene.synthetic_array:
            # Expand for broadcasting with antenna dimensions
            # [num_rx, 1, num_tx, 1, max_num_paths, 3]
            k_rx = tf.expand_dims(tf.expand_dims(k_rx, axis=1), axis=3)
            k_tx = tf.expand_dims(tf.expand_dims(k_tx, axis=1), axis=3)
            # [num_rx, 1, num_tx, 1, max_num_paths]
            theta_t = tf.expand_dims(tf.expand_dims(theta_t, axis=1), axis=3)
            phi_t = tf.expand_dims(tf.expand_dims(phi_t, axis=1), axis=3)
            theta_r = tf.expand_dims(tf.expand_dims(theta_r, axis=1), axis=3)
            phi_r = tf.expand_dims(tf.expand_dims(phi_r, axis=1), axis=3)

        # Normalized wave transmit vector in the local coordinate system of
        # the transmitters
        # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths, 3]
        k_prime_t = tf.linalg.matvec(tx_rot_mat, k_tx, transpose_a=True)

        # Normalized wave receiver vector in the local coordinate system of
        # the receivers
        # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths, 3]
        k_prime_r = tf.linalg.matvec(rx_rot_mat, k_rx, transpose_a=True)

        # Angles of departure in the local coordinate system of the
        # transmitter
        # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths, 3]
        theta_prime_t, phi_prime_t = theta_phi_from_unit_vec(k_prime_t)

        # Angles of arrival in the local coordinate system of the
        # receivers
        # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths, 3]
        theta_prime_r, phi_prime_r = theta_phi_from_unit_vec(k_prime_r)

        # Spherical global frame vectors for tx and rx
        # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths, 3]
        theta_hat_t = theta_hat(theta_t, phi_t)
        phi_hat_t = phi_hat(phi_t)
        theta_hat_r = theta_hat(theta_r, phi_r)
        phi_hat_r = phi_hat(phi_r)

        # Spherical local frame vectors for tx and rx
        # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths, 3]
        theta_hat_prime_t = theta_hat(theta_prime_t, phi_prime_t)
        phi_hat_prime_t = phi_hat(phi_prime_t)
        theta_hat_prime_r = theta_hat(theta_prime_r, phi_prime_r)
        phi_hat_prime_r = phi_hat(phi_prime_r)

        # Rotation matrix for going from the spherical LCS to the spherical GCS
        # For transmitters
        # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths]
        tx_lcs2gcs_11 = dot(
            theta_hat_t, tf.linalg.matvec(tx_rot_mat, theta_hat_prime_t)
        )
        tx_lcs2gcs_12 = dot(theta_hat_t, tf.linalg.matvec(tx_rot_mat, phi_hat_prime_t))
        tx_lcs2gcs_21 = dot(phi_hat_t, tf.linalg.matvec(tx_rot_mat, theta_hat_prime_t))
        tx_lcs2gcs_22 = dot(phi_hat_t, tf.linalg.matvec(tx_rot_mat, phi_hat_prime_t))
        # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths,2, 2]
        tx_lcs2gcs = tf.stack(
            [
                tf.stack([tx_lcs2gcs_11, tx_lcs2gcs_12], axis=-1),
                tf.stack([tx_lcs2gcs_21, tx_lcs2gcs_22], axis=-1),
            ],
            axis=-2,
        )
        tx_lcs2gcs = tf.complex(tx_lcs2gcs, tf.zeros_like(tx_lcs2gcs))
        # For receivers
        # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths]
        rx_lcs2gcs_11 = dot(
            theta_hat_r, tf.linalg.matvec(rx_rot_mat, theta_hat_prime_r)
        )
        rx_lcs2gcs_12 = dot(theta_hat_r, tf.linalg.matvec(rx_rot_mat, phi_hat_prime_r))
        rx_lcs2gcs_21 = dot(phi_hat_r, tf.linalg.matvec(rx_rot_mat, theta_hat_prime_r))
        rx_lcs2gcs_22 = dot(phi_hat_r, tf.linalg.matvec(rx_rot_mat, phi_hat_prime_r))
        # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths,2, 2]
        rx_lcs2gcs = tf.stack(
            [
                tf.stack([rx_lcs2gcs_11, rx_lcs2gcs_12], axis=-1),
                tf.stack([rx_lcs2gcs_21, rx_lcs2gcs_22], axis=-1),
            ],
            axis=-2,
        )
        rx_lcs2gcs = tf.complex(rx_lcs2gcs, tf.zeros_like(rx_lcs2gcs))

        # List of antenna patterns (callables)
        tx_patterns = self._scene.tx_array.antenna.patterns
        rx_patterns = self._scene.rx_array.antenna.patterns

        tx_ant_fields_hat = []
        for pattern in tx_patterns:
            # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size,  max_num_paths, 2]
            tx_ant_f = tf.stack(pattern(theta_prime_t, phi_prime_t), axis=-1)
            tx_ant_fields_hat.append(tx_ant_f)

        rx_ant_fields_hat = []
        for pattern in rx_patterns:
            # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths, 2]
            rx_ant_f = tf.stack(pattern(theta_prime_r, phi_prime_r), axis=-1)
            rx_ant_fields_hat.append(rx_ant_f)

        # Stacking the patterns, corresponding to different polarization directions, as an additional dimension
        # [num_rx, num_rx_patterns, 1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths, 2]
        rx_ant_fields_hat = tf.stack(rx_ant_fields_hat, axis=1)
        # Expand for broadcasting with tx polarization
        # [num_rx, num_rx_patterns, 1/rx_array_size, num_tx, 1, 1, 1/tx_array_size, max_num_paths, 2]
        rx_ant_fields_hat = tf.expand_dims(rx_ant_fields_hat, axis=4)

        # Stacking the patterns, corresponding to different polarization
        # [num_rx, 1/rx_array_size, num_tx, num_tx_patterns, 1/tx_array_size, max_num_paths, 2]
        tx_ant_fields_hat = tf.stack(tx_ant_fields_hat, axis=3)
        # Expand for broadcasting with rx polarization
        # [num_rx, 1, 1/rx_array_size, num_tx, num_tx_patterns, 1/tx_array_size, max_num_paths, 2]
        tx_ant_fields_hat = tf.expand_dims(tx_ant_fields_hat, axis=1)

        # Antenna patterns to spherical global coordinate system
        # Expand to broadcast with antenna patterns
        # [num_rx, 1, 1/rx_array_size, num_tx, 1, 1/tx_array_size, max_num_paths, 2, 2]
        rx_lcs2gcs = tf.expand_dims(tf.expand_dims(rx_lcs2gcs, axis=1), axis=4)
        # [num_rx, num_rx_patterns, 1/rx_array_size, num_tx, 1, 1/tx_array_size, max_num_paths, 2]
        rx_ant_fields = tf.linalg.matvec(rx_lcs2gcs, rx_ant_fields_hat)
        # Expand to broadcast with antenna patterns
        # [num_rx, 1, 1/rx_array_size, num_tx, 1, 1/tx_array_size, max_num_paths, 2, 2]
        tx_lcs2gcs = tf.expand_dims(tf.expand_dims(tx_lcs2gcs, axis=1), axis=4)
        # [num_rx, 1, 1/rx_array_size, num_tx, num_tx_patterns, 1/tx_array_size,
        #   max_num_paths, 2, 2]
        tx_ant_fields = tf.linalg.matvec(tx_lcs2gcs, tx_ant_fields_hat)

        # Expand the field to broadcast with the antenna patterns
        # [num_rx, 1, rx_array_size, num_tx, 1, tx_array_size, max_num_paths, 2, 2]
        a = tf.expand_dims(tf.expand_dims(a, axis=1), axis=4)

        # Compute transmitted field
        # [num_rx, 1, 1/rx_array_size, num_tx, num_tx_patterns, 1/tx_array_size, max_num_paths, 2]
        a = tf.linalg.matvec(a, tx_ant_fields)

        # Scattering: For scattering, a is the field specularly reflected by the last interaction point. We need to compute the scattered field.
        # [num_scat_paths]
        scat_ind = tf.where(types == Paths.SCATTERED)[:, 0]
        n_scat = tf.size(scat_ind)
        if n_scat > 0:
            n_other = a.shape[-2] - n_scat
            # On CPU, indexing with -1 does not work. Hence we replace -1 by 0.
            # This makes no difference on the resulting paths as such paths are not flagged as active.
            # [max_num_paths]
            # valid_object_idx = tf.where(paths_tmp.scat_last_objects == -1, 0, paths_tmp.scat_last_objects)
            valid_object_idx = paths_tmp.scat_last_objects
            scat_points = paths_tmp.scat_last_vertices

            field_vec_shape = tf.concat([tf.shape(valid_object_idx), [2]], axis=0)
            field_vec = tf.ones(field_vec_shape, dtype=self._rdtype)
            field_vec = tf.complex(field_vec, tf.cast(0, self._rdtype))

            # Complete the computation of the field
            # The term cos(theta_i)*dA is equal to 4*PI/N*r^2
            # [num_targets, num_sources, max_num_paths]
            num_samples = tf.cast(num_samples, self._rdtype)
            scaling = tf.sqrt(
                4 * tf.cast(PI, self._rdtype) / (scat_keep_prob * num_samples)
            )
            scaling *= paths_tmp.scat_src_2_last_int_dist

            # Apply path loss due to propagation from scattering point to target
            # [num_targets, num_sources, max_num_paths]
            scaling = tf.math.divide_no_nan(scaling, paths_tmp.scat_2_target_dist)

            # Compute scaled field vector
            # [num_targets, num_sources, max_num_paths, 2]
            field_vec *= tf.expand_dims(tf.complex(scaling, tf.zeros_like(scaling)), -1)

            # [num_targets, num_sources, max_num_paths, 3]
            e_i_s, e_i_p = compute_field_unit_vectors(
                paths_tmp.scat_last_k_i,
                paths_tmp.scat_k_s,
                paths_tmp.scat_last_normals,
                SolverBase.EPSILON,
                return_e_r=False,
            )

            # a_scat : [num_rx, 1, rx_array_size, num_tx, num_tx_patterns, tx_array_size, n_scat, 2]
            # a_other : [num_rx, 1, rx_array_size, num_tx, num_tx_patterns, tx_array_size, max_num_paths - n_scat, 2]
            a_other, a_scat_in = tf.split(a, [n_other, n_scat], axis=-2)

            # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths, 3]
            _, scat_theta_hat_r = tf.split(theta_hat_r, [n_other, n_scat], axis=-2)
            # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths, 3]
            _, scat_phi_hat_r = tf.split(phi_hat_r, [n_other, n_scat], axis=-2)

            # Compute incoming field
            # [num_rx, 1, 1/rx_array_size, num_tx, 1, 1/tx_array_size, n_scat, (3)]
            scat_k_i = paths_tmp.scat_last_k_i
            scat_k_s = paths_tmp.scat_k_s
            scat_normal = paths_tmp.scat_last_normals
            if self._scene.synthetic_array:
                e_i_s = insert_dims(e_i_s, 2, axis=1)
                e_i_s = insert_dims(e_i_s, 2, axis=4)
                e_i_p = insert_dims(e_i_p, 2, axis=1)
                e_i_p = insert_dims(e_i_p, 2, axis=4)
                scat_k_i = insert_dims(scat_k_i, 2, axis=1)
                scat_k_i = insert_dims(scat_k_i, 2, axis=4)
                scat_k_s = insert_dims(scat_k_s, 2, axis=1)
                scat_k_s = insert_dims(scat_k_s, 2, axis=4)
                scat_normal = insert_dims(scat_normal, 2, axis=1)
                scat_normal = insert_dims(scat_normal, 2, axis=4)
                scat_points = insert_dims(scat_points, 2, axis=1)
                scat_points = insert_dims(scat_points, 2, axis=4)
                valid_object_idx = insert_dims(valid_object_idx, 2, axis=1)
                valid_object_idx = insert_dims(valid_object_idx, 2, axis=4)
                field_vec = insert_dims(field_vec, 2, axis=1)
                field_vec = insert_dims(field_vec, 2, axis=4)
            else:
                num_rx = len(self._scene.receivers)
                num_tx = len(self._scene.transmitters)

                e_i_s = split_dim(e_i_s, [num_rx, -1], 0)
                e_i_s = tf.expand_dims(e_i_s, axis=1)
                e_i_s = split_dim(e_i_s, [num_tx, -1], 3)
                e_i_s = tf.expand_dims(e_i_s, axis=4)
                e_i_p = split_dim(e_i_p, [num_rx, -1], 0)
                e_i_p = tf.expand_dims(e_i_p, axis=1)
                e_i_p = split_dim(e_i_p, [num_tx, -1], 3)
                e_i_p = tf.expand_dims(e_i_p, axis=4)

                scat_k_i = split_dim(scat_k_i, [num_rx, -1], 0)
                scat_k_i = tf.expand_dims(scat_k_i, axis=1)
                scat_k_i = split_dim(scat_k_i, [num_tx, -1], 3)
                scat_k_i = tf.expand_dims(scat_k_i, axis=4)
                scat_k_s = split_dim(scat_k_s, [num_rx, -1], 0)
                scat_k_s = tf.expand_dims(scat_k_s, axis=1)
                scat_k_s = split_dim(scat_k_s, [num_tx, -1], 3)
                scat_k_s = tf.expand_dims(scat_k_s, axis=4)
                scat_normal = split_dim(scat_normal, [num_rx, -1], 0)
                scat_normal = tf.expand_dims(scat_normal, axis=1)
                scat_normal = split_dim(scat_normal, [num_tx, -1], 3)
                scat_normal = tf.expand_dims(scat_normal, axis=4)
                scat_points = split_dim(scat_points, [num_rx, -1], 0)
                scat_points = tf.expand_dims(scat_points, axis=1)
                scat_points = split_dim(scat_points, [num_tx, -1], 3)
                scat_points = tf.expand_dims(scat_points, axis=4)
                valid_object_idx = split_dim(valid_object_idx, [num_rx, -1], 0)
                valid_object_idx = tf.expand_dims(valid_object_idx, axis=1)
                valid_object_idx = split_dim(valid_object_idx, [num_tx, -1], 3)
                valid_object_idx = tf.expand_dims(valid_object_idx, axis=4)
                field_vec = split_dim(field_vec, [num_rx, -1], 0)
                field_vec = tf.expand_dims(field_vec, axis=1)
                field_vec = split_dim(field_vec, [num_tx, -1], 3)
                field_vec = tf.expand_dims(field_vec, axis=4)

            a_in_s, a_in_p = tf.split(a_scat_in, 2, axis=-1)

            a_scat_in_rt_pwr = tf.math.sqrt(
                tf.math.abs(a_in_s) ** 2 + tf.math.abs(a_in_p) ** 2
            )
            a_scat_in_rt_pwr = tf.complex(
                a_scat_in_rt_pwr, tf.zeros_like(a_scat_in_rt_pwr)
            )
            a_in_s = tf.math.divide_no_nan(a_in_s, a_scat_in_rt_pwr, name=None)
            a_in_p = tf.math.divide_no_nan(a_in_p, a_scat_in_rt_pwr, name=None)

            a_in_s_real = tf.math.real(a_in_s)
            a_in_s_imag = tf.math.imag(a_in_s)
            a_in_p_real = tf.math.real(a_in_p)
            a_in_p_imag = tf.math.imag(a_in_p)
            a_in_all = tf.concat(
                [a_in_s_real, a_in_s_imag, a_in_p_real, a_in_p_imag], -1
            )

            geo_feature_dis = tf.concat([a_in_all, scat_k_i, scat_k_s, scat_normal], -1)

            valid_object_idx = tf.squeeze(valid_object_idx, axis=[1, 2, 3])
            scat_points = tf.squeeze(scat_points, axis=[1, 2, 3])
            geo_feature_dis = tf.squeeze(geo_feature_dis, axis=[1, 2, 3])

            a_scat_out = self._scene.gen_spec_trans_mtx_NN(
                valid_object_idx, scat_points, geo_feature_dis, 2
            )
            a_scat_out = a_scat_out[..., 0]
            a_scat_out = insert_dims(a_scat_out, 3, axis=1)
            a_scat_out = a_scat_out * a_scat_in_rt_pwr

            # Compute polarization field vector
            e_i_s = tf.complex(e_i_s, tf.zeros_like(e_i_s))
            e_i_p = tf.complex(e_i_p, tf.zeros_like(e_i_p))
            e_in_pol = a_in_s * e_i_s + a_in_p * e_i_p
            e_pol_hat, _ = normalize(tf.math.real(e_in_pol))
            e_xpol_hat = cross(e_pol_hat, scat_k_i)

            # Compute incoming spherical unit vectors in GCS
            scat_theta_i, scat_phi_i = theta_phi_from_unit_vec(-scat_k_i)
            scat_theta_hat_i = theta_hat(scat_theta_i, scat_phi_i)
            scat_phi_hat_i = phi_hat(scat_phi_i)

            # Transformation to theta_hat_i, phi_hat_i
            trans_mat = component_transform(
                e_pol_hat, e_xpol_hat, scat_theta_hat_i, scat_phi_hat_i
            )
            trans_mat = tf.complex(trans_mat, tf.zeros_like(trans_mat))

            # Transformation from theta_hat_s, phi_hat_s to theta_hat_r, phi_hat_r
            # [num_targets, num_sources, max_num_paths, 3]
            scat_theta_s, scat_phi_s = theta_phi_from_unit_vec(paths_tmp.scat_k_s)
            scat_theta_hat_s = theta_hat(scat_theta_s, scat_phi_s)
            scat_phi_hat_s = phi_hat(scat_phi_s)

            # [num_rx, 1/rx_array_size, num_sources, max_num_paths, 3]
            scat_theta_hat_s = split_dim(scat_theta_hat_s, [num_rx, rx_array_size], 0)
            scat_phi_hat_s = split_dim(scat_phi_hat_s, [num_rx, rx_array_size], 0)

            # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths, 3]
            scat_theta_hat_s = split_dim(scat_theta_hat_s, [num_tx, tx_array_size], 2)
            scat_phi_hat_s = split_dim(scat_phi_hat_s, [num_tx, tx_array_size], 2)

            # [num_rx, 1,  1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths, 3]
            scat_theta_hat_s = tf.expand_dims(scat_theta_hat_s, 1)
            scat_phi_hat_s = tf.expand_dims(scat_phi_hat_s, 1)

            # [num_rx, 1,  1/rx_array_size, num_tx, 1, 1/tx_array_size, max_num_paths, 3]
            scat_theta_hat_s = tf.expand_dims(scat_theta_hat_s, 4)
            scat_phi_hat_s = tf.expand_dims(scat_phi_hat_s, 4)

            # [num_rx, 1, 1/rx_array_size, num_tx, 1, 1/tx_array_size, max_num_scat_paths, 3]
            scat_theta_hat_r = tf.expand_dims(scat_theta_hat_r, axis=1)
            scat_theta_hat_r = tf.expand_dims(scat_theta_hat_r, axis=4)
            # [num_rx, 1, 1/rx_array_size, num_tx, 1, 1/tx_array_size, max_num_scat_paths, 3]
            scat_phi_hat_r = tf.expand_dims(scat_phi_hat_r, axis=1)
            scat_phi_hat_r = tf.expand_dims(scat_phi_hat_r, axis=4)

            trans_mat2 = component_transform(
                scat_theta_hat_s, scat_phi_hat_s, scat_theta_hat_r, scat_phi_hat_r
            )
            trans_mat2 = tf.complex(trans_mat2, tf.zeros_like(trans_mat2))

            a_scat_out = tf.linalg.matvec(trans_mat2, a_scat_out)
            a_scat_out = field_vec * a_scat_out

            # Concat with other paths
            a = tf.concat([a_other, a_scat_out], axis=-2)

        # [num_rx, num_rx_patterns, 1/rx_array_size, num_tx, num_tx_patterns, 1/tx_array_size, max_num_paths]
        a = dot(rx_ant_fields, a)

        if not self._scene.synthetic_array:
            # Reshape as expected to merge antenna and antenna patterns into one
            # dimension, as expected by Sionna
            # [ num_rx, num_rx_ant = num_rx_patterns*rx_array_size,
            #   num_tx, num_tx_ant = num_tx_patterns*tx_array_size,
            #   max_num_paths]
            a = flatten_dims(flatten_dims(a, 2, 1), 2, 3)

        return a

    def compute_fields_interaction_modeling(
        self,
        spec_paths,
        diff_paths,
        scat_paths,
        spec_paths_tmp,
        diff_paths_tmp,
        scat_paths_tmp,
        testing,
    ):
        r"""
        Computes the EM fields for a set of traced paths.

        Input
        ------
        spec_paths : Paths
            Specular paths

        diff_paths : Paths
            Diffracted paths

        scat_paths : Paths
            Scattered paths

        spec_paths_tmp : PathsTmpData
            Additional data required to compute the EM fields of the specular
            paths

        diff_paths_tmp : PathsTmpData
            Additional data required to compute the EM fields of the diffracted
            paths

        scat_paths_tmp : PathsTmpData
            Additional data required to compute the EM fields of the scattered
            paths

        testing : bool
            If set to `True`, then additional data is returned for testing.

        Output
        -------
        sources : [num_sources, 3], tf.float
            Coordinates of the sources

        targets : [num_targets, 3], tf.float
            Coordinates of the targets

        list : Paths as a list
            The computed paths as a dictionnary of tensors, i.e., the output of
            `Paths.to_dict()`.
            Returning the paths as a list of tensors is required to enable
            the execution of this function in graph mode.

        list : PathsTmpData as a list
            Additional data required to compute the EM fields of the specular
            paths as list of tensors.
            Only returned if `testing` is set to `True`.

        list : PathsTmpData as a list
            Additional data required to compute the EM fields of the diffracted
            paths as list of tensors.
            Only returned if `testing` is set to `True`.

        list : PathsTmpData as a list
            Additional data required to compute the EM fields of the scattered
            paths as list of tensors.
            Only returned if `testing` is set to `True`.
        """

        sources = spec_paths.sources
        targets = spec_paths.targets

        # Create empty paths object
        all_paths = Paths(sources=sources, targets=targets, scene=self._scene)
        # Create empty objects for storing tensors that are required to compute
        # paths, but that will not be returned to the user
        all_paths_tmp = PathsTmpData(sources, targets, self._dtype)

        # Rotation matrices corresponding to the orientations of the radio
        # devices
        # rx_rot_mat : [num_rx, 3, 3]
        # tx_rot_mat : [num_tx, 3, 3]
        rx_rot_mat, tx_rot_mat = self._get_tx_rx_rotation_matrices()

        # Number of receive antennas (not counting for dual polarization)
        tx_array_size = self._scene.tx_array.array_size
        # Number of transmit antennas (not counting for dual polarization)
        rx_array_size = self._scene.rx_array.array_size

        ##############################################
        # LoS and Specular paths
        ##############################################

        if spec_paths.objects.shape[3] > 0:

            # Compute the EM transition matrices
            spec_mat_t = self._spec_transition_matrices_interaction(
                spec_paths, spec_paths_tmp, False
            )
            all_paths = all_paths.merge(spec_paths)
            # Only the transition matrix and vector of incidence/reflection are required for the computation of the paths coefficients
            all_paths_tmp.mat_t = tf.concat([all_paths_tmp.mat_t, spec_mat_t], axis=-3)
            all_paths_tmp.k_tx = tf.concat(
                [all_paths_tmp.k_tx, spec_paths_tmp.k_tx], axis=-2
            )
            all_paths_tmp.k_rx = tf.concat(
                [all_paths_tmp.k_rx, spec_paths_tmp.k_rx], axis=-2
            )
            # If testing, the transition matrices are also returned
            if testing:
                spec_paths_tmp.mat_t = spec_mat_t

        ############################################
        # Diffracted paths
        ############################################

        if diff_paths.objects.shape[3] > 0:

            # Compute the transition matrices
            diff_mat_t = self._diffraction_transition_matrices_interaction(
                diff_paths, diff_paths_tmp
            )
            all_paths = all_paths.merge(diff_paths)
            # Only the transition matrix and vector of incidence/reflection are
            # required for the computation of the paths coefficients
            all_paths_tmp.mat_t = tf.concat([all_paths_tmp.mat_t, diff_mat_t], axis=-3)
            all_paths_tmp.k_tx = tf.concat(
                [all_paths_tmp.k_tx, diff_paths_tmp.k_tx], axis=-2
            )
            all_paths_tmp.k_rx = tf.concat(
                [all_paths_tmp.k_rx, diff_paths_tmp.k_rx], axis=-2
            )
            # If testing, the transition matrices are also returned
            if testing:
                diff_paths_tmp.mat_t = diff_mat_t

        ############################################
        # Scattered paths
        ############################################

        if scat_paths.objects.shape[3] > 0:

            # Compute transition matrices up to the scattering point
            scat_mat_t = self._spec_transition_matrices_interaction(
                scat_paths, scat_paths_tmp, True
            )

            all_paths = all_paths.merge(scat_paths)
            # The transition matrix and vector of incidence/reflection are
            # required for the computation of the paths coefficients, as well
            # as other scattering specific quantities.
            all_paths_tmp.mat_t = tf.concat([all_paths_tmp.mat_t, scat_mat_t], axis=-3)
            all_paths_tmp.k_tx = tf.concat(
                [all_paths_tmp.k_tx, scat_paths_tmp.k_tx], axis=-2
            )
            all_paths_tmp.k_rx = tf.concat(
                [all_paths_tmp.k_rx, scat_paths_tmp.k_rx], axis=-2
            )
            all_paths_tmp.scat_last_objects = scat_paths_tmp.scat_last_objects
            all_paths_tmp.scat_last_k_i = scat_paths_tmp.scat_last_k_i
            all_paths_tmp.scat_k_s = scat_paths_tmp.scat_k_s
            all_paths_tmp.scat_last_normals = scat_paths_tmp.scat_last_normals
            all_paths_tmp.scat_src_2_last_int_dist = (
                scat_paths_tmp.scat_src_2_last_int_dist
            )
            all_paths_tmp.scat_2_target_dist = scat_paths_tmp.scat_2_target_dist
            all_paths_tmp.scat_last_vertices = scat_paths_tmp.scat_last_vertices
            # If testing, the transition matrices are also returned
            if testing:
                scat_paths_tmp.mat_t = scat_mat_t

        #################################################
        # Splitting the sources (targets) dimension into
        # transmitters (receivers) and antennas, or
        # applying the synthetic arrays
        #################################################

        # If not using synthetic array, then the paths for the different
        # antenna elements were generated and reshaping is needed.
        # Otherwise, expand with the antenna dimensions.
        # [num_targets, num_sources, max_num_paths]
        all_paths.targets_sources_mask = all_paths.mask
        if self._scene.synthetic_array:
            # [num_rx, num_tx, 2, 2]
            mat_t = all_paths_tmp.mat_t
            # [num_rx, 1, num_tx, 1, max_num_paths, 2, 2]
            mat_t = tf.expand_dims(tf.expand_dims(mat_t, axis=1), axis=3)
            all_paths_tmp.mat_t = mat_t
        else:
            num_rx = len(self._scene.receivers)
            num_tx = len(self._scene.transmitters)
            max_num_paths = tf.shape(all_paths.vertices)[3]
            batch_dims = [num_rx, rx_array_size, num_tx, tx_array_size, max_num_paths]
            # [num_rx, tx_array_size, num_tx, tx_array_size, max_num_paths]
            all_paths.mask = tf.reshape(all_paths.mask, batch_dims)
            all_paths.tau = tf.reshape(all_paths.tau, batch_dims)
            all_paths.theta_t = tf.reshape(all_paths.theta_t, batch_dims)
            all_paths.phi_t = tf.reshape(all_paths.phi_t, batch_dims)
            all_paths.theta_r = tf.reshape(all_paths.theta_r, batch_dims)
            all_paths.phi_r = tf.reshape(all_paths.phi_r, batch_dims)
            # [num_rx, rx_array_size, num_tx, tx_array_size, max_num_paths, 2,2]
            all_paths_tmp.mat_t = tf.reshape(all_paths_tmp.mat_t, batch_dims + [2, 2])
            # [num_rx, rx_array_size, num_tx, tx_array_size, max_num_paths, 3]
            all_paths_tmp.k_tx = tf.reshape(all_paths_tmp.k_tx, batch_dims + [3])
            all_paths_tmp.k_rx = tf.reshape(all_paths_tmp.k_rx, batch_dims + [3])

        ####################################################
        # Compute the channel coefficients
        ####################################################
        scat_keep_prob = scat_paths_tmp.scat_keep_prob
        num_samples = scat_paths_tmp.num_samples
        all_paths.a = self._compute_paths_coefficients_interaction(
            rx_rot_mat,
            tx_rot_mat,
            all_paths,
            all_paths_tmp,
            num_samples,
            scat_keep_prob,
        )

        # If using synthetic array, adds the antenna dimentions by applying
        # synthetic phase shifts
        if self._scene.synthetic_array:
            all_paths.a = self._apply_synthetic_array(
                rx_rot_mat, tx_rot_mat, all_paths, all_paths_tmp
            )

        ##################################################
        # If not using synthetic arrays, tile the AoAs,
        # AoDs, and delays to handle dual-polarization
        ##################################################
        if not self._scene.synthetic_array:
            num_rx_patterns = len(self._scene.rx_array.antenna.patterns)
            num_tx_patterns = len(self._scene.tx_array.antenna.patterns)
            # [num_rx, 1,rx_array_size, num_tx, 1,tx_array_size, max_num_paths]
            mask = tf.expand_dims(tf.expand_dims(all_paths.mask, axis=2), axis=5)
            tau = tf.expand_dims(tf.expand_dims(all_paths.tau, axis=2), axis=5)
            theta_t = tf.expand_dims(tf.expand_dims(all_paths.theta_t, axis=2), axis=5)
            phi_t = tf.expand_dims(tf.expand_dims(all_paths.phi_t, axis=2), axis=5)
            theta_r = tf.expand_dims(tf.expand_dims(all_paths.theta_r, axis=2), axis=5)
            phi_r = tf.expand_dims(tf.expand_dims(all_paths.phi_r, axis=2), axis=5)
            # [num_rx, num_rx_patterns, rx_array_size, num_tx, num_tx_patterns,
            #   tx_array_size, max_num_paths]
            mask = tf.tile(mask, [1, num_rx_patterns, 1, 1, num_tx_patterns, 1, 1])
            tau = tf.tile(tau, [1, num_rx_patterns, 1, 1, num_tx_patterns, 1, 1])
            theta_t = tf.tile(
                theta_t, [1, num_rx_patterns, 1, 1, num_tx_patterns, 1, 1]
            )
            phi_t = tf.tile(phi_t, [1, num_rx_patterns, 1, 1, num_tx_patterns, 1, 1])
            theta_r = tf.tile(
                theta_r, [1, num_rx_patterns, 1, 1, num_tx_patterns, 1, 1]
            )
            phi_r = tf.tile(phi_r, [1, num_rx_patterns, 1, 1, num_tx_patterns, 1, 1])
            # [num_rx, num_rx_ant = num_rx_patterns*num_rx_ant,
            #   ... num_tx, num_tx_ant = num_tx_patterns*tx_array_size,
            #   ... max_num_paths]
            all_paths.mask = flatten_dims(flatten_dims(mask, 2, 1), 2, 3)
            all_paths.tau = flatten_dims(flatten_dims(tau, 2, 1), 2, 3)
            all_paths.theta_t = flatten_dims(flatten_dims(theta_t, 2, 1), 2, 3)
            all_paths.phi_t = flatten_dims(flatten_dims(phi_t, 2, 1), 2, 3)
            all_paths.theta_r = flatten_dims(flatten_dims(theta_r, 2, 1), 2, 3)
            all_paths.phi_r = flatten_dims(flatten_dims(phi_r, 2, 1), 2, 3)

        # If testing, additinal data is returned
        if testing:
            output = (
                sources,
                targets,
                all_paths.to_dict(),
                spec_paths_tmp.to_dict(),
                diff_paths_tmp.to_dict(),
                scat_paths_tmp.to_dict(),
            )
        else:
            output = (sources, targets, all_paths.to_dict())

        return output
