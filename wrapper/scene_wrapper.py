from sionna.rt import PlanarArray, load_scene, Paths
from sionna.rt.solver_paths import PathsTmpData
from sionna.rt.paths import Paths
import tensorflow as tf

from .solver_path_wrapper import SolverPathsWrapper


class SceneWrapper:
    def __init__(
        self,
        env_filename,
        tx_pattern="tr38901",
        rx_pattern="dipole",
        dtype=tf.complex64,
    ):
        """This is used together with the custom_scene"""
        self.scene = load_scene(env_filename)
        self.scene.frequency = 3.438e9
        self.scene.synthetic_array = False
        num_rows = 1
        num_cols = 1

        self.scene.tx_array = PlanarArray(
            num_rows=num_rows,
            num_cols=num_cols,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern=tx_pattern,
            polarization="V",
        )

        # This is the antenna used by the measurement robot
        self.scene.rx_array = PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern=rx_pattern,
            polarization="V",
        )

        self._solver_paths = SolverPathsWrapper(self, dtype=dtype)

    @property
    def objects(self):
        return self.scene.objects

    @property
    def receivers(self):
        return self.scene.receivers

    @property
    def frequency(self):
        return self.scene.frequency

    @property
    def wavelength(self):
        return self.scene.wavelength

    @property
    def tx_array(self):
        return self.scene.tx_array

    @property
    def rx_array(self):
        return self.scene.rx_array

    @property
    def transmitters(self):
        return self.scene.transmitters

    @property
    def receivers(self):
        return self.scene.receivers

    @property
    def dtype(self):
        return self.scene.dtype

    @property
    def synthetic_array(self):
        return self.scene.synthetic_array

    @property
    def mi_scene(self):
        return self.scene.mi_scene

    @property
    def objects(self):
        return self.scene.objects

    def add(self, item):
        return self.scene.add(item)

    def get(self, name):
        return self.scene.get(name)

    def remove(self, name):
        return self.scene.remove(name)

    @property
    def neural_scene(self):
        return self._neural_scene

    @neural_scene.setter
    def neural_scene(self, neural_scene_model):
        self._neural_scene = neural_scene_model

    def compute_fields(
        self,
        spec_paths,
        diff_paths,
        scat_paths,
        spec_paths_tmp,
        diff_paths_tmp,
        scat_paths_tmp,
        check_scene=True,
        testing=False,
    ):

        # Check that all is set to compute paths
        if check_scene:
            self._check_scene(False)

        # Compute the fields and merge the paths
        output = self._solver_paths.compute_fields_interaction_modeling(
            spec_paths,
            diff_paths,
            scat_paths,
            spec_paths_tmp,
            diff_paths_tmp,
            scat_paths_tmp,
            testing,
        )
        sources, targets, paths_as_dict = output[:3]
        paths = Paths(sources, targets, self)
        paths.from_dict(paths_as_dict)

        # If the hidden input flag testing is True, additional data
        # is returned which is required for some unit tests
        if testing:
            spec_tmp_as_dict, diff_tmp_as_dict, scat_tmp_as_dict = output[3:]
            spec_tmp = PathsTmpData(sources, targets, self._dtype)
            spec_tmp.from_dict(spec_tmp_as_dict)
            diff_tmp = PathsTmpData(sources, targets, self._dtype)
            diff_tmp.from_dict(diff_tmp_as_dict)
            scat_tmp = PathsTmpData(sources, targets, self._dtype)
            scat_tmp.from_dict(scat_tmp_as_dict)
            paths.spec_tmp = spec_tmp
            paths.diff_tmp = diff_tmp
            paths.scat_tmp = scat_tmp

        # Finalize paths computation
        paths.finalize()

        return paths
