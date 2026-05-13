import os
import shutil
from typing import Optional, Sequence
from ase.calculators.lammps import convert, Prism
from kim_tools import KIMTestDriverError
from kim_tools.symmetry_util.core import reduce_and_avg, PeriodExtensionException
from kim_tools.test_driver import SingleCrystalTestDriver
from .structure_utils import compute_supercell_for_target_size
from .helper_functions import get_cell_from_averaged_lammps_dump, get_positions_from_averaged_lammps_dump, run_lammps


class TestDriver(SingleCrystalTestDriver):
    def _calculate(
        self,
        timestep_ps: float = 0.001,
        target_size: int = 10000,
        repeat: Optional[Sequence[int]] = None,
        lammps_command: str = "lmp",
        msd_threshold_angstrom_squared_per_sampling_timesteps: float = 0.1,
        msd_timesteps: int = 20000,
        thermo_sampling_period: int = 100,
        random_seed: int = 1,
        rlc_n_every: int = 10,
        rlc_initial_run_length: int = 10000,
        rlc_min_samples: int = 100,
        output_dir: str = "output",
        equilibration_plots: bool = True,
        **kwargs) -> None:
        """
        Compute crystal structure at constant pressure and temperature (NPT) with a Lammps molecular-dynamics simulation.

        This test driver repeats the unit cell to build a supercell and then runs a molecular-dynamics simulation in the
        NPT ensemble using Lammps.

        This test driver uses kim_convergence to detect an equilibrated molecular-dynamics simulation. It checks
        convergence of the volume, temperature, enthalpy and cell shape parameters every rlc_run_length timesteps.

        After the molecular-dynamics simulation, the symmetry of the structure is checked to ensure that it did not
        change.

        The crystal might melt or vaporize during the simulation. In that case, kim_convergence would only detect
        equilibration after an unnecessarily long simulation. Therefore, we initially check for melting or vaporization
        during a short initial simulation. During this run, we monitor the mean-squared displacement (MSD) of atoms
        during the simulation. If the MSD exceeds a given threshold value
        (msd_threshold_angstrom_squared_per_sampling_timesteps), an error is raised.

        All output files are written to the given output directory.

        :param timestep_ps:
            Time step in picoseconds.
            Default is 0.001 ps (1 fs).
            Should be bigger than zero.
        :type timestep_ps: float
        :param target_size:
            Target number of atoms in the supercell to build by repeating the unit cell. Uses cutoff-based expansion
            with target size constraint (good for non-cubic cells). The algorithm starts with an 20Å cutoff radius and
            recursively decreases it until the supercell has fewer atoms than target_size.
            Default is 10000.
            Should be bigger than zero.
            Ignored if repeat is specified.
        :type target_size: int
        :param repeat:
            Tuple of three integers specifying how often to repeat the unit cell in each direction to build the
            supercell.
            If None, the repeat will be determined based on the target_size argument using a cutoff-based approach.
            Default is None.
            If not None, all entries have to be bigger than zero.
        :type repeat: Sequence[int]
        :param lammps_command:
            Command to run Lammps.
            Default is "lmp".
        :type lammps_command: str
        :param msd_threshold_angstrom_squared_per_sampling_timesteps:
            Mean-squared displacement threshold in Angstroms^2 per thermo_sampling_period to detect melting or
            vaporization.
            Default is 0.1.
            Should be bigger than zero.
        :type msd_threshold_angstrom_squared_per_sampling_timesteps: float
        :param msd_timesteps:
            Number of timesteps to monitor the mean-squared displacement in Lammps.
            Before the mean-squared displacement is monitored, the system will be equilibrated for the same number of
            timesteps.
            Default is 20000 timesteps.
            Should be bigger than zero and a multiple of thermo_sampling_period.
        :type msd_timesteps: int
        :param thermo_sampling_period:
            Sample thermodynamic variables every thermo_sampling_period timesteps in Lammps.
            Default is 100 timesteps.
            Should be bigger than zero.
        :type thermo_sampling_period: int
        :param random_seed:
            Random seed for Lammps simulation.
            Default is 1.
            Should be bigger than zero.
        :type random_seed: int
        :param rlc_n_every:
            Number of timesteps between storage of values for the run-length control in kim-convergence.
            Default is 10.
            Should be bigger than zero.
        :type rlc_n_every: int
        :param rlc_initial_run_length:
            Run length in timesteps for run-length control with kim-convergence.
            This will also be the timestep interval in generated trajectory files.
            Default is 10000 timesteps.
            Should be bigger than zero and a multiple of thermo_sampling_period.
        :type rlc_initial_run_length: int
        :param rlc_min_samples:
            Minimum number of independent samples for convergence in run-length control with kim-convergence.
            Default is 100.
            Should be bigger than zero.
        :type rlc_min_samples: int
        :param output_dir:
            Directory to which all output files will be written.
            Default is "output".
        :type output_dir: str
        :param equilibration_plots:
            Whether to generate diagnostic plots for the equilibration checks in kim-convergence.
            Default is True.
        :type equilibration_plots: bool

        :raises ValueError:
            If any of the input arguments are invalid.
        :raises KIMTestDriverError:
            If the crystal melts or vaporizes during the simulation.
            If the symmetry of the structure changes.
            If the output directory does not exist.
        """
        # Set prototype label.
        self.prototype_label = self._get_nominal_crystal_structure_npt()["prototype-label"]["source-value"]

        # Get temperature in Kelvin.
        temperature_K = self._get_temperature(unit="K")

        # Get cauchy stress tensor in bar.
        cell_cauchy_stress_bar = self._get_cell_cauchy_stress(unit="bar")

        # Check arguments.
        if not temperature_K > 0.0:
            raise ValueError("Temperature has to be larger than zero.")

        if not len(cell_cauchy_stress_bar) == 6:
            raise ValueError("Specify all six (x, y, z, xy, xz, yz) entries of the cauchy stress tensor.")

        if not (cell_cauchy_stress_bar[0] == cell_cauchy_stress_bar[1] == cell_cauchy_stress_bar[2]):
            raise ValueError("The diagonal entries of the stress tensor have to be equal so that a hydrostatic "
                             "pressure is used.")

        if not (cell_cauchy_stress_bar[3] == cell_cauchy_stress_bar[4] == cell_cauchy_stress_bar[5] == 0.0):
            raise ValueError("The off-diagonal entries of the stress tensor have to be zero so that a hydrostatic "
                             "pressure is used.")

        if not timestep_ps > 0.0:
            raise ValueError("Timestep has to be larger than zero.")

        if not thermo_sampling_period > 0:
            raise ValueError("Number of timesteps between sampling in Lammps has to be bigger than zero.")

        if not target_size > 0:
            raise ValueError("Target size for supercell construction has to be bigger than zero.")

        if repeat is not None:
            if not len(repeat) == 3:
                raise ValueError("The repeat argument has to be a tuple of three integers.")

            if not all(r > 0 for r in repeat):
                raise ValueError("All number of repeats must be bigger than zero.")
        
        if not msd_threshold_angstrom_squared_per_sampling_timesteps > 0.0:
            raise ValueError("The mean-squared displacement threshold has to be bigger than zero.")

        if not msd_timesteps > 0:
            raise ValueError("The number of timesteps to monitor the mean-squared displacement has to be bigger than "
                             "zero.")

        if not msd_timesteps % thermo_sampling_period == 0:
            raise ValueError("The number of timesteps to monitor the mean-squared displacement has to be a multiple of "
                             "the thermo sampling period.")

        if not random_seed > 0:
            raise ValueError("The random seed has to be bigger than zero.")

        if not rlc_n_every > 0:
            raise ValueError("The number of timesteps between storage of values for run-length control has to be "
                             "bigger than zero.")

        if not rlc_initial_run_length > 0:
            raise ValueError("The run length for run-length control has to be bigger than zero.")

        if not rlc_initial_run_length % thermo_sampling_period == 0:
            raise ValueError("The run length for run-length control has to be a multiple of the number of the thermo"
                             "sampling period.")

        if not rlc_min_samples > 0:
            raise ValueError("The minimum number of samples to use for convergence checks in run-length control has to "
                             "be bigger than zero.")

        # Get pressure from cauchy stress tensor.
        pressure_bar = -cell_cauchy_stress_bar[0]

        # Create atoms object that will contain the supercell.
        atoms_new = self._get_atoms()

        # This is how ASE obtains the species that are written to the initial configuration.
        # These species are passed to kim interactions.
        # See https://wiki.fysik.dtu.dk/ase/_modules/ase/io/lammpsdata.html#write_lammps_data
        symbols = atoms_new.get_chemical_symbols()
        species = sorted(set(symbols))

        if repeat is not None:
            # Use explicit repeat tuple.
            atoms_new = atoms_new.repeat(repeat)
        else:
            # Use cutoff-based expansion with target size constraint
            # (good for non-cubic cells, ensures natoms >= target_size)
            atoms_new, repeat = compute_supercell_for_target_size(atoms_new.copy(), target_size)

        # Make sure output directory for all data files exists and copy over necessary files.
        if not os.path.exists(output_dir):
            raise KIMTestDriverError(f"Output directory '{output_dir}' does not exist.")
        test_driver_directory = os.path.dirname(os.path.realpath(__file__))
        shutil.copyfile(os.path.join(test_driver_directory, "npt.lammps"), f"{output_dir}/npt.lammps")
        shutil.copyfile(os.path.join(test_driver_directory, "run_length_control.py"),
                        f"{output_dir}/run_length_control.py")
        # Choose the correct accuracies file for kim-convergence based on whether the cell is orthogonal or not.
        with open(f"{output_dir}/accuracies.py", "w") as file:
            print("""from typing import Optional, Sequence

# A relative half-width requirement or the accuracy parameter. Target value
# for the ratio of halfwidth to sample mean. If n_variables > 1,
# relative_accuracy can be a scalar to be used for all variables or a 1darray
# of values of size n_variables.
# For cells, we can only use a relative accuracy for all non-zero variables.
# The last three variables, however, correspond to the tilt factors of the orthogonal cell (see npt.lammps which are
# expected to fluctuate around zero. For these, we should use an absolute accuracy instead.""", file=file)
            relative_accuracies = ["0.01", "0.01", "0.01", "0.01", "0.01", "0.01", "0.01", "0.01", "0.01"]
            absolute_accuracies = ["None", "None", "None", "None", "None", "None", "None", "None", "None"]
            _, _, _, xy, xz, yz = convert(Prism(atoms_new.get_cell()).get_lammps_prism(), "distance",
                                          "ASE", "metal")
            if abs(xy) < 1.0e-6:
                relative_accuracies[6] = "None"
                absolute_accuracies[6] = "0.01"
            if abs(xz) < 1.0e-6:
                relative_accuracies[7] = "None"
                absolute_accuracies[7] = "0.01"
            if abs(yz) < 1.0e-6:
                relative_accuracies[8] = "None"
                absolute_accuracies[8] = "0.01"
            print(f"RELATIVE_ACCURACY: Sequence[Optional[float]] = [{', '.join(relative_accuracies)}]", file=file)
            print(f"ABSOLUTE_ACCURACY: Sequence[Optional[float]] = [{', '.join(absolute_accuracies)}]", file=file)

        with open(f"{output_dir}/rlc_parameters.py", "w") as file:
            print(f"""from typing import Optional

INITIAL_RUN_LENGTH: int = {rlc_initial_run_length}
MINIMUM_NUMBER_OF_INDEPENDENT_SAMPLES: Optional[int] = {rlc_min_samples}""", file=file)

        # Write lammps file.
        structure_file = f"{output_dir}/zero_temperature_crystal.lmp"
        atom_style = self._get_supported_lammps_atom_style()
        atoms_new.write(structure_file, format="lammps-data", masses=True, units="metal", atom_style=atom_style)

        # Run single Lammps simulation.
        log_filename, restart_filename, average_position_filename, average_cell_filename, melted_crystal_filename = run_lammps(
            modelname=self.kim_model_name,
            temperature_K=temperature_K,
            pressure_bar=pressure_bar,
            timestep_ps=timestep_ps,
            thermo_sampling_period=thermo_sampling_period,
            species=species,
            msd_threshold_angstrom_squared_per_sampling_timesteps=msd_threshold_angstrom_squared_per_sampling_timesteps,
            number_msd_timesteps=msd_timesteps,
            rlc_run_length=rlc_initial_run_length,
            rlc_n_every=rlc_n_every,
            output_dir=output_dir,
            equilibration_plots=equilibration_plots,
            lammps_command=lammps_command,
            random_seed=random_seed,
            )

        # Check that crystal did not melt or vaporize.
        with open(log_filename, "r") as f:
            for line in f:
                if line.startswith("Crystal melted or vaporized"):
                    assert os.path.exists(melted_crystal_filename)
                    raise KIMTestDriverError(f"Crystal melted or vaporized during simulation at temperature {temperature_K} K.")                    
        assert not os.path.exists(melted_crystal_filename)

        # Process results and check that symmetry is unchanged after simulation.
        atoms_new.set_cell(get_cell_from_averaged_lammps_dump(average_cell_filename))
        atoms_new.set_scaled_positions(
            get_positions_from_averaged_lammps_dump(average_position_filename))
        try:
            reduced_atoms = reduce_and_avg(atoms_new, repeat)
        except PeriodExtensionException as e:
            atoms_new.write(f"output/final_configuration_failing.poscar",
                            format="vasp", sort=True)
            raise KIMTestDriverError(f"Could not reduce structure after NPT simulation at "
                                     f"temperature {temperature_K} K: {e}")

        # Check that the symmetry of the structure did not change.
        if not self._verify_unchanged_symmetry(reduced_atoms):
            reduced_atoms.write(f"output/reduced_atoms_failing.poscar",
                                format="vasp", sort=True)
            raise KIMTestDriverError(f"Symmetry of structure changed during simulation at temperature {temperature_K} K.")

        # Write NPT crystal structure.
        self._update_nominal_parameter_values(reduced_atoms)
        self._add_property_instance_and_common_crystal_genome_keys("crystal-structure-npt", write_stress=True,
                                                                   write_temp=temperature_K)
        self._add_file_to_current_property_instance("restart-file", restart_filename)

        print('####################################')
        print('# NPT Crystal Structure Results #')
        print('####################################')
        print(f'Temperature: {temperature_K} K')
        print(f'Pressure: {pressure_bar} bar')
