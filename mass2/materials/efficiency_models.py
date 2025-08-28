from dataclasses import dataclass, field
from uncertainties import ufloat, Variable
from uncertainties import unumpy as unp
from .uncertainties_helpers import ensure_uncertain, with_fractional_uncertainty
from numpy.typing import NDArray, ArrayLike
import numpy as np
import pylab as plt

try:
    import xraydb

except ModuleNotFoundError:
    print('** Skipping module mass2.materials, because it requires the "xraydb" python package.')
    print("** Please see https://xraypy.github.io/XrayDB/installation.html for installation instructions.")

# TODO: This module might be clearer if we remove the distinction between FilterStack and Film, since each
# does (after all) represent a sequence of materials. Also the specific materials like `AlFilmWithOxide`
# become instances, created by classmethods with that name.
# Make it a frozen dataclass that can return a copy with an extra material.


@dataclass()
class FilterStack:
    """Represent a sequence of named materials"""

    name: str
    components: list["Filter | FilterStack"] = field(default_factory=list)

    def add(self, film: "Filter | FilterStack") -> None:
        self.components.append(film)

    def add_filter(
        self,
        name: str,
        material: str,
        area_density_g_per_cm2: float | None = None,
        thickness_nm: float | None = None,
        density_g_per_cm3: float | None = None,
        fill_fraction: Variable = ufloat(1, 1e-8),
        absorber: bool = False,
    ) -> None:
        self.add(
            Filter.newfilter(
                name,
                material,
                area_density_g_per_cm2=area_density_g_per_cm2,
                thickness_nm=thickness_nm,
                density_g_per_cm3=density_g_per_cm3,
                fill_fraction=fill_fraction,
                absorber=absorber,
            )
        )

    def get_efficiency(self, xray_energies_eV: ArrayLike, uncertain: bool = False) -> NDArray:
        assert len(self.components) > 0, f"{self.name} has no components of which to calculate efficiency"
        individual_efficiency = np.array([
            iComponent.get_efficiency(xray_energies_eV, uncertain=uncertain) for iComponent in self.components
        ])
        efficiency = np.prod(individual_efficiency, axis=0)
        if uncertain:
            return efficiency
        else:
            return unp.nominal_values(efficiency)

    def __call__(self, xray_energies_eV: ArrayLike, uncertain: bool = False) -> NDArray:
        return self.get_efficiency(xray_energies_eV, uncertain=uncertain)

    def plot_efficiency(self, xray_energies_eV: ArrayLike, ax: plt.Axes | None = None) -> None:
        efficiency = unp.nominal_values(self.get_efficiency(xray_energies_eV))
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        ax.plot(xray_energies_eV, efficiency * 100.0, label="total", lw=2)
        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel("Efficiency (%)")
        ax.set_title(self.name)
        ax.set_title(f"{self.name} Efficiency")

        for v in self.components:
            efficiency = v.get_efficiency(xray_energies_eV)
            ax.plot(xray_energies_eV, efficiency * 100.0, "--", label=v.name)

        ax.legend()

    def __repr__(self) -> str:
        s = f"{type(self)}(\n"
        for v in self.components:
            s += f"{v.name}: {v}\n"
        s += ")"
        return s


@dataclass(frozen=True)
class Filter:
    name: str
    material: NDArray
    atomic_number: NDArray
    density_g_per_cm3: NDArray[np.float64]
    thickness_cm: NDArray[np.float64]
    fill_fraction: Variable = ufloat(1.0, 1e-8)
    absorber: bool = False

    def get_efficiency(self, xray_energies_eV: ArrayLike, uncertain: bool = False) -> NDArray:
        optical_depth = np.vstack([
            xraydb.material_mu(m, xray_energies_eV, density=d) * t
            for (m, d, t) in zip(self.material, self.density_g_per_cm3, self.thickness_cm)
        ])
        individual_transmittance = unp.exp(-optical_depth)
        transmittance = np.prod(individual_transmittance, axis=0)
        if self.absorber:
            efficiency = (1.0 - transmittance) * self.fill_fraction
        else:
            efficiency = (transmittance * self.fill_fraction) + (1.0 - self.fill_fraction)
        if uncertain:
            return efficiency
        else:
            return unp.nominal_values(efficiency)

    def __repr__(self) -> str:
        s = f"{type(self)}("
        for material, density, thick in zip(self.material, self.density_g_per_cm3, self.thickness_cm):
            area_density = density * thick
            s += f"{material} {area_density:.3g} g/cm^2, "
        s += f"fill_fraction={self.fill_fraction:.3f}, absorber={self.absorber})"
        return s

    @classmethod
    def newfilter(
        cls,
        name: str,
        material: ArrayLike,
        area_density_g_per_cm2: ArrayLike | None = None,
        thickness_nm: ArrayLike | None = None,
        density_g_per_cm3: ArrayLike | None = None,
        fill_fraction: Variable = ufloat(1, 1e-8),
        absorber: bool = False,
    ) -> "Filter":
        material = np.array(material, ndmin=1)
        atomic_number = np.array([xraydb.atomic_number(iMaterial) for iMaterial in material], ndmin=1)
        fill_fraction = ensure_uncertain(fill_fraction)

        # Save density, either default values for that element, or the given density.
        if density_g_per_cm3 is None:
            density_g_per_cm3 = np.array([xraydb.atomic_density(int(iAtomicNumber)) for iAtomicNumber in atomic_number], ndmin=1)
        else:
            density_g_per_cm3 = np.array(density_g_per_cm3, ndmin=1)
            assert len(material) == len(density_g_per_cm3)

        # Handle input value of areal density or thickness, but not both.
        assert np.logical_xor(area_density_g_per_cm2 is None, thickness_nm is None), (
            "must specify either areal density or thickness, not both"
        )
        if thickness_nm is not None:
            thickness_cm = np.array(thickness_nm, ndmin=1) * 1e-7
        elif area_density_g_per_cm2 is not None:
            area_density_g_per_cm2 = np.array(area_density_g_per_cm2, ndmin=1)
            thickness_cm = area_density_g_per_cm2 / density_g_per_cm3
            if np.ndim == 0:
                thickness_cm = np.array(thickness_cm, ndmin=1)
        else:
            raise ValueError("must specify either areal density or thickness, not both")
        thickness_cm = ensure_uncertain(thickness_cm)
        assert len(thickness_cm) >= 1

        return cls(name, material, atomic_number, density_g_per_cm3, thickness_cm, fill_fraction, absorber)


def AlFilmWithOxide(
    name: str,
    Al_thickness_nm: float,
    Al_density_g_per_cm3: float | None = None,
    num_oxidized_surfaces: int = 2,
    oxide_density_g_per_cm3: ArrayLike | None = None,
) -> Filter:
    """Create a Filter made of an alumninum film with oxides on one or both surfaces

    Args:
        name: name given to filter object, e.g. '50K Filter'.
        Al_thickness_nm: thickness, in nm, of Al film
        Al_density_g_per_cm3: Al film density, in g/cm3, defaults to xraydb value
        num_oxidized_surfaces: Number of film surfaces that contain a native oxide, default 2
        oxide_density_g_per_cm3: Al2O3 oxide density, in g/cm3, defaults to bulk xraydb value
    """
    assert num_oxidized_surfaces in {1, 2}, "only 1 or 2 oxidzed surfaces allowed"
    if Al_density_g_per_cm3 is None:
        Al_density_g_per_cm3 = float(xraydb.atomic_density("Al"))
    arbE = 5000.0  # an arbitrary energy (5 keV) is used to get answers from material_mu_components()
    oxide_dict = xraydb.material_mu_components("sapphire", arbE)
    oxide_material = oxide_dict["elements"]
    oxide_mass_fractions = [oxide_dict[x][0] * oxide_dict[x][1] / oxide_dict["mass"] for x in oxide_material]

    # Assume oxidized surfaces are each 3 nm thick.
    num_oxide_elements = len(oxide_material)
    oxide_thickness_nm = np.repeat(num_oxidized_surfaces * 3.0, num_oxide_elements)
    if oxide_density_g_per_cm3 is None:
        oxide_density_g_per_cm3 = np.repeat(oxide_dict["density"], num_oxide_elements)
    else:
        oxide_density_g_per_cm3 = np.asarray(oxide_density_g_per_cm3)

    material = np.hstack(["Al", oxide_material])
    density_g_per_cm3 = np.hstack([Al_density_g_per_cm3, oxide_density_g_per_cm3 * oxide_mass_fractions])
    thickness_nm = np.hstack([Al_thickness_nm, oxide_thickness_nm])
    return Filter.newfilter(name, material, thickness_nm=thickness_nm, density_g_per_cm3=density_g_per_cm3)


def AlFilmWithPolymer(
    name: str,
    Al_thickness_nm: float,
    polymer_thickness_nm: float,
    Al_density_g_per_cm3: float | None = None,
    num_oxidized_surfaces: int = 1,
    oxide_density_g_per_cm3: float | None = None,
    polymer_density_g_per_cm3: float | None = None,
) -> Filter:
    """Create a Filter made of an alumninum film with polymer backing

    Args:
        name: name given to filter object, e.g. '50K Filter'.
        Al_thickness_nm: thickness, in nm, of Al film
        polymer_thickness_nm: thickness, in nm, of filter backside polymer
        Al_density_g_per_cm3: Al film density, in g/cm3, defaults to xraydb value
        num_oxidized_surfaces: Number of film surfaces that contain a native oxide, default 2
        oxide_density_g_per_cm3: Al2O3 oxide density, in g/cm3, defaults to bulk xraydb value
        polymer_density_g_per_cm3: Polymer density, in g/cm3, defaults to Kapton
    """
    assert num_oxidized_surfaces in {1, 2}, "only 1 or 2 oxidzed surfaces allowed"
    if Al_density_g_per_cm3 is None:
        Al_density_g_per_cm3 = xraydb.atomic_density("Al")

    arbE = 5000.0  # an arbitrary energy (5 keV) is used to get answers from material_mu_components()
    oxide_dict = xraydb.material_mu_components("sapphire", arbE)
    oxide_thickness_nm = num_oxidized_surfaces * 3.0  # assume 3 nm per oxidized surface
    oxide_material = oxide_dict["elements"]
    oxide_mass_fractions = np.array([oxide_dict[x][0] * oxide_dict[x][1] / oxide_dict["mass"] for x in oxide_material])
    if oxide_density_g_per_cm3 is None:
        oxide_density_g_per_cm3 = oxide_dict["density"] * np.ones(len(oxide_material))

    polymer_dict = xraydb.material_mu_components("kapton", arbE)
    polymer_material = polymer_dict["elements"]
    polymer_thickness_nm_array = np.ones(len(polymer_material)) * polymer_thickness_nm
    polymer_mass_fractions = np.array([polymer_dict[x][0] * polymer_dict[x][1] / polymer_dict["mass"] for x in polymer_material])
    if polymer_density_g_per_cm3 is None:
        polymer_density_g_per_cm3 = polymer_dict["density"] * np.ones(len(polymer_material))

    material = np.hstack(["Al", oxide_material, polymer_material])
    density_g_per_cm3 = np.hstack([
        [Al_density_g_per_cm3],
        oxide_density_g_per_cm3 * oxide_mass_fractions,
        polymer_density_g_per_cm3 * polymer_mass_fractions,
    ])
    thickness_nm = np.hstack([Al_thickness_nm, oxide_thickness_nm, polymer_thickness_nm_array])

    return Filter.newfilter(name=name, material=material, thickness_nm=thickness_nm, density_g_per_cm3=density_g_per_cm3)


def LEX_HT(name: str) -> FilterStack:
    """Create an Al film with polymer and stainless steel backing.

    Models the LEX-HT vacuum window.

    Args:
        name: name given to filter object, e.g. '50K Filter'.
    """
    # Set up Al + polyimide film
    film_material = ["C", "H", "N", "O", "Al"]
    film_area_density_g_per_cm2_given = np.array([6.7e-5, 2.6e-6, 7.2e-6, 1.7e-5, 1.7e-5])
    film_area_density_g_per_cm2 = with_fractional_uncertainty(film_area_density_g_per_cm2_given, 0.03)
    film1 = Filter.newfilter(name="LEX_HT Film", material=film_material, area_density_g_per_cm2=film_area_density_g_per_cm2)
    # Set up mesh
    mesh_material = ["Fe", "Cr", "Ni", "Mn", "Si"]
    mesh_thickness = 100.0e-4  # cm
    mesh_density = 8.0  # g/cm^3
    mesh_material_fractions = np.array([0.705, 0.19, 0.09, 0.01, 0.005])  # fraction by weight
    mesh_area_density_g_per_cm2_scalar = mesh_material_fractions * mesh_density * mesh_thickness  # g/cm^2
    mesh_area_density_g_per_cm2 = with_fractional_uncertainty(mesh_area_density_g_per_cm2_scalar, 0.02)
    mesh_fill_fraction = ufloat(0.19, 0.01)
    film2 = Filter.newfilter(
        name="LEX_HT Mesh",
        material=mesh_material,
        area_density_g_per_cm2=mesh_area_density_g_per_cm2,
        fill_fraction=mesh_fill_fraction,
    )
    stack = FilterStack(name)
    stack.add(film1)
    stack.add(film2)
    return stack


def get_filter_stacks_dict() -> dict[str, FilterStack]:
    """Create a dictionary with a few examples of FilterStack objects

    Returns
    -------
    dict
        A dictionary of named FilterStacks
    """
    fs_dict: dict[str, FilterStack] = {}

    # EBIT Instrument
    EBIT_filter_stack = FilterStack(name="EBIT 2018")
    EBIT_filter_stack.add_filter(
        name="Electroplated Au Absorber", material="Au", thickness_nm=with_fractional_uncertainty(965.5, 0.03), absorber=True
    )

    EBIT_filter_stack.add(AlFilmWithOxide(name="50mK Filter", Al_thickness_nm=with_fractional_uncertainty(112.5, 0.02)))
    EBIT_filter_stack.add(AlFilmWithOxide(name="3K Filter", Al_thickness_nm=with_fractional_uncertainty(108.5, 0.02)))

    filter_50K = FilterStack(name="50K Filter")
    filter_50K.add(AlFilmWithOxide(name="Al Film", Al_thickness_nm=with_fractional_uncertainty(102.6, 0.02)))
    nickel = Filter.newfilter(name="Ni Mesh", material="Ni", thickness_nm=ufloat(15.0e3, 2e3), fill_fraction=ufloat(0.17, 0.01))
    filter_50K.add(nickel)
    EBIT_filter_stack.add(filter_50K)
    luxel1 = LEX_HT("Luxel Window TES")
    luxel2 = LEX_HT("Luxel Window EBIT")
    EBIT_filter_stack.add(luxel1)
    EBIT_filter_stack.add(luxel2)
    fs_dict[EBIT_filter_stack.name] = EBIT_filter_stack

    # RAVEN Instrument
    RAVEN1_fs = FilterStack(name="RAVEN1 2019")
    RAVEN1_fs.add_filter(name="Evaporated Bi Absorber", material="Bi", thickness_nm=4.4e3, absorber=True)
    RAVEN1_fs.add(AlFilmWithPolymer(name="50mK Filter", Al_thickness_nm=108.4, polymer_thickness_nm=206.4))
    RAVEN1_fs.add(AlFilmWithPolymer(name="3K Filter", Al_thickness_nm=108.4, polymer_thickness_nm=206.4))
    RAVEN1_fs.add(AlFilmWithOxide(name="50K Filter", Al_thickness_nm=1.0e3))
    RAVEN1_fs.add_filter(name="Be TES Vacuum Window", material="Be", thickness_nm=200.0e3)
    RAVEN1_fs.add(AlFilmWithOxide(name="e- Filter", Al_thickness_nm=5.0e3))
    RAVEN1_fs.add_filter(name="Be SEM Vacuum Window", material="Be", thickness_nm=200.0e3)
    fs_dict[RAVEN1_fs.name] = RAVEN1_fs

    # Horton spring 2018, for metrology campaign.
    Horton_filter_stack = FilterStack(name="Horton 2018")
    Horton_filter_stack.add_filter(name="Electroplated Au Absorber", material="Au", thickness_nm=965.5, absorber=True)
    Horton_filter_stack.add(AlFilmWithOxide(name="50mK Filter", Al_thickness_nm=5000))
    Horton_filter_stack.add(AlFilmWithOxide(name="3K Filter", Al_thickness_nm=5000))
    Horton_filter_stack.add(AlFilmWithOxide(name="50K Filter", Al_thickness_nm=12700))
    Horton_filter_stack.add(LEX_HT("Luxel Window TES"))
    fs_dict[Horton_filter_stack.name] = Horton_filter_stack

    return fs_dict


filterstack_models = get_filter_stacks_dict()
