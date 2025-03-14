"""Basilisk world models are given in ``bsk_rl.sim.world``.

In most cases, the user does not need to specify the world model, as it is inferred from
the requirements of the :class:`~bsk_rl.sim.fsw.FSWModel`. However, the user can specify
the world model in the :class:`~bsk_rl.GeneralSatelliteTasking` constructor if desired.

Customization of the world model parameters is via the ``world_args`` parameter in the
:class:`~bsk_rl.GeneralSatelliteTasking`. As with ``sat_args``, these parameters are
passed as a dictionary of key-value or key-function pairs, with the latter called to
generate the value each time the simulation is reset.

.. code-block:: python

    world_args = dict(
        utc_init="2018 SEP 29 21:00:00.000 (UTC)",  # set the epoch
        scaleHeight=np.random.uniform(7e3, 9e3),  # randomize the atmosphere
    )

In general, ``world_args`` parameter names match those used in Basilisk. See the setup
functions for short descriptions of what parameters do and the Basilisk documentation
for more detail on their exact model effects.

"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Union
from weakref import proxy

import numpy as np
from Basilisk import __path__
from Basilisk.simulation import (
    eclipse,
    ephemerisConverter,
    exponentialAtmosphere,
    groundLocation,
)
from Basilisk.utilities import macros as mc
from Basilisk.utilities import orbitalMotion, simIncludeGravBody

from bsk_rl.utils.functional import collect_default_args, default_args
from bsk_rl.utils.orbital import random_epoch

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.sim import Simulator

logger = logging.getLogger(__name__)

bsk_path = __path__[0]


class WorldModel(ABC):
    """Abstract Basilisk world model."""

    @classmethod
    def default_world_args(cls, **kwargs) -> dict[str, Any]:
        """Compile default arguments for the world model.

        Args:
            **kwargs: Arguments to override in the default arguments.

        Returns:
            Dictionary of arguments for simulation models.
        """
        defaults = collect_default_args(cls)
        for k, v in kwargs.items():
            if k not in defaults:
                raise KeyError(f"{k} not a valid key for world_args")
            defaults[k] = v
        return defaults

    def __init__(
        self,
        simulator: "Simulator",
        world_rate: float,
        priority: int = 300,
        **kwargs,
    ) -> None:
        """Abstract Basilisk world model.

        One WorldModel is instantiated for the environment each time a new simulator
        is created.

        Args:
            simulator: Simulator using this model.
            world_rate: Rate of world simulation [s]
            priority: Model priority.
            kwargs: Passed through to setup functions.
        """
        self.simulator: Simulator = proxy(simulator)

        world_proc_name = "WorldProcess"
        world_proc = self.simulator.CreateNewProcess(world_proc_name, priority)

        # Define process name, task name and task time-step
        self.world_task_name = "WorldTask"
        world_proc.addTask(
            self.simulator.CreateNewTask(
                self.world_task_name, mc.sec2nano(world_rate))
        )

        self._setup_world_objects(**kwargs)

    def __del__(self):
        """Log when world is deleted."""
        logger.debug("Basilisk world deleted")

    @abstractmethod  # pragma: no cover
    def _setup_world_objects(self, **kwargs) -> None:
        """Caller for all world objects."""
        pass


class BasicWorldModel(WorldModel):
    """Basic world with minimum necessary Basilisk world components."""

    def __init__(self, *args, **kwargs) -> None:
        """Basic world with minimum necessary Basilisk world components.

        This model includes ephemeris and SPICE-based Earth gravity and dynamics models,
        an exponential atmosphere model, and an eclipse model.

        Args:
            *args: Passed to superclass.
            **kwargs: Passed to superclass.
        """
        super().__init__(*args, **kwargs)

    @property
    def PN(self):
        """Planet relative to inertial frame rotation matrix."""
        return np.array(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.body_index]
            .read()
            .J20002Pfix
        ).reshape((3, 3))

    @property
    def omega_PN_N(self):
        """Planet angular velocity in inertial frame [rad/s]."""
        PNdot = np.array(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.body_index]
            .read()
            .J20002Pfix_dot
        ).reshape((3, 3))
        skew_PN_N = -np.matmul(np.transpose(self.PN), PNdot)
        return np.array([skew_PN_N[2, 1], skew_PN_N[0, 2], skew_PN_N[1, 0]])

    def _setup_world_objects(self, **kwargs) -> None:
        self.setup_gravity_bodies(**kwargs)
        self.setup_ephem_object(**kwargs)
        self.setup_atmosphere_density_model(**kwargs)
        self.setup_eclipse_object(**kwargs)

    @default_args(utc_init=random_epoch)
    def setup_gravity_bodies(
        self, utc_init: str, priority: int = 1100, **kwargs
    ) -> None:
        """Specify gravitational models to use in the simulation.

        Args:
            utc_init: UTC datetime string, in the format ``YYYY MMM DD hh:mm:ss.sss (UTC)``
            priority: Model priority.
            **kwargs: Passed to other setup functions.
        """
        self.gravFactory = simIncludeGravBody.gravBodyFactory()
        self.gravFactory.createSun()
        self.planet = self.gravFactory.createEarth()
        self.sun_index = 0
        self.body_index = 1

        self.planet.isCentralBody = (
            True  # ensure this is the central gravitational body
        )
        self.planet.useSphericalHarmonicsGravityModel(
            bsk_path + "/supportData/LocalGravData/GGM03S.txt", 10
        )

        # setup Spice interface for some solar system bodies
        timeInitString = utc_init
        self.gravFactory.createSpiceInterface(
            bsk_path + "/supportData/EphemerisData/", timeInitString, epochInMsg=True
        )
        self.gravFactory.spiceObject.zeroBase = "earth"

        self.simulator.AddModelToTask(
            self.world_task_name, self.gravFactory.spiceObject, ModelPriority=priority
        )

    def setup_ephem_object(self, priority: int = 988, **kwargs) -> None:
        """Set up the ephemeris object to use with the SPICE library.

        Args:
            priority: Model priority.
            **kwargs: Passed to other setup functions.
        """
        self.ephemConverter = ephemerisConverter.EphemerisConverter()
        self.ephemConverter.ModelTag = "ephemConverter"
        self.ephemConverter.addSpiceInputMsg(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.sun_index]
        )
        self.ephemConverter.addSpiceInputMsg(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.body_index]
        )
        self.simulator.AddModelToTask(
            self.world_task_name, self.ephemConverter, ModelPriority=priority
        )

    @default_args(
        planetRadius=orbitalMotion.REQ_EARTH * 1e3,
        baseDensity=1.22,
        scaleHeight=8e3,
    )
    def setup_atmosphere_density_model(
        self,
        planetRadius: float,
        baseDensity: float,
        scaleHeight: float,
        priority: int = 1000,
        **kwargs,
    ) -> None:
        """Set up the exponential gravity model.

        Args:
            planetRadius: [m] Planet ground radius.
            baseDensity: [kg/m^3] Exponential model parameter.
            scaleHeight: [m] Exponential model parameter.
            priority: Model priority.
            **kwargs: Passed to other setup functions.
        """
        self.densityModel = exponentialAtmosphere.ExponentialAtmosphere()
        self.densityModel.ModelTag = "expDensity"
        self.densityModel.planetRadius = planetRadius
        self.densityModel.baseDensity = baseDensity
        self.densityModel.scaleHeight = scaleHeight
        self.densityModel.planetPosInMsg.subscribeTo(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.body_index]
        )
        self.simulator.AddModelToTask(
            self.world_task_name, self.densityModel, ModelPriority=priority
        )

    def setup_eclipse_object(self, priority: int = 988, **kwargs) -> None:
        """Set up the celestial object that is causing an eclipse message.

        Args:
            priority: Model priority.
            kwargs: Ignored
        """
        self.eclipseObject = eclipse.Eclipse()
        self.eclipseObject.addPlanetToModel(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.body_index]
        )
        self.eclipseObject.sunInMsg.subscribeTo(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.sun_index]
        )
        self.simulator.AddModelToTask(
            self.world_task_name, self.eclipseObject, ModelPriority=priority
        )

    def __del__(self) -> None:
        """Log when world is deleted and unload SPICE."""
        super().__del__()
        try:
            self.gravFactory.unloadSpiceKernels()
        except AttributeError:
            pass


class GroundStationWorldModel(BasicWorldModel):
    """Model that includes downlink ground stations."""

    def __init__(self, *args, **kwargs) -> None:
        """Model that includes downlink ground stations.

        This model includes the basic world components, as well as ground stations for
        downlinking data.

        Args:
            *args: Passed to superclass.
            **kwargs: Passed to superclass.
        """
        super().__init__(*args, **kwargs)

    def _setup_world_objects(self, **kwargs) -> None:
        super()._setup_world_objects(**kwargs)
        self.setup_ground_locations(**kwargs)

    @default_args(
        groundStationsData=[
            dict(name="Boulder", lat=40.009971, long=-105.243895, elev=1624),
            dict(name="Merritt", lat=28.3181, long=-80.6660, elev=0.9144),
            dict(name="Singapore", lat=1.3521, long=103.8198, elev=15),
            dict(name="Weilheim", lat=47.8407, long=11.1421, elev=563),
            dict(name="Santiago", lat=-33.4489, long=-70.6693, elev=570),
            dict(name="Dongara", lat=-29.2452, long=114.9326, elev=34),
            dict(name="Hawaii", lat=19.8968, long=-155.5828, elev=9),
        ],
        groundLocationPlanetRadius=orbitalMotion.REQ_EARTH * 1e3,
        gsMinimumElevation=np.radians(10.0),
        gsMaximumRange=-1,
    )
    def setup_ground_locations(
        self,
        groundStationsData: list[dict[str, Union[str, float]]],
        groundLocationPlanetRadius: float,
        gsMinimumElevation: float,
        gsMaximumRange: float,
        priority: int = 1399,
        **kwargs,
    ) -> None:
        """Specify the ground locations of interest.

        Args:
            groundStationsData: List of dictionaries of ground station data. Each dictionary
                must include keys for ``lat`` and ``long`` [deg], and may include
                ``elev`` [m], ``name``. For example:

                .. code-block:: python

                    groundStationsData=[
                        dict(name="Boulder", lat=40.009971, long=-105.243895, elev=1624),
                        dict(lat=28.3181, long=-80.6660),
                    ]

                ``groundLocationPlanetRadius``, ``gsMinimumElevation``, and ``gsMaximumRange``
                may also be specified in the dictionary to override the global values
                for those parameters for a specific ground station.

            groundLocationPlanetRadius: [m] Radius of ground locations from center of
                planet.
            gsMinimumElevation: [rad] Minimum elevation angle from station to satellite
                to be able to downlink data.
            gsMaximumRange: [m] Maximum range from station to satellite when
                downlinking. Set to ``-1`` to disable.
            priority: Model priority.
            kwargs: Passed to other setup functions.
        """
        self.groundStations = []
        self.groundLocationPlanetRadius = groundLocationPlanetRadius
        self.gsMinimumElevation = gsMinimumElevation
        self.gsMaximumRange = gsMaximumRange
        for i, groundStationData in enumerate(groundStationsData):
            self._create_ground_station(
                **groundStationData, priority=priority - i)

    def _create_ground_station(
        self,
        lat: float,
        long: float,
        elev: float = 0,
        name: Optional[str] = None,
        groundLocationPlanetRadius: Optional[float] = None,
        gsMinimumElevation: Optional[float] = None,
        gsMaximumRange: Optional[float] = None,
        priority: int = 1399,
    ) -> None:
        """Add a ground station with given parameters.

        Args:
            lat: [deg] Latitude.
            long: [deg] Longitude.
            elev: [m] Elevation.
            name: Ground station identifier.
            groundLocationPlanetRadius: [m] Radius of planet.
            gsMinimumElevation: [rad] Minimum elevation angle to downlink to ground station.
            gsMaximumRange: [m] Maximum range to downlink to ground station. Set to ``-1`` for infinity.
            priority: Model priority.
        """
        if name is None:
            name = str(len(self.groundStations))

        groundStation = groundLocation.GroundLocation()
        groundStation.ModelTag = "GroundStation" + name
        if groundLocationPlanetRadius:
            groundStation.planetRadius = groundLocationPlanetRadius
        else:
            groundStation.planetRadius = self.groundLocationPlanetRadius
        groundStation.specifyLocation(np.radians(lat), np.radians(long), elev)
        groundStation.planetInMsg.subscribeTo(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.body_index]
        )
        if gsMinimumElevation:
            groundStation.minimumElevation = gsMinimumElevation
        else:
            groundStation.minimumElevation = self.gsMinimumElevation
        if gsMaximumRange:
            groundStation.maximumRange = gsMaximumRange
        else:
            groundStation.maximumRange = self.gsMaximumRange
        self.groundStations.append(groundStation)

        self.simulator.AddModelToTask(
            self.world_task_name, groundStation, ModelPriority=priority
        )


class ManyGroundStationWorldModel(BasicWorldModel):
    """Model that includes downlink ground stations."""

    def __init__(self, *args, **kwargs) -> None:
        """Model that includes downlink ground stations.

        This model includes the basic world components, as well as ground stations for
        downlinking data.

        Args:
            *args: Passed to superclass.
            **kwargs: Passed to superclass.
        """
        super().__init__(*args, **kwargs)

    def _setup_world_objects(self, **kwargs) -> None:
        super()._setup_world_objects(**kwargs)
        self.setup_ground_locations(**kwargs)

    @default_args(
        groundStationsData=[
            # North America
            dict(name="Boulder", lat=40.009971, long=-105.243895, elev=1624),
            dict(name="Anchorage", lat=61.2181, long=-149.9003, elev=31),
            dict(name="Toronto", lat=43.65107, long=-79.347015, elev=76),
            dict(name="Mexico City", lat=19.4326, long=-99.1332, elev=2240),
            dict(name="Los Angeles", lat=34.0522, long=-118.2437, elev=71),
            dict(name="Chicago", lat=41.8781, long=-87.6298, elev=181),
            dict(name="Miami", lat=25.7617, long=-80.1918, elev=2),
            dict(name="Vancouver", lat=49.2827, long=-123.1207, elev=70),

            # South America
            dict(name="Santiago", lat=-33.4489, long=-70.6693, elev=570),
            dict(name="Buenos Aires", lat=-34.6037, long=-58.3816, elev=25),
            dict(name="Brasilia", lat=-15.8267, long=-47.9218, elev=1172),
            dict(name="Bogotá", lat=4.7110, long=-74.0721, elev=2640),
            dict(name="Lima", lat=-12.0464, long=-77.0428, elev=154),
            dict(name="Caracas", lat=10.4806, long=-66.9036, elev=900),
            dict(name="Montevideo", lat=-34.9011, long=-56.1645, elev=43),

            # Europe
            dict(name="Weilheim", lat=47.8407, long=11.1421, elev=563),
            dict(name="Reykjavik", lat=64.1355, long=-21.8954, elev=45),
            dict(name="Madrid", lat=40.4168, long=-3.7038, elev=667),
            dict(name="Moscow", lat=55.7558, long=37.6173, elev=144),
            dict(name="Paris", lat=48.8566, long=2.3522, elev=35),
            dict(name="Rome", lat=41.9028, long=12.4964, elev=21),
            dict(name="London", lat=51.5074, long=-0.1278, elev=24),
            dict(name="Berlin", lat=52.5200, long=13.4050, elev=34),

            # Africa
            dict(name="Cape Town", lat=-33.9249, long=18.4241, elev=15),
            dict(name="Nairobi", lat=-1.2864, long=36.8172, elev=1795),
            dict(name="Lagos", lat=6.5244, long=3.3792, elev=41),
            dict(name="Cairo", lat=30.0444, long=31.2357, elev=23),
            dict(name="Accra", lat=5.6037, long=-0.1870, elev=61),
            dict(name="Johannesburg", lat=-26.2041, long=28.0473, elev=1753),
            dict(name="Algiers", lat=36.7372, long=3.0863, elev=24),
            dict(name="Addis Ababa", lat=9.0300, long=38.7400, elev=2355),

            # Asia
            dict(name="Beijing", lat=39.9042, long=116.4074, elev=43),
            dict(name="Mumbai", lat=19.0760, long=72.8777, elev=14),
            dict(name="Tokyo", lat=35.6895, long=139.6917, elev=40),
            dict(name="Bangkok", lat=13.7563, long=100.5018, elev=1.5),
            dict(name="Singapore", lat=1.3521, long=103.8198, elev=15),
            dict(name="Seoul", lat=37.5665, long=126.9780, elev=38),
            dict(name="Manila", lat=14.5995, long=120.9842, elev=16),
            dict(name="Jakarta", lat=-6.2088, long=106.8456, elev=8),

            # Oceania
            dict(name="Sydney", lat=-33.8688, long=151.2093, elev=58),
            dict(name="Perth", lat=-31.9505, long=115.8605, elev=34),
            dict(name="Auckland", lat=-36.8485, long=174.7633, elev=7),
            dict(name="Canberra", lat=-35.2809, long=149.1300, elev=577),
            dict(name="Melbourne", lat=-37.8136, long=144.9631, elev=31),
            dict(name="Brisbane", lat=-27.4698, long=153.0251, elev=27),
            dict(name="Port Moresby", lat=-9.4438, long=147.1803, elev=58),
            dict(name="Honiara", lat=-9.4295, long=159.9556, elev=29),

            # Antarctica
            dict(name="McMurdo Station", lat=-77.8419, long=166.6863, elev=10),
            dict(name="South Pole Station", lat=-90.0, long=0.0, elev=2835),

            # Oceans and Remote Areas
            dict(name="Central Pacific Ocean", lat=0.0, long=-140.0, elev=0),
            dict(name="South Pacific Ocean", lat=-15.0, long=150.0, elev=0),
            dict(name="North Atlantic Ocean", lat=25.0, long=-45.0, elev=0),
            dict(name="South Atlantic Ocean", lat=-30.0, long=-10.0, elev=0),
            dict(name="Indian Ocean Station", lat=-10.0, long=70.0, elev=0),
            dict(name="Southern Ocean Station", lat=-60.0, long=20.0, elev=0),
            dict(name="Central Arctic Ocean Station",
                 lat=85.0, long=0.0, elev=0),

            # Additional Ocean Coverage
            dict(name="Western Pacific", lat=30.0, long=150.0, elev=0),
            dict(name="Eastern Indian Ocean", lat=-20.0, long=100.0, elev=0),
            dict(name="Central Atlantic Ocean", lat=0.0, long=-30.0, elev=0),
            dict(name="Eastern Atlantic Ocean", lat=-15.0, long=-15.0, elev=0),
            dict(name="Northern Pacific", lat=45.0, long=-160.0, elev=0),
            dict(name="Mid-Southern Ocean", lat=-45.0, long=50.0, elev=0),
            dict(name="Western Atlantic", lat=10.0, long=-55.0, elev=0),
            dict(name="Northern Indian Ocean", lat=10.0, long=80.0, elev=0),
        ],
        groundLocationPlanetRadius=orbitalMotion.REQ_EARTH * 1e3,
        gsMinimumElevation=np.radians(10.0),
        gsMaximumRange=-1,
    )
    def setup_ground_locations(
        self,
        groundStationsData: list[dict[str, Union[str, float]]],
        groundLocationPlanetRadius: float,
        gsMinimumElevation: float,
        gsMaximumRange: float,
        priority: int = 1399,
        **kwargs,
    ) -> None:
        """Specify the ground locations of interest.

        Args:
            groundStationsData: List of dictionaries of ground station data. Each dictionary
                must include keys for ``lat`` and ``long`` [deg], and may include
                ``elev`` [m], ``name``. For example:

                .. code-block:: python

                    groundStationsData=[
                        dict(name="Boulder", lat=40.009971, long=-105.243895, elev=1624),
                        dict(lat=28.3181, long=-80.6660),
                    ]

                ``groundLocationPlanetRadius``, ``gsMinimumElevation``, and ``gsMaximumRange``
                may also be specified in the dictionary to override the global values
                for those parameters for a specific ground station.

            groundLocationPlanetRadius: [m] Radius of ground locations from center of
                planet.
            gsMinimumElevation: [rad] Minimum elevation angle from station to satellite
                to be able to downlink data.
            gsMaximumRange: [m] Maximum range from station to satellite when
                downlinking. Set to ``-1`` to disable.
            priority: Model priority.
            kwargs: Passed to other setup functions.
        """
        self.groundStations = []
        self.groundLocationPlanetRadius = groundLocationPlanetRadius
        self.gsMinimumElevation = gsMinimumElevation
        self.gsMaximumRange = gsMaximumRange
        for i, groundStationData in enumerate(groundStationsData):
            self._create_ground_station(
                **groundStationData, priority=priority - i)

    def _create_ground_station(
        self,
        lat: float,
        long: float,
        elev: float = 0,
        name: Optional[str] = None,
        groundLocationPlanetRadius: Optional[float] = None,
        gsMinimumElevation: Optional[float] = None,
        gsMaximumRange: Optional[float] = None,
        priority: int = 1399,
    ) -> None:
        """Add a ground station with given parameters.

        Args:
            lat: [deg] Latitude.
            long: [deg] Longitude.
            elev: [m] Elevation.
            name: Ground station identifier.
            groundLocationPlanetRadius: [m] Radius of planet.
            gsMinimumElevation: [rad] Minimum elevation angle to downlink to ground station.
            gsMaximumRange: [m] Maximum range to downlink to ground station. Set to ``-1`` for infinity.
            priority: Model priority.
        """
        if name is None:
            name = str(len(self.groundStations))

        groundStation = groundLocation.GroundLocation()
        groundStation.ModelTag = "GroundStation" + name
        if groundLocationPlanetRadius:
            groundStation.planetRadius = groundLocationPlanetRadius
        else:
            groundStation.planetRadius = self.groundLocationPlanetRadius
        groundStation.specifyLocation(np.radians(lat), np.radians(long), elev)
        groundStation.planetInMsg.subscribeTo(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.body_index]
        )
        if gsMinimumElevation:
            groundStation.minimumElevation = gsMinimumElevation
        else:
            groundStation.minimumElevation = self.gsMinimumElevation
        if gsMaximumRange:
            groundStation.maximumRange = gsMaximumRange
        else:
            groundStation.maximumRange = self.gsMaximumRange
        self.groundStations.append(groundStation)

        self.simulator.AddModelToTask(
            self.world_task_name, groundStation, ModelPriority=priority
        )


__doc_title__ = "World Sims"
__all__ = ["WorldModel", "BasicWorldModel", "GroundStationWorldModel"]
