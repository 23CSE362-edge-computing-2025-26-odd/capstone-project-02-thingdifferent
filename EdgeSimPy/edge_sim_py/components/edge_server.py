""" Contains edge-server-related functionality."""
# EdgeSimPy components
from edge_sim_py.component_manager import ComponentManager
from edge_sim_py.components.network_flow import NetworkFlow
from edge_sim_py.components.container_registry import ContainerRegistry
from edge_sim_py.components.container_image import ContainerImage
from edge_sim_py.components.container_layer import ContainerLayer


# Mesa modules
from mesa import Agent


# Python libraries
import networkx as nx
import typing



class EdgeServer(ComponentManager, Agent):
    """Class that represents an edge server."""

    # Class attributes that allow this class to use helper methods from the ComponentManager
    _instances = []
    _object_count = 0

    def __init__(
        self,
        obj_id: int = None,
        coordinates: tuple = None,
        model_name: str = "",
        cpu: int = 0,
        memory: int = 0,
        disk: int = 0,
        power_model: typing.Callable = None,
    ) -> object:
        """Creates an EdgeServer object.

        Args:
            obj_id (int, optional): Object identifier.
            coordinates (tuple, optional): 2-tuple that represents the edge server coordinates.
            model_name (str, optional): Edge server model name. Defaults to "".
            cpu (int, optional): Edge server's CPU capacity. Defaults to 0.
            memory (int, optional): Edge server's memory capacity. Defaults to 0.
            disk (int, optional): Edge server's disk capacity. Defaults to 0.
            power_model (typing.Callable, optional): Edge server power model. Defaults to None.

        Returns:
            object: Created EdgeServer object.
        """
        # Adding the new object to the list of instances of its class
        self.__class__._instances.append(self)

        # Object's class instance ID
        self.__class__._object_count += 1
        if obj_id is None:
            obj_id = self.__class__._object_count
        self.id = obj_id

        # Edge server model name
        self.model_name = model_name

        # Edge server base station
        self.base_station = None

        # Edge server network switch
        self.network_switch = None

        # Edge server coordinates
        self.coordinates = coordinates

        # Edge server capacity
        self.cpu = cpu
        self.memory = memory
        self.disk = disk

        # Edge server demand
        self.cpu_demand = 0
        self.memory_demand = 0
        self.disk_demand = 0

        # Edge server's availability status
        self.available = True

        # Number of active migrations involving the edge server
        self.ongoing_migrations = 0

        # Power Features
        self.active = True
        self.power_model = power_model
        self.power_model_parameters = {}

        # Container registries and services hosted by the edge server
        self.container_registries = []
        self.services = []

        # Container images and container layers hosted by the edge server
        self.container_images = []
        self.container_layers = []

        # Lists that control the layers being pulled to the edge server
        self.waiting_queue = []
        self.download_queue = []

        # Number of container layers the edge server can download simultaneously (default = 3)
        self.max_concurrent_layer_downloads = 3

        # Model-specific attributes (defined inside the model's "initialize()" method)
        self.model = None
        self.unique_id = None

        # List of currently processing tasks with resource usage and time left
        self.processing_tasks = []

    @property
    def task_queue(self):
        # Return a dummy list for compatibility, each element representing one active task
        return [1] * len(self.processing_tasks)

    def assign_tasks(self, tasks, cpu_per_task=1, mem_per_task=0.5, disk_per_task=0.2, duration=3):
        """
        Assign new tasks to this edge server.

        Args:
            tasks (int): Number of tasks to assign.
            cpu_per_task (float): CPU demand per task.
            mem_per_task (float): Memory demand per task.
            disk_per_task (float): Disk demand per task.
            duration (int): Task duration in time steps.
        """
        for _ in range(tasks):
            self.processing_tasks.append({
                'cpu': cpu_per_task,
                'mem': mem_per_task,
                'disk': disk_per_task,
                'remaining': duration
            })
            self.cpu_demand += cpu_per_task
            self.memory_demand += mem_per_task
            self.disk_demand += disk_per_task

    def process_time_step(self):
        """
        Process the passing of one time step, decrement task remaining times,
        and release resources of finished tasks.
        """
        finished_tasks = []
        for task in self.processing_tasks:
            task['remaining'] -= 1
            if task['remaining'] <= 0:
                finished_tasks.append(task)
        for task in finished_tasks:
            self.processing_tasks.remove(task)
            self.cpu_demand -= task['cpu']
            self.memory_demand -= task['mem']
            self.disk_demand -= task['disk']

    def step(self):
        """Method that executes the events involving the object at each time step."""
        # Process downloading container layers as per original logic
        while len(self.waiting_queue) > 0 and len(self.download_queue) < self.max_concurrent_layer_downloads:
            layer = self.waiting_queue.pop(0)

            # Gathering the list of registries that have the layer
            registries_with_layer = []
            for registry in [reg for reg in ContainerRegistry.all() if reg.available]:
                if registry.server and any(layer.digest == l.digest for l in registry.server.container_layers):
                    path = nx.shortest_path(
                        G=self.model.topology,
                        source=registry.server.base_station.network_switch,
                        target=self.base_station.network_switch,
                    )
                    registries_with_layer.append({"object": registry, "path": path})

            registries_with_layer = sorted(registries_with_layer, key=lambda r: len(r["path"]))
            registry = registries_with_layer[0]["object"]
            path = registries_with_layer[0]["path"]

            flow = NetworkFlow(
                topology=self.model.topology,
                source=registry.server,
                target=self,
                start=self.model.schedule.steps + 1,
                path=path,
                data_to_transfer=layer.size,
                metadata={"type": "layer", "object": layer, "container_registry": registry},
            )
            self.model.initialize_agent(agent=flow)

            self.download_queue.append(flow)

        # Process currently running tasks: advance and release resources accordingly
        self.process_time_step()

    # The rest of your existing methods unchanged...


    def get_power_consumption(self) -> float:
        """Gets the edge server's power consumption.

        Returns:
            power_consumption (float): Edge server's power consumption.
        """
        power_consumption = self.power_model.get_power_consumption(device=self) if self.power_model is not None else 0
        return power_consumption

    # ... Rest of the existing methods unchanged ...


    def has_capacity_to_host(self, service: object) -> bool:
        """Checks if the edge server has enough free resources to host a given service.

        Args:
            service (object): Service object that we are trying to host on the edge server.

        Returns:
            can_host (bool): Information of whether the edge server has capacity to host the service or not.
        """
        # Calculating the additional disk demand that would be incurred to the edge server
        additional_disk_demand = self._get_disk_demand_delta(service=service)

        # Calculating the edge server's free resources
        free_cpu = self.cpu - self.cpu_demand
        free_memory = self.memory - self.memory_demand
        free_disk = self.disk - self.disk_demand

        # Checking if the host would have resources to host the registry and its (additional) layers
        can_host = free_cpu >= service.cpu_demand and free_memory >= service.memory_demand and free_disk >= additional_disk_demand
        return can_host

    def _add_container_image(self, template_container_image: object) -> object:
        """Adds a new container image to the edge server based on the specifications of an existing image.
        Args:
            template_container_image (object): Template container image.

        Returns:
            image (ContainerImage): New ContainerImage object.
        """
        # Checking if the edge server has no existing instance representing the same container image
        digest = template_container_image.digest
        if digest in [image.digest for image in self.container_images]:
            raise Exception(f"Failed in adding an image to {self} as it already hosts a image with the same digest ({digest}).")

        # Checking if the edge server has all the container layers that compose the container image
        for layer_digest in template_container_image.layers_digests:
            if not any([layer_digest == layer.digest for layer in self.container_layers]):
                raise Exception(
                    f"Failed in adding an image to {self} as it does not hosts all the layers necessary ({layer_digest})."
                )

        # Creating a ContainerImage object to represent the new image
        image = ContainerImage()
        image.name = template_container_image.name
        image.digest = template_container_image.digest
        image.tag = template_container_image.tag
        image.architecture = template_container_image.architecture
        image.layers_digests = template_container_image.layers_digests

        # Connecting the new image to the target host
        image.server = self
        self.container_images.append(image)

        # Adding the new ContainerImage object to the list of simulator agents
        self.model.initialize_agent(agent=image)

        return image

    def _get_uncached_layers(self, service: object) -> list:
        """Gets the list of container layers from a given service that are not present in the edge server's layers cache list.

        Args:
            service (object): Service whose disk demand delta will be calculated.

        Returns:
            uncached_layers (float): List of layers from service's image not present in the edge server's layers cache list.
        """
        # Gathering layers present in the target server (layers, download_queue, waiting_queue)
        layers_downloaded = [layer for layer in self.container_layers]
        layers_on_download_queue = [flow.metadata["object"] for flow in self.download_queue if flow.metadata["object"] == "layer"]
        layers_on_waiting_queue = [layer for layer in self.waiting_queue]
        layers = layers_downloaded + layers_on_download_queue + layers_on_waiting_queue

        # Gathering the service's container image
        service_image = ContainerImage.find_by(attribute_name="digest", attribute_value=service.image_digest)

        # Gathering the list of uncached layers
        uncached_layers = []
        for layer_digest in service_image.layers_digests:
            if not any(layer_digest == layer.digest for layer in layers):
                layer = ContainerLayer.find_by(attribute_name="digest", attribute_value=layer_digest)
                if layer not in uncached_layers:
                    uncached_layers.append(layer)

        return uncached_layers

    def _get_disk_demand_delta(self, service: object) -> float:
        """Calculates the additional disk demand necessary to host a registry inside the edge server considering
        the list of cached layers inside the edge server and the layers that compose the service's image.

        Args:
            service (object): Service whose disk demand delta will be calculated.

        Returns:
            disk_demand_delta (float): Disk demand delta.
        """
        # Gathering the list of layers that compose the service's image that are not present in the edge server
        uncached_layers = self._get_uncached_layers(service=service)

        # Calculating the amount of disk resources required by all service layers not present in the host's disk
        disk_demand_delta = sum([layer.size for layer in uncached_layers])

        return disk_demand_delta
