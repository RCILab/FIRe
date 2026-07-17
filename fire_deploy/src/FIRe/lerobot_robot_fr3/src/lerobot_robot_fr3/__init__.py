from .tasks.base_task import Task
from .tasks.factory import Factory
from .tasks.forge import Forge
from .tasks.pick_place import PickPlace
# from .tasks.my_task import MyTask


def create_task(task_name: str) -> Task:
    # factory
    if task_name == "factory-peg_insert":
        return Factory(name="peg_insert")
    elif task_name == "factory-gear_mesh":
        return Factory(name="gear_mesh")
    elif task_name == "factory-nut_thread":
        return Factory(name="nut_thread")
    # forge
    elif task_name == "forge-peg_insert":
        return Forge(name="peg_insert")
    elif task_name == "forge-gear_mesh":
        return Forge(name="gear_mesh")
    elif task_name == "forge-nut_thread":
        return Forge(name="nut_thread")
    # pick_place
    elif task_name == "pick_place":
        return PickPlace(name="pick_place")
    # extras
    # elif task_name == "my_task":
    #     return MyTask(name="my_task")
    else:
        raise ValueError(f"Unknown task name: {task_name}")
    
def get_model_cfg_path(task_name: str) -> str:
    task = create_task(task_name)
    return task.model_cfg_path