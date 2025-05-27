from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.env import MAX_ACTIVE_MODELS
from inference.core.managers.base import ModelManager
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.models.utils import ROBOFLOW_MODEL_TYPES
from inference.core.workflows.execution_engine.core  import ExecutionEngine


# initialisation of Model registry to manage models load into memory
# (required by core blocks exposing Roboflow models)
model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
model_manager = ModelManager(model_registry=model_registry)
model_manager = WithFixedSizeCache(model_manager, max_size=MAX_ACTIVE_MODELS)

# workflow definition
# Workflows in Python package
# https://inference.roboflow.com/workflows/modes_of_running/#workflows-in-python-package
OBJECT_DETECTION_WORKFLOW = {
  "version": "1.0",
  "inputs": [
    {
      "type": "InferenceImage",
      "name": "image"
    }
  ],
  "steps": [
    {
      "type": "roboflow_core/roboflow_object_detection_model@v2",
      "name": "model",
      "images": "$inputs.image",
      "model_id": "yolov11n-640"
    },
    {
      "type": "roboflow_core/bounding_box_visualization@v1",
      "name": "bounding_box_visualization",
      "image": "$inputs.image",
      "predictions": "$steps.model.predictions",
      "thickness": 1
    }
  ],
  "outputs": [
    {
      "type": "JsonField",
      "name": "output",
      "coordinates_system": "own",
      "selector": "$steps.model.predictions"
    }
  ]
}

# example init paramaters for blocks - dependent on set of blocks
# used in your workflow
workflow_init_parameters = {
    "workflows_core.model_manager": model_manager,
    "workflows_core.api_key": 'K5WvJpndyK2UDTl7eNYB',
    "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
}

# instance of Execution Engine - init(...) method invocation triggers
# the compilation process
execution_engine = ExecutionEngine.init(
    workflow_definition=OBJECT_DETECTION_WORKFLOW,
    init_parameters=workflow_init_parameters,
    max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
)

result = execution_engine.run(
    runtime_parameters={
        "image": ['09.jpeg'],
        "model_id": "yolov8n-640",
    }
)
print(result)

# runing the workflow
for i in range(0, 1000):
    result = execution_engine.run(
        runtime_parameters={
            "image": ['09.jpeg'],
            "model_id": "yolov8n-640",
        }
    )
    print(i, result)


