from inference_sdk import InferenceHTTPClient
import atexit
import time
import cv2


client = InferenceHTTPClient(
    api_url="http://localhost:9010", # use local inference server
    api_key="K5WvJpndyK2UDTl7eNYB" # optional to access your private data and models
)

# Start a stream on an rtsp stream
result = client.start_inference_pipeline_with_workflow(
    video_reference=["http://61.211.241.239/nphMotionJpeg?Resolution=320x240&Quality=Standard"],
    workspace_name="m-insvj",
    workflow_id="simple-coco-detection",
    max_fps=30
)

pipeline_id = result["context"]["pipeline_id"]

# Terminate the pipeline when the script exits
atexit.register(lambda: client.terminate_inference_pipeline(pipeline_id))

while True:
  result = client.consume_inference_pipeline_result(pipeline_id=pipeline_id)

  if not result["bounding_box_visualization"] or not result["bounding_box_visualization"][0]:
    # still initializing
    continue

  print(result)

  time.sleep(1/30)