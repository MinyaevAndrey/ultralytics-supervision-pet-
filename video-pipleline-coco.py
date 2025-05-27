# Import the InferencePipeline object
from inference import InferencePipeline
import cv2

def my_sink(result, video_frame):
    if result.get("bounding_box_visualization"): # Display an image from the workflow response
        cv2.imshow("Workflow Image", result["bounding_box_visualization"].numpy_image)
        cv2.waitKey(1)
    print(result) # do something with the predictions of each frame


# initialize a pipeline object
pipeline = InferencePipeline.init_with_workflow(
    api_key="K5WvJpndyK2UDTl7eNYB",
    workspace_name="m-insvj",
    workflow_id="simple-coco-detection",
    video_reference='output.mp4', # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
    max_fps=30,
    on_prediction=my_sink
)
pipeline.start() #start the pipeline
pipeline.join() #wait for the pipeline thread to finish
