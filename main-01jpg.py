from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="http://localhost:9010", # use local inference server
    api_key="K5WvJpndyK2UDTl7eNYB" # optional to access your private data and models
)
result = client.run_workflow(
    workspace_name="m-insvj",
    workflow_id="jast-car-plate-detect",
    images={
        "image": "01.jpg"
    }
)
print(result)
# for i in range(0, 1000):
#     result = client.run_workflow(
#         workspace_name="m-insvj",
#         workflow_id="jast-car-plate-detect",
#         images={
#             "image": "01.jpg"
#         }
#     )
#     print(i, result)



# from inference import InferencePipeline
# from inference.core.interfaces.stream.sinks import render_boxes
# import cv2
#
#
# def custum_sink(result, video_frame):
#     if result.get("model_predictions"):  # model_predictions
#         cv2.imshow("", result["model_predictions"].numpy_image)
#         cv2.waitKey(1)
#
#
# pipeline = InferencePipeline.init_with_workflow(
#     api_key="K5WvJpndyK2UDTl7eNYB",
#     workspace_name="m-insvj",
#     workflow_id="jast-car-plate-detect",  # Roboflow model to use
#     video_reference="output.mp4",  # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
#     on_prediction=custum_sink  # Function to run after each prediction
# )
# pipeline.start()
# pipeline.join()