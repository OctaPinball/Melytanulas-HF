import gradio as gr
import cv2
import numpy as np
import os
from model_list import models, all_model_name, default_model_paths
from parameters import MODEL_PATH
import torch
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Placeholder function to simulate model inference
def model_inference(slices, model_name):
    n1 = 112
    number_of_slices = len(slices)
    midpoint = slices[0].shape[0] // 2
    n11, n12 = midpoint - int(n1 / 2), midpoint + int(n1 / 2)
    models[model_name].eval()
    input1 = np.zeros(shape=[number_of_slices, n1, n1])
    for n in range(number_of_slices):
        ImageIn = slices[n]
        input1[n, :, :] = ImageIn[n11:n12, n11:n12]
    output = np.zeros(shape=[number_of_slices, 2, slices[0].shape[0], slices[0].shape[1]])  # Correct shape with 2 channels
    for n in range(number_of_slices):
        img_tensor = torch.tensor(input1[n], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            out = models[model_name](img_tensor)

            # Apply softmax to get probabilities
            out = F.softmax(out, dim=1)

            # Resize the output to match the expected size (112, 112)
            out = F.interpolate(out, size=(n1, n1), mode='bilinear', align_corners=False)

            # Assign the result directly
            output[n, :, n11:n12, n11:n12] = out.cpu().numpy()

    # Perform argmax over the class dimension (axis 1)
    output = np.argmax(output, 1)
    return np.uint8(255 * output)

def majority_voting(model_outputs):
    stacked_outputs = np.stack(model_outputs, axis=-1)  # Stack outputs along a new dimension
    majority = np.mean(stacked_outputs, axis=-1) > 0.5  # Majority vote
    return majority.astype(np.uint8) * 255  # Scale to 0-255

def process_uploaded_tiffs(uploaded_files, model_names):
    try:
        print(f"Uploaded files: {[file.name for file in uploaded_files]}")
        slices = [cv2.imread(file.name, cv2.IMREAD_GRAYSCALE) for file in uploaded_files]
        if not slices or any(slice_img is None for slice_img in slices):
            print("Error: One or more files could not be read as TIFF.")
            raise ValueError("One or more uploaded files could not be read as valid TIFF images.")
        
        print(f"Number of slices loaded: {len(slices)}")
        
        model_outputs = [model_inference(slices, model_name) for model_name in model_names]
        combined_outputs = [majority_voting([model_outputs[m][i] for m in range(len(model_names))]) for i in range(len(slices))]

        print("Processing complete.")
        return slices, model_outputs, combined_outputs

    except Exception as e:
        print(f"Error during TIFF processing: {e}")
        return None, None, None

def viewer(slice_idx, input_slices, model_outputs, combined_outputs):
    max_idx = len(input_slices) - 1
    slice_idx = max(0, min(slice_idx, max_idx))  # Clamp index within range

    input_slice = input_slices[slice_idx]
    model_slices = [model_output[slice_idx] for model_output in model_outputs]
    combined_slice = combined_outputs[slice_idx]

    return [input_slice] + model_slices + [combined_slice]

def gradio_interface(model_names):
    with gr.Blocks() as demo:
        # Input and process button
        with gr.Row():
            file_uploader = gr.File(
                file_types=[".tiff"], 
                file_count="multiple", 
                label="Upload MRI Slices",
                elem_id="file_uploader"  # Id for custom styling
            )
            process_button = gr.Button("Process")

        # Slice slider
        slice_slider = gr.Slider(
            minimum=0,
            maximum=0,
            step=1,
            value=0,
            label="Slice Index"
        )

        # Image outputs
        with gr.Row():
            input_image = gr.Image(label="Input Slice")
            model_outputs_images = [gr.Image(label=f"{model_name} Output") for model_name in model_names]
            ensemble_image = gr.Image(label="Combined Output")

        # Define state variables
        input_slices_state = gr.State(value=None)
        model_outputs_state = gr.State(value=None)
        combined_outputs_state = gr.State(value=None)

        # Process function
        def process(uploaded_files):
            if not uploaded_files:
                print("No files uploaded.")
                return None, None, None, gr.update(value=0, minimum=0, maximum=0)

            input_slices, model_outputs, combined_outputs = process_uploaded_tiffs(uploaded_files, model_names)

            if input_slices is None or model_outputs is None or combined_outputs is None:
                print("TIFF processing failed. Returning default values.")
                return None, None, None, gr.update(value=0, minimum=0, maximum=0)

            slice_count = len(input_slices)
            return (
                input_slices,
                model_outputs,
                combined_outputs,
                gr.update(value=0, minimum=0, maximum=slice_count - 1)
            )

        # Update function
        def update(slice_idx, input_slices, model_outputs, combined_outputs):
            if slice_idx is None:
                slice_idx = 0
            if not input_slices or not model_outputs or not combined_outputs:
                raise ValueError("No valid data available for display. Please check the uploaded files and processing.")
            return viewer(slice_idx, input_slices, model_outputs, combined_outputs)

        # Connect functionality
        process_button.click(
            process,
            inputs=[file_uploader],
            outputs=[
                input_slices_state,
                model_outputs_state,
                combined_outputs_state,
                slice_slider
            ],
        )

        slice_slider.change(
            update,
            inputs=[slice_slider, input_slices_state, model_outputs_state, combined_outputs_state],
            outputs=[input_image] + model_outputs_images + [ensemble_image],
        )

        # Custom HTML with CSS to make the file uploader scrollable with a max height
        gr.HTML("""
            <style>
                #file_uploader {
                    max-height: 200px;  /* Set the max height */
                    overflow-y: auto;   /* Enable vertical scrolling */
                }
            </style>
        """)

    return demo



for model_name in all_model_name:
    models[model_name].load_state_dict(torch.load(os.path.join(MODEL_PATH, default_model_paths[model_name])))
    models[model_name] = models[model_name].to(device)
interface = gradio_interface(all_model_name)
interface.launch(share=True)

