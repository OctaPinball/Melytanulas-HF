import gradio as gr
import cv2
import numpy as np
import os
from model_list import models, all_model_name, default_model_paths
from parameters import MODEL_PATH
import torch
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_dice_score(prediction, ground_truth):
    """
    Compute Dice score.
    Args:
        prediction (np.array): Binary prediction array.
        ground_truth (np.array): Binary ground truth array.
    Returns:
        float: Dice score.
    """
    prediction = prediction > 0
    ground_truth = ground_truth > 0
    intersection = np.sum(prediction * ground_truth)
    return (2. * intersection) / (np.sum(prediction) + np.sum(ground_truth) + 1e-7)  # Add epsilon to avoid division by zero

def calculate_f1_score(prediction, ground_truth):
    """
    Compute F1 score.
    Args:
        prediction (np.array): Binary prediction array.
        ground_truth (np.array): Binary ground truth array.
    Returns:
        float: F1 score.
    """
    prediction = prediction > 0
    ground_truth = ground_truth > 0
    tp = np.sum(prediction * ground_truth)
    fp = np.sum(prediction * (1 - ground_truth))
    fn = np.sum((1 - prediction) * ground_truth)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    return 2 * (precision * recall) / (precision + recall + 1e-7)

def calculate_dice(predicted, ground_truth):
    # Konvertáljuk a listákat NumPy tömbökké
    predicted = np.array(predicted)
    ground_truth = np.array(ground_truth)
    
    # Dice metrika számítása
    intersection = np.sum(predicted * ground_truth)
    sum_union = np.sum(predicted) + np.sum(ground_truth)
    return 2 * intersection / sum_union if sum_union > 0 else 1.0


def calculate_f1(predicted, ground_truth):
    # Konvertáljuk a listákat NumPy tömbökké
    predicted = np.array(predicted)
    ground_truth = np.array(ground_truth)
    
    # F1 metrika számítása
    tp = np.sum((predicted == 1) & (ground_truth == 1))
    fp = np.sum((predicted == 1) & (ground_truth == 0))
    fn = np.sum((predicted == 0) & (ground_truth == 1))
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0




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

            out = models[model_name](img_tensor)  # Model output: [B, num_classes, H, W]
            if isinstance(out, list):
                out = out[0]  # Take the primary tensor output

            # Apply softmax to get probabilities
            out = F.softmax(out, dim=1)

            # Resize the output to match the expected size (112, 112)
            out = F.interpolate(out, size=(n1, n1), mode='bilinear', align_corners=False)

            # Assign the result directly
            output[n, :, n11:n12, n11:n12] = out.cpu().numpy()

    # Perform argmax over the class dimension (axis 1)
    output = np.argmax(output, 1)
    return np.uint8(255 * output), output

def majority_voting(model_outputs):
    stacked_outputs = np.stack(model_outputs, axis=-1)  # Stack outputs along a new dimension
    majority = np.mean(stacked_outputs, axis=-1) > 0.5  # Majority vote
    return majority.astype(np.uint8) * 255  # Scale to 0-255

def process_uploaded_tiffs(uploaded_files, model_names, ground_truth_files=None):
    try:
        print(f"Uploaded files: {[file.name for file in uploaded_files]}")
        slices = [cv2.imread(file.name, cv2.IMREAD_GRAYSCALE) for file in uploaded_files]
        if not slices or any(slice_img is None for slice_img in slices):
            print("Error: One or more files could not be read as TIFF.")
            raise ValueError("One or more uploaded files could not be read as valid TIFF images.")

        print(f"Number of slices loaded: {len(slices)}")

        # Process ground truth files if provided
        ground_truth = None
        if ground_truth_files:
            ground_truth = [cv2.imread(file.name, cv2.IMREAD_GRAYSCALE) for file in ground_truth_files]
            if len(ground_truth) != len(slices):
                raise ValueError("Ground truth file count does not match the number of slices.")
            print(f"Number of ground truth slices loaded: {len(ground_truth)}")

        model_outputs, normalized_outputs = zip(*[model_inference(slices, model_name) for model_name in model_names])
        normalized_outputs = list(normalized_outputs)
        model_outputs = list(model_outputs)

        combined_outputs = [majority_voting([normalized_outputs[m][i] for m in range(len(model_names))]) for i in range(len(slices))]

        print("Processing complete.")
        return slices, model_outputs, combined_outputs, ground_truth

    except Exception as e:
        print(f"Error during TIFF processing: {e}")
        return None, None, None, None
    
def viewer(slice_idx, input_slices, model_outputs, combined_outputs, ground_truth):
    max_idx = len(input_slices) - 1
    slice_idx = max(0, min(slice_idx, max_idx))  # Clamp index within range

    input_slice = input_slices[slice_idx]
    model_slices = [model_output[slice_idx] for model_output in model_outputs]
    combined_slice = combined_outputs[slice_idx]

    # Metrikák számítása (ha van ground truth)
    dice_scores = []
    f1_scores = []
    ensemble_dice = ensemble_f1 = None
    if ground_truth:
        for model_output in model_slices:
            dice_scores.append(calculate_dice_score(model_output, ground_truth[slice_idx]))
            f1_scores.append(calculate_f1_score(model_output, ground_truth[slice_idx]))
        # Ensemble metrikák
        ensemble_dice = calculate_dice_score(combined_slice, ground_truth[slice_idx])
        ensemble_f1 = calculate_f1_score(combined_slice, ground_truth[slice_idx])

    # Megjelenített képek listája
    result = [input_slice] + model_slices + [combined_slice]

    # Ha van ground truth, adjuk hozzá
    if ground_truth and len(ground_truth) > slice_idx:
        result.append(ground_truth[slice_idx])
    else:
        result.append(None)  # No ground truth available

    # Kiegészítés metrikák megjelenítéséhez
    metric_info = {
        "dice_scores": dice_scores,
        "f1_scores": f1_scores,
        "ensemble_dice": ensemble_dice,
        "ensemble_f1": ensemble_f1,
    }
    return result, metric_info


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
            ground_truth_uploader = gr.File(
                file_types=[".tiff"], 
                file_count="multiple", 
                label="Upload Ground Truth (Optional)",
                elem_id="ground_truth_uploader"  # Id for custom styling
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
            ensemble_image = gr.Image(label="Ensembled Output")
            ground_truth_image = gr.Image(label="Ground Truth (if provided)")

        with gr.Row():
            columns = []
            for model_name in model_names:
                with gr.Column():
                    gr.Text(label=f"{model_name} Dice")    # Dice metrika
                    gr.Text(label=f"{model_name} F1")      # F1 metrika
                    gr.Text(label=f"{model_name} Dice Total")  # Összesített Dice
                    gr.Text(label=f"{model_name} F1 Total")    # Összesített F1
                    columns.append(gr.Column())



        # Define state variables
        input_slices_state = gr.State(value=None)
        model_outputs_state = gr.State(value=None)
        combined_outputs_state = gr.State(value=None)
        ground_truth_state = gr.State(value=None)

        # Process function
        def process(uploaded_files, ground_truth_files):
            if not uploaded_files:
                print("No files uploaded.")
                return None, None, None, None, gr.update(value=0, minimum=0, maximum=0)

            input_slices, model_outputs, combined_outputs, ground_truth = process_uploaded_tiffs(
                uploaded_files, model_names, ground_truth_files
            )

            if input_slices is None or model_outputs is None or combined_outputs is None:
                print("TIFF processing failed. Returning default values.")
                return None, None, None, None, gr.update(value=0, minimum=0, maximum=0)

            slice_count = len(input_slices)
            return (
                input_slices,
                model_outputs,
                combined_outputs,
                ground_truth,
                gr.update(value=0, minimum=0, maximum=slice_count - 1)
            )

        def display_metrics(model_names, metrics_per_model, ensemble_metrics):
            with gr.Row():
                for idx, model_name in enumerate(model_names):
                    with gr.Column():
                        gr.Text(value=f"{model_name} Metrics", bold=True)
                        gr.Text(value=f"Slice Dice: {metrics_per_model[idx][0]}")
                        gr.Text(value=f"Slice F1: {metrics_per_model[idx][1]}")
                        gr.Text(value=f"Total Dice: {metrics_per_model[idx][2]}")
                        gr.Text(value=f"Total F1: {metrics_per_model[idx][3]}")
                with gr.Column():
                    gr.Text(value="Ensemble Metrics", bold=True)
                    gr.Text(value=f"Slice Dice: {ensemble_metrics[0]}")
                    gr.Text(value=f"Slice F1: {ensemble_metrics[1]}")
                    gr.Text(value=f"Total Dice: {ensemble_metrics[2]}")
                    gr.Text(value=f"Total F1: {ensemble_metrics[3]}")


        def update(slice_idx, input_slices, model_outputs, combined_outputs, ground_truth):
            if not input_slices or not model_outputs or not combined_outputs:
                raise ValueError("No valid data available for display.")

            # Get the slice data
            input_slice = input_slices[slice_idx]
            ground_truth_slice = ground_truth[slice_idx] if ground_truth else None

            # Compute metrics
            metrics_per_model = [
                (
                    calculate_dice(model_output[slice_idx], ground_truth_slice),
                    calculate_f1(model_output[slice_idx], ground_truth_slice),
                    calculate_dice(model_output, ground_truth),
                    calculate_f1(model_output, ground_truth)
                )
                for model_output in model_outputs
            ]

            ensemble_metrics = (
                calculate_dice(combined_outputs[slice_idx], ground_truth_slice),
                calculate_f1(combined_outputs[slice_idx], ground_truth_slice),
                calculate_dice(combined_outputs, ground_truth),
                calculate_f1(combined_outputs, ground_truth)
            )

            # Prepare image outputs
            output_images = [input_slice] + [model_output[slice_idx] for model_output in model_outputs] + [combined_outputs[slice_idx], ground_truth_slice]

            return output_images, metrics_per_model, ensemble_metrics




        # Connect functionality
        process_button.click(
            process,
            inputs=[file_uploader, ground_truth_uploader],
            outputs=[
                input_slices_state,
                model_outputs_state,
                combined_outputs_state,
                ground_truth_state,
                slice_slider
            ],
        )

        slice_slider.change(
            update,
            inputs=[slice_slider, input_slices_state, model_outputs_state, combined_outputs_state, ground_truth_state],
            outputs=[
                input_image,
                *model_outputs_images,
                ensemble_image,
                ground_truth_image,
                display_metrics(model_names, metrics_per_model, ensemble_metrics),
            ]
        )





        # Custom HTML with CSS to make the file uploader scrollable with a max height
        gr.HTML("""
            <style>
                #file_uploader, #ground_truth_uploader {
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

