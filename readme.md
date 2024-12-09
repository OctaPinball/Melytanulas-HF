# Model Ensemble for Medical Image Segmentation

## Team Information

- **Team Name:** `Segmentation  Faults`
- **Team Members:**
  - `Scholtz Bálint András`, `A8O5M2`
  - `Schmieder Nándor`, `E9CLSH`

## Project Description

This project aims to enhance the accuracy of deep learning models for semantic segmentation by leveraging model ensembles. The primary focus is on the 2018 Atria Segmentation Dataset, which involves segmenting cardiac MRI images.

### Purpose:

The purpose of the project is to explore how combining multiple models through ensemble methods can lead to better segmentation performance compared to using a single model. Ensemble methods are known for improving accuracy at the cost of increased computational resources, making them a popular choice in AI competitions.

### Main Features:

Model Training: Train multiple deep learning models on the 2018 Atria Segmentation Dataset to perform cardiac MRI segmentation.
Ensemble Construction: Combine the outputs of these models using various ensemble techniques to improve segmentation accuracy.
Performance Analysis: Evaluate the benefits and drawbacks of using ensembles, focusing on accuracy improvements and computational costs.
### Problem:

Cardiac MRI segmentation is a challenging task that requires high precision for clinical use. This project addresses the need to improve segmentation performance by constructing and analyzing ensembles, which can yield more reliable and accurate results than single-model solutions.

## Repository Structure and Functions

| File/Folder                   | Description                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| `Dockerfile`                   | Defines the Docker image with necessary dependencies, including PyTorch and CUDA support. |
| `requirements.txt`             | Lists the Python packages and dependencies needed to run the project.       |
| `data/data-download.txt`       | Contains the URL of the source from which the dataset was manually downloaded. |
| `data/preprocess_data.py`      | Python script that preprocesses the raw MRI and LA cavity label data, and creates folders of the test data and an HDF5 file from the training data. |
| `data/preprocessed_data/`      | Placeholder folder. The actual preprocessed test data is not included due to size limitations, but can be reproduced by following the instructions in the "How to Run" section. |
| `data/TrainingSet.h5`          | Placeholder for the HDF5 file containing the preprocessed MRI scans and LA cavity labels for the training set (100 MRIs). This file is not included due to size limitations, but can be reproduced. |


## Data Preparation

In this project, the `preprocess_data.py` script is responsible for preparing the data by converting the MRI scans and corresponding LA cavity labels into a format suitable for training. This includes generating the `TrainingSet.h5` file and saving the test data into the `preprocessed_data/` folder.

- **Training and Validation Split:**  
  During training, 20% of the training data will be randomly selected and set aside as validation data. This ensures the model's performance can be evaluated on unseen data throughout the training process. Random seeding will be used to ensure the split is reproducible across different runs.

- **Test Data:**  
  The test set consists of 54 MRI scans, each saved as a set of `.tiff` files within the `preprocessed_data/` folder. These folders contain both the MRI slices and their corresponding cavity mask labels.

The final output of the preprocessing step includes:
- `TrainingSet.h5`: Contains the full training dataset, saved in the `preprocessed_data/` folder.
- 54 folders in `preprocessed_data/` with `.tiff` files of the MRI slices and cavity masks for the test set.



## Related Works

### Papers:
- **[DivergentNets: Medical Image Segmentation by Network Ensemble](https://arxiv.org/abs/2107.00283):**  
  This paper introduces a network ensemble approach to improve medical image segmentation accuracy, similar to the ensemble method we are using in our project for atrial segmentation.
  
- **[SenFormer: A Unified Approach for Liver and Brain Tumor Segmentation](https://arxiv.org/abs/2111.13280):**  
  SenFormer is a transformer-based architecture for medical image segmentation. It explores novel segmentation architectures, and we drew inspiration from its ensemble approach for improving segmentation performance.

- **[Combining Convolutional Neural Networks for Atrial Segmentation](https://dl.acm.org/doi/abs/10.1145/3555776.3577682):**  
  This paper discusses the combination of multiple CNN models for improving segmentation accuracy in atrial datasets. While no official code is available, its concepts guided our design choices for model ensemble construction.

- **[U-Net Approaches for Medical Image Segmentation](https://arxiv.org/pdf/2011.01118):**  
  This paper describes various U-Net architectures used for medical image segmentation. We will be heavily relying on the techniques outlined in this paper for our segmentation tasks, particularly for constructing our individual models in the ensemble.

### GitHub Repositories:
- **[Divergent Nets GitHub Repository](https://github.com/vlbthambawita/divergent-nets):**  
  This repository provides code for implementing network ensembles for medical image segmentation. It is a good ensemble for medical segmentation tasks similar to ours, making it highly relevant to our project’s approach for atrial segmentation.

- **[SenFormer GitHub Repository](https://github.com/WalBouss/SenFormer):**  
  This repository includes code for a transformer-based medical image segmentation approach. We explored this model architecture and its ensemble techniques when constructing our solution for semantic segmentation of cardiac MRI.



## How to Run the Project

1. **Build the Docker image:**  
   From the project root directory, run the following command to build the Docker image:
   ```bash
   docker build -t your-image-name .
# Step 2: Download the dataset
Manually download the dataset using the URL provided in data/data-download.txt. After downloading, extract the contents of the zip file and place the Training Set and Testing Set folders into the data/ directory of the project. Your folder structure should look like this:

```
/data
├── Training Set/
└── Testing Set/ 
```

# Step 3: Run the Docker container
Run the container with the current directory mounted, so the files are accessible inside the container:
```bash
docker run -it --name your-container-name -v ${PWD}:/workspace your-image-name
```
Run this if you want to use CUDA:
```bash
docker run --gpus all -it --name your-container-name -v ${PWD}:/workspace your-image-name
```

# Step 4: Run the pipeline
Inside the container, navigate to the /work folder and run the pipeline script:
```bash
cd work/
python pipeline.py
```
This script will run the whole project including preprocessing, training and evaluating.
- The script will generate the `TrainingSet.h5` file and save it into the `preprocessed_data/` folder. Additionally, it will create 54 subfolders within `preprocessed_data/`, each containing `.tiff` files for the cavity masks and the corresponding MRI data slices from the test set.
- The script will train the models with the preprocessed data.
- The script will evaluate all models, by firstly generate and save the results for each MR image as `.tiff` files in the `auto segmentation` folder, which is in the folder of the corresponding MR image. After all generation is finished the script will calculate the F1 score for the model. F1 Score = 2 * (Precision * Recall) / (Precision + Recall)

### Arguments
There are numerous options to customize the run operation by using arguments after the `python pipeline.py` command. Do not use more than one flag from the same type. For example, you can't use `-tm` and `-ta` at the same time, because they are in the same flag type (training arguments).

Preprocess arguments
- `-pr` preprocess data (default).
- `-ps` skip data preprocessing. This flag can be dangerous, if you want to train or evaluate without preprocessed dataset exception will be thrown. Use only when you have already done the preprocessing!

Training arguments
- `-ta` train all models (default).
- `-tm` train selected models. After the flag, the name of the model must be given in the following format: `-tm selected_models_name`. You can select multiple models: `-tm model_name_1 model_name_2`.
- `-tsa` skip all training. No model will be trained.
- `-tsm` skip selected models. All models will be queued up for training, but the selected ones will get removed from the queue. After the flag, the name of the model must be given in the following format: `-tsm selected_models_name`. You can select multiple models: `-tsm model_name_1 model_name_2`.

Evaluation arguments
- `-ea` evaluate all models (default).
- `-em` evaluate selected models. After the flag, the name of the model must be given in the following format: `-em selected_models_name`. You can select multiple models: `-em model_name_1 model_name_2`.
- `-esa` skip all evaluation. No model will be evaluated.
- `-esm` skip selected models. All models will be queued up for evaluation, but the selected ones will get removed from the queue. After the flag, the name of the model must be given in the following format: `-esm selected_models_name`. You can select multiple models: `-esm model_name_1 model_name_2`.

Ensemble evaluation arguments
- `-eea` use all models to create end evaluate ensemble model (default).
- `-eem` use selected models to create end evaluate ensemble model. After the flag, the name of the model must be given in the following format: `-eem selected_models_name`. You can select multiple models: `-eem model_name_1 model_name_2`.
- `-eesa` skip ensemble evaluation.
- `-eesm` skip selected models. All models will be queued up to create and evaluate ensembled model, but the selected ones will get removed from the queue. After the flag, the name of the model must be given in the following format: `-eesm selected_models_name`. You can select multiple models: `-eesm model_name_1 model_name_2`.

# Step 5: Run UI
After the training you have the ability to use multiple models and the ensembled model on any data input in an interactive UI. Use the following:
```bash
cd work/
python ui.py
```
![image](https://github.com/user-attachments/assets/ca104320-2476-4236-bd5b-8c8c704de5dd)
By uploading tiff files in the Upload MRI slices section, and uploading the corresponding labels in the Upload Ground Truth section (optional), and by clicking on the Process button you have the ability the segmentate the uploaded files and compare the results to the ground truth. You can navigate through the slices using the slider.
![image](https://github.com/user-attachments/assets/da63bf99-d239-4b14-ba8c-88c2f2958581)
You can see the input slice, and all the output slices from the models

It's important to know that the default models, which the UI uses are in the `preprocessed_data/model` folder. You can add trained models to this folder, and in order to use them, you must give them the correct file names. These names can be found in the Model list section.

### Model list
Current supported model list. Use the names in the 'Model name' column as arguments.

| Model | Model name | Default path |
| --- | --- | --- |
| UnetR | `unetr` | `unetr.pth` |
| Unet | `basicunet` | `basicunet.pth` |
| Unet++ | `unetplusplus` | `unetplusplus.pth` |
| DYNUnet | `dynunet` | `dynunet.pth` |


