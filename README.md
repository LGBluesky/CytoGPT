# Project Documentation

## Project Overview
This project provides an image testing framework developed based on [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4), primarily through the `evaluate.py` script for testing a **single image**. This document details how to use `evaluate.py` and the required model weights and parameters for setup. For program dependencies, please refer to the [MiniGPT-4 GitHub repository](https://github.com/Vision-CAIR/MiniGPT-4).

## evaluate.py File Introduction
The `evaluate.py` script is the core component of this project, designed to test **a single image**. Before running the script, ensure all required model weights and parameters are correctly configured (see below). The script loads pretrained models, processes the input image, and outputs the test results.

**Usage**:
1. Ensure all dependencies are installed (refer to `requirements.txt` or the [MiniGPT-4 repository](https://github.com/Vision-CAIR/MiniGPT-4) for environment setup).
2. Download and configure the model weights and parameters listed below.
3. Run the command:
   ```bash
   python evaluate.py --config eval_configs/Cyto_minigpt4_eval1.yaml --image_path <your_image_path>



## Model Weights and Parameters
Running evaluate.py requires downloading and configuring the following six model weights and parameters. Below are the details:
### 1. Image Encoder Weights

Description: Pretrained weights for image feature extraction.
Download Link: [image_encoder_weight.pth](https://pan.baidu.com/s/1KLLJDdJh6oIzW9reu0KCVA)  password: cuda
Configuration Location:
File: minigpt4/models/MAE_vit.py
Function: create_mae_vit_b
Configuration: Specify the weight path via the url parameter in the create_mae_vit_b function.


### 2. Q-former Parameters

Description: Configuration file for Q-former, based on the BERT model.
Download Link: [bert-base-uncased](https://pan.baidu.com/s/1Oam6_aXWNFirITOoNIsU9w) password: cuda
Configuration Location:
File: minigpt4/models/minigpt4.py
Function: init_Qformer
Configuration: Specify the parameter path in BertConfig.from_pretrained, e.g.:PythonBertConfig.from_pretrained('.../bert-base-uncased/config.json')


### 3. Q-former Weights

Description: Pretrained weights for Q-former.
Download Link: [Q-former.pth](https://pan.baidu.com/s/1rhiv7YR1rpK-1dk88pWRkw) password: cuda
Configuration Location:
File: eval_configs/Cyto_minigpt4_eval1.yaml
Configuration: Set the q_former_model field to the weight path, e.g.:YAMLq_former_model: 'q_former weight path'


### 4. Transformer-enhanced Classifier Weights

Description: Weights for the Transformer-enhanced classifier used in classification tasks.
Download Link: [classifier.pth](https://pan.baidu.com/s/1-kcCQ_oRkKE9v60YWP5nKw) password: cuda
Configuration Location:
File: minigpt4/models/MAE_vit.py
Function: create_cls_head
Configuration: Specify the weight path via the url parameter in the create_cls_head function.


### 5. Linear Projector Weights

Description: Weights for the linear projector used in feature projection.
Download Link: [linear projector.pth](https://pan.baidu.com/s/1LURsG3nrAz6lXWOuj1aaWg) password: cuda
Configuration Location:
File: eval_configs/Cyto_minigpt4_eval1.yaml
Configuration: Set the ckpt field to the weight path, e.g.:YAMLckpt: 'linear_projector weight path'


### 6. Vicuna Weights

Description: Pretrained weights for the Vicuna language model.
Download Link: [vicuna-7B](https://huggingface.co/lmsys/vicuna-7b-v1.5)
Configuration Location:
File: minigpt4/configs/models/minigpt4_vicuna0.yaml
Configuration: Set the llama_model field to the weight path, e.g.:YAMLllama_model: 'vicuna weight path'


### Configuration Steps

Download Weights and Parameters: Obtain all files from the cloud storage links provided above.
Update Configuration Files:
Modify minigpt4/models/MAE_vit.py and minigpt4/models/minigpt4.py to set the correct url or parameter paths.
Edit eval_configs/Cyto_minigpt4_eval1.yaml and minigpt4/configs/models/minigpt4_vicuna0.yaml to ensure paths are accurate.

Verify Configuration: Confirm all paths point to valid files and that the files are not corrupted.
Run Testing: Execute evaluate.py to perform image testing.

## Notes

Ensure the downloaded weights are compatible with the project version.
Back up original configuration files before making changes to avoid errors.
If you encounter issues with paths or missing files, verify the cloud storage links or contact the project maintainers.

## Acknowledgement
This project is built upon the foundational work of MiniGPT-4 and leverages the Vicuna language model. We express our gratitude to the MiniGPT-4 and Vicuna teams for their contributions to the open-source community.
## License
This project follows the licensing terms outlined in the MiniGPT-4 repository. Please refer to the MiniGPT-4 GitHub page for detailed licensing information.
