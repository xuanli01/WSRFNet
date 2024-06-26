# WSRFNet: Wavelet-Based Scale-Specific Recurrent Feedback Network for Diabetic Retinopathy Lesion Segmentation

## Code Environment
```bash
conda create -n WSRFNet python=3.8
conda activate WSRFNet
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

## Data Preparation
You can refer to the work  ['Automated lesion segmentation in fundus images with many-to-many reassembly of features'](https://github.com/CVIU-CSU/M2MRF-Lesion-Segmentation) [Pattern Recognition 2023] for the IDRID and DDR datasets. 

The download of the datasets can be found [here](https://github.com/CVIU-CSU/M2MRF-Lesion-Segmentation#results-and-models).

The preparation of the datasets can be found [here](https://github.com/CVIU-CSU/M2MRF-Lesion-Segmentation#training-and-testing). 

## Inference
We use the same evaluation code as the ['Automated lesion segmentation in fundus images with many-to-many reassembly of features'](https://github.com/CVIU-CSU/M2MRF-Lesion-Segmentation) to evaluate the predicted segmentation maps.

## Code
Our code is about to be open-sourced.
