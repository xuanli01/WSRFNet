# WSRFNet: Wavelet-Based Scale-Specific Recurrent Feedback Network for Diabetic Retinopathy Lesion Segmentation

PaperID-756 for IJCAI 2024

## Code Environment
```bash
conda create -n WSRFNet python=3.8
conda activate WSRFNet
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

## Data Preparation
### Method 1
You can refer to the work  ['Automated lesion segmentation in fundus images with many-to-many reassembly of features'](https://github.com/CVIU-CSU/M2MRF-Lesion-Segmentation) [Pattern Recognition 2023] for the IDRID and DDR datasets. 

The download of the datasets can be found [here](https://github.com/CVIU-CSU/M2MRF-Lesion-Segmentation#results-and-models).

 The preparation of the datasets can be found [here](https://github.com/CVIU-CSU/M2MRF-Lesion-Segmentation#training-and-testing). 

 Then, reorganize the file structure of the test sets as follows and put the 'test_set' in the root directory of the code file.

 ```bash
├── test_set
│   ├── DDR
│   │   ├── annotations
│   │   └── image
│   └── IDRID
│       ├── annotations
│       └── image
 ```

### Method 2

You can directly use our processed test sets by unpacking the compressed files in the folder 'test_dataset' into 'test_set'.
 
```base
zip -s 0 test_dataset/test_set.zip --out test_set.zip
unzip test_set.zip
```

## Inference
We use the same evaluation code as the ['Automated lesion segmentation in fundus images with many-to-many reassembly of features'](https://github.com/CVIU-CSU/M2MRF-Lesion-Segmentation) to evaluate the predicted segmentation maps.

We ensemble the aboved evaluation code into our code.

### IDRID

First, unzip the compressed files 'idrid_model.zip' into the checkpoint file 'idrid_model.pth'.

```bash
unzip idrid_model.zip
```
Then, test the model and save the predicted segmentation maps via '--vis_results'.

```bash
python test.py -d IDRID -p test_set/IDRID -c idrid_model.pth --vis_results
```

### DDR

First, unzip the compressed files 'ddr_model.zip' into the checkpoint file 'ddr_model.pth'.

```bash
unzip ddr_model.zip
```
Then, test the model and save the predicted segmentation maps via '--vis_results'.

```bash
python test.py -d DDR -p test_set/DDR -c ddr_model.pth --vis_results
```

## Predicted Segmentation Maps
You can directly obtain the predicted segmentation maps in the folder 'seg_maps' without running the code.

Or you can obtain the the predicted segmentation maps of IDRID and DDR test sets in the folder 'IDRID_vis' and 'DDR_vis' via the above-mentioned code, respectively. 

### Notice

The color saturation of the ground truth (GT) maps is a little different from that of the predicted segmentation maps. Specifically, the color of the predicted segmentation maps is brighter, which is for better viewing. The color of the GT map is set  according to the library of MMCV. The corresponding relationship between color and lesion is as follows.

|  Lesion   | GT (RGB)  | Prediction (RGB) | Color|
|  :----:  | :----:  | :----: | :----: |
| EX  | [128, 0, 0] | [255, 0, 0] | Red |
| HE  | [0, 128, 0] | [0, 255, 0] | Green|
| SE  |[128, 128, 0] | [255, 255, 0] | Yellow|
| MA  |[0, 0, 128] |  [0, 128, 255] | Blue|
