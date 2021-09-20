# Experiment_code
Code for training models for Optical Music Recognition (OMR) and making predictions

# Corpus Structure Requirement
In order for the code to work out of the box, please have the structure of the corpus being used as following
```
Dataset 
└───images
└───labels_note
└───labels_length
│   train.txt
│   valid.txt
│   test.txt
```

# Instructions
There are three different models provided as described in the paper -- Baseline, FlagDecoder, and RNNDecoder

## Baseline
Training
```
python train.py -voc_p <path to pitch vocabulary> -voc_r <path to rhythm vocabulary> -corpus <path to corpus>
```
Inference
```
python predict.py -voc_p <path to pitch vocabulary> -voc_r <path to rhythm vocabulary> -model <path to trained model> -image <path to directory of images> -out <directory to output predictions to>
```

## FlagDecoder
Training
```
python train_flag_accidental.py -voc_s <path to symbol vocabulary> -voc_d <path to duration vocabulary> -voc_a <path to alter vocabulary> -corpus <path to corpus>
```
Inference
```
python predict_flag.py -voc_d <path to duration vocabulary> -voc_s <path to symbol vocabulary> -voc_a <path to alter vocabulary> -model <path to trained model> -images <path to image directory to predict> -out_p <output path for pitch predictions> -out_r <output path for rhythm predictions>
```

## RNNDecoder
Training
```
python train_multi.py -voc_p <path to pitch vocabulary> -voc_r <path to rhythm vocabulary> -corpus <path to corpus>
```
Inference
```
python predict_multi.py -voc_p <path to pitch vocab> -voc_r <path to rhythm vocab> -model <path to trained model> -images <path to directory of images to predict> -out <directory to output predictions>
```

## Pretrained Models
Please download pretrained models from https://drive.google.com/drive/folders/1OVPg_oSsb1X9YaXI5mB7nxhO1WQx53u9?usp=sharing
