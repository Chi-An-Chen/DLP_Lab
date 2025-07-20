# Lab3 MaskGIT for Image Inpainting

## Run the code
Train:  
```
python training_transformer.py
```
Inference:  
```
python inpainting.py
```
(Make sure to edit the path for the dataset or checkpoint path etc.)

## Calculate FID Score
```
 cd faster-pytorch-fid
 python fid_score_gpu.py --predicted-path /path/your_inpainting_results_folder --device cuda:0
```

## Results
FID score after training 70 epochs  
```
FID : 29.1001
```
FID score after training 200 epochs  
```
FID : 28.7192
```  
