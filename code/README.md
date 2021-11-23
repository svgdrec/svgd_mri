# Code repository for Posterior Sampling and Uncertainty in Accelerated MRI using Stein Variational Gradient Descent
## Acknowledgement
This code is heavily influenced and can be seen as an extension of the public NYU/facebook research fastMRI code, under MIT licence. You can find the original code here: https://github.com/facebookresearch/fastMRI

## Dataset
Download NYU fastMRI dataset here: https://fastmri.med.nyu.edu/

For computational reasons, we only train on T2 weighted images of the official train subset. For evaluation, we use a split of the official validation set to validation/test.

## SVGD sampling with pretrained weights  
1. Set up environment:
```
pip install -r requirements.txt
```
2. Download dataset from https://fastmri.med.nyu.edu/
3. Download pretrained weights for E2EVarNet trained with exp: [https://drive.google.com/file/d/1zTj0h0a1UYbWiyCZR33VlRGdCRyMHGcl/view]
4. Run sampling:
```
python run_svgd.py --test_path '/path_to_data/' --out_path './' --varnet_path './epoch=43-step=271039.ckpt' --ae_path './ae_50.pth'
```

## Training E2E VarNet with exponentially weighted loss  
1. Set up environment:
```
pip install -r requirements.txt
```
2. Download dataset from https://fastmri.med.nyu.edu/
4. Run E2E VarNet training:
```
python train_varnet.py --challenge multicoil --data_path ./fastMRI_T2 --mask_type equispaced_fraction --exp_loss 1 --drop_prob 0
```

## Training Auto-encoder for RBF kernel in feature space
1. Set up environment:
```
pip install -r requirements.txt
```
2. Download dataset from https://fastmri.med.nyu.edu/
4. Run Auto-encoder training:
```
python train_ae.py --train_datapath ./fastMRI_T2/train --val_datapath ./fastMRI_T2/validation --save_dir ./
```

## Issues
If you have any issues or questions, feel free to email XX or raise an issue.
