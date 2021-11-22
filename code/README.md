# Code repository for Posterior Sampling and Uncertainty in Accelerated MRI using Stein Variational Gradient Descent
## Acknowledgement
Much of this code included in this repository is an extension of public NYU/facebook research fastMRI code, with MIT licenced. You can find the original code here: https://github.com/facebookresearch/fastMRI

## Dataset
NYU fastMRI dataset can be downloaded here: https://fastmri.med.nyu.edu/

For computational reasons we only train on T2 weighted images of the official train subset. For evaluation we use a split of the official validation set to validation/test.

## SVGD sampling with pretrained weights  
1. Set up environment using *requirements.txt*
2. Download dataset (see above)
3. Download pretrained weights for E2EVarNet trained with exp: [https://drive.google.com/file/d/1zTj0h0a1UYbWiyCZR33VlRGdCRyMHGcl/view]
4. Run command:
```
python run_svgd.py --test_path '/path_to_data/' --out_path './' --varnet_path './epoch=43-step=271039.ckpt' --ae_path './ae_50.pth'
```

## Training E2E VarNet with exponential weighted loss  
1. Set up environment using *requirements.txt*
2. Download dataset (see above)
4. Run command:
```
python train_varnet_demo.py --challenge multicoil --data_path ./fastMRI_T2 --mask_type equispaced_fraction --exp_loss 1 --drop_prob 0
```

## Training Auto-encoder for feature RBF kernel
1. Set up environment using *requirements.txt*
2. Download dataset (see above)
4. Run command:
```
python train_ae.py --train_datapath ./fastMRI_T2/train --val_datapath ./fastMRI_T2/validation --save_dir ./
```

## Issues
If you have any issues or questions, feel free to email XX or raise an issue.
