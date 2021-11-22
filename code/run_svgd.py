import argparse
import torch
from ae_model import AE
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import fastmri
from fastmri.data import SliceDataset
import fastmri.data.transforms as T
from fastmri.pl_modules.varnet_module import VarNetModule
from fastmri.data.subsample import create_mask_for_mask_type

def create_particles(ksp, P, mask):
    """
    create_particles creates inital particles as Eq on row 393.

    :param ksp: measuremnts
    :param P: number of samples created
    :param mask: undersampling mask

    :return: returns inital samples
    """

    ksp = torch.view_as_complex(ksp)
    _,C,W,H = ksp.shape
    x_p = torch.zeros((P,C,W,H), dtype=ksp.dtype)

    for i in range(P):
        E_r = torch.empty(ksp.shape).normal_(mean=torch.mean(ksp.real),std=torch.std(ksp.real)*0.01)
        E_i = torch.empty(ksp.shape).normal_(mean=torch.mean(ksp.imag),std=torch.std(ksp.imag)*0.01)

        x_p[i] = ksp + (1-mask.squeeze(-1).int()) * (E_r + 1j * E_i)

    return torch.view_as_real(x_p)

def get_features(x, ae_model, crop_size):
    """
    get_features calculates auto-encoder features

    :param x: samples x
    :param ae_model: Auto-encoder model

    :return: returns x in feature space and x to calculate gradient
    """
    x = x.detach()
    x = fastmri.ifft2c(torch.view_as_real(x))
    x.requires_grad = True
    y_ = fastmri.rss(fastmri.complex_abs(x), dim=1)

    # Normalise
    y_ = T.center_crop(y_, crop_size).unsqueeze(1)
    mean = torch.mean(y_, (1,2,3), keepdim=True)
    std_y = torch.std(y_, (1,2,3), keepdim=True)
    y_ = (y_-mean)/std_y

    z = ae_model.encode(y_)

    return z, x

def rbf_ae_grad(y_z,x_z,x):
    """
    rbf_ae_grad calculates the rbf and gradient rbf in feature space

    :param y_z: latent samples of y
    :param x_z: latent samples of x
    :param x: gradient with respect to x

    :return: returns gradient from rbf and rbf value
    """
    yx = torch.sum((y_z-x_z).view(x.shape[0],-1).pow(2),1)
    med_z = torch.sqrt(torch.median(torch.abs(yx)))

    bw = med_z**2 / np.log(x.shape[0])
    exp = torch.exp(-(1/bw) * yx)
    torch.sum(exp).backward(retain_graph=True)

    grad = torch.view_as_complex(fastmri.fft2c(x.grad))

    return grad.detach().to('cpu'), exp.detach().to('cpu')

def svgd_grad_step(x, p_grad, ae_model, crop_size, weight):
    """
    svgd_grad_step calculates the gradient step of Eq. 8

    :param x: samples x
    :param p_grad: p_grad gradient from VarNet
    :param net: VarNet model
    :param ae_model: Auto-encoder model
    :param weight: Hyperparameter weight gamma

    :return: returns gradient from Eq. 8
    """

    x = torch.view_as_complex(x)
    p_grad = torch.view_as_complex(p_grad)

    z, x_in = get_features(x.to('cuda:0'), ae_model.to('cuda:0'), crop_size,)

    P,C,W,H = x.shape
    rbf_x = torch.zeros((P,P)).to(x.device)
    rbf_x_grad = torch.zeros((P,C,W,H), dtype=x.dtype).to(x.device)
    # Iterate all samples and calculate gradient and kernel
    for i in range(P):
        gr, ker = rbf_ae_grad(z[i],z,x_in)
        rbf_x_grad[i] = torch.mean(gr,0)
        rbf_x[i] = ker

    svgd_grad = torch.mean(rbf_x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * p_grad.unsqueeze(1),1)
    # Normalise
    svgd_grad = svgd_grad * torch.linalg.norm(p_grad,dim=(1,2,3)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) / torch.linalg.norm(svgd_grad,dim=(1,2,3)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    svgd_grad = svgd_grad + weight * rbf_x_grad

    # Normalise
    #svgd_grad = svgd_grad * torch.linalg.norm(p_grad,dim=(1,2,3)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) / torch.linalg.norm(svgd_grad,dim=(1,2,3)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    del x_in
    del z
    return torch.view_as_real(svgd_grad)

def forward_svgd(masked_kspace, mask, varnet_model, ae_model, crop_size, weight=5e-9):
    """
    forward_svgd runs the unrolled gradient reconstruction using SVGD

    :param masked_kspace: Initial masked samples
    :param mask: Undersampling mask
    :param net: VarNet model
    :param ae_model: Auto-encoder model
    :param weight: Hyperparameter weight gamma

    :return: returns reconstructed samples approximating p(k|y)
    """

    with torch.no_grad(): # Estimate coil sensitivity maps
        #varnet_model.sens_net.to('cuda:0')
        sens_maps = varnet_model.sens_net(masked_kspace,mask) #.to('cuda:0'), mask.to('cuda:0')).detach().to('cpu')
    kspace_pred = masked_kspace.clone()

    # Unrolled reconstruction
    for ix, cascade in enumerate(varnet_model.cascades):
        with torch.no_grad(): # Estimate gradient step
            p_grad = cascade(kspace_pred, masked_kspace, mask, sens_maps)

        kspace_pred = kspace_pred -  svgd_grad_step(kspace_pred, p_grad, ae_model, crop_size, weight) # SVGD gradient step Eq. 8

    return kspace_pred

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--test_path",
        type=str,
        help="Data path to test folder.",
    )

    parser.add_argument(
        "--out_path",
        type=str,
        help="Data path to output files.",
    )

    parser.add_argument(
        "--varnet_path",
        type=str,
        help="Path for pretrained varnet model.",
    )

    parser.add_argument(
        "--ae_path",
        type=str,
        help="Path for pretrained ae model.",
    )

    parser.add_argument(
        "--gamma",
        default=5e-9,
        type=float,
        help="Weight factor gamma.",
    )

    parser.add_argument(
        "--num_sampels",
        default=25,
        type=int,
        help="Number of samples for reconstruction.",
    )

    # Collect inputs
    args = parser.parse_args()
    data_path = args.test_path
    out_path = args.out_path
    varnet_dir_path = args.varnet_path
    ae_dir_path = args.ae_path
    weight = args.gamma
    num_particles = args.num_sampels

    # Create and load E2E VarNet and Auto-encoder models
    model = VarNetModule(num_cascades=12, pools=4, chans=16, sens_pools=4, sens_chans=8)
    state_dict = torch.load(varnet_dir_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict['state_dict'])
    model = model.varnet
    model = model.eval()

    ae_model = AE()
    ae_model.load_state_dict(torch.load(ae_dir_path,map_location=torch.device('cpu')))
    ae_model.eval()

    # Create dataloader using fastmri
    data_transform = T.VarNetDataTransform(mask_func=create_mask_for_mask_type("equispaced", [0.04], [8]))
    dataset = SliceDataset(root=data_path, transform=data_transform, challenge="multicoil")
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=4, shuffle=True)

    for batch in tqdm(dataloader, desc="Running inference"):
        init_particles = create_particles(batch.masked_kspace, num_particles, batch.mask) # Create XX numbers of y_0 from y
        # Run SVGD
        rec_p = forward_svgd(init_particles, batch.mask, model, ae_model, batch.crop_size, weight=weight)

        # Save samples
        rec_part = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(rec_p)), dim=1).cpu()
        target, rec_part = fastmri.data.transforms.center_crop_to_smallest(batch.target, rec_part)
        name = batch.fname[0] + '_' + str(batch.slice_num.item()) + '.npy'
        print('Sampled subject/slice num: ', name)
        with open(out_path + name, 'wb') as f:
            np.save(f, rec_part.numpy())
