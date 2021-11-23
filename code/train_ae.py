# Incl. code from facebook/fastMRI
# https://github.com/facebookresearch/fastMRI/tree/b3f01ed5d4a1f2c597c9b7c7a1b846d8de5db06c/fastmri

import os
import numpy as np
import argparse
import random
import h5py
import os
import fastmri
import torch
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset

from ae_model import AE

def rss(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS).
    RSS is computed assuming that dim is the coil dimension.
    Args:
        data: The input tensor
        dim: The dimensions along which to apply the RSS transform
    Returns:
        The RSS value.
    """
    return torch.sqrt((data ** 2).sum(dim))

def normalize(x):
    return (x-x.min())/(x.max()-x.min())

def nmse(pred, gt):
    return np.linalg.norm(pred.flatten() - gt.flatten()) ** 2 / np.linalg.norm(gt.flatten()) ** 2

def complex_to_chan_dim(x):
    out_array = torch.stack([torch.real(x), torch.imag(x)], axis=1)
    return out_array

def center_crop(data, shape) -> torch.Tensor:
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data: The input tensor to be center cropped. It should
            have at least 2 dimensions and the cropping is applied along the
            last two dimensions.
        shape: The output shape. The shape should be smaller
            than the corresponding dimensions of data.

    Returns:
        The center cropped image.
    """
    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]

def center_crop_to_smallest(
    x: torch.Tensor, y: torch.Tensor):
    """
    Apply a center crop on the larger image to the size of the smaller.

    The minimum is taken over dim=-1 and dim=-2. If x is smaller than y at
    dim=-1 and y is smaller than x at dim=-2, then the returned dimension will
    be a mixture of the two.

    Args:
        x: The first image.
        y: The second image.

    Returns:
        tuple of tensors x and y, each cropped to the minimim size.
    """
    smallest_width = min(x.shape[-1], y.shape[-1])
    smallest_height = min(x.shape[-2], y.shape[-2])
    x = center_crop(x, (smallest_height, smallest_width))
    y = center_crop(y, (smallest_height, smallest_width))

    return x, y

class MaskFunc:
    """
    An object for GRAPPA-style sampling masks.

    This crates a sampling mask that densely samples the center while
    subsampling outer k-space regions based on the undersampling factor.
    """

    def __init__(self, center_fractions, accelerations):
        """
        Args:
            center_fractions: Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is
                chosen uniformly each time.
            accelerations: Amount of under-sampling. This should have the same
                length as center_fractions. If multiple values are provided,
                then one of these is chosen uniformly each time.
        """
        if not len(center_fractions) == len(accelerations):
            raise ValueError(
                "Number of center fractions should match number of accelerations"
            )

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.rng = np.random.RandomState()  # pylint: disable=no-member

    def __call__(
        self, shape, seed = None
    ) -> torch.Tensor:
        raise NotImplementedError

    def choose_acceleration(self):
        """Choose acceleration based on class parameters."""
        choice = self.rng.randint(0, len(self.accelerations))
        center_fraction = self.center_fractions[choice]
        acceleration = self.accelerations[choice]

        return center_fraction, acceleration

def cartesian_mask(shape, acc, sample_n=10, centred=False):
    """
    Sampling density estimated from implementation of kt FOCUSS
    shape: tuple - of form (..., nx, ny)
    acc: float - doesn't have to be integer 4, 8, etc..
    """
    N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
    pdf_x = normal_pdf(Nx, 0.5/(Nx/10.)**2)
    lmda = Nx/(2.*acc)
    n_lines = int(Nx / acc)

    # add uniform distribution
    pdf_x += lmda * 1./Nx

    if sample_n:
        pdf_x[Nx//2-sample_n//2:Nx//2+sample_n//2] = 0
        pdf_x /= np.sum(pdf_x)
        n_lines -= sample_n

    mask = np.zeros((N, Nx))
    for i in range(N):
        idx = np.random.choice(Nx, n_lines, False, pdf_x)
        mask[i, idx] = 1

    if sample_n:
        mask[:, Nx//2-sample_n//2:Nx//2+sample_n//2] = 1

    size = mask.itemsize
    mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))

    mask = mask.reshape(shape)

    if not centred:
        mask = ifftshift(mask, axes=(-1, -2))

    return mask

def create_particles(ksp, P, mask):
    B,C,W,H = ksp.shape
    x_p = torch.zeros((B,P,W,H))
    ksp_p = torch.zeros((B,P,W,H), dtype=ksp.dtype)

    ksp_m = ksp * mask.unsqueeze(0).unsqueeze(0)

    for i in range(P):
        E_r = torch.empty(ksp.shape).normal_(mean=torch.mean(ksp_m.real),std=torch.std(ksp_m.real)*0.1) #* (1-mask.unsqueeze(0).unsqueeze(0))
        E_i = torch.empty(ksp.shape).normal_(mean=torch.mean(ksp_m.imag),std=torch.std(ksp_m.imag)*0.1) #* (1-mask.unsqueeze(0).unsqueeze(0))

        filled = E_r + 1j * E_i

        x_p[:,i] = torch.abs(torch.fft.ifft2(ksp_m[:,0] + filled[:,0])).float()
        ksp_p[:,i] = ksp_m[:,0]

    return x_p, ksp_p

class EquispacedMaskFunc(MaskFunc):
    """
    EquispacedMaskFunc creates a sub-sampling mask of a given shape.

    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding tovlow-frequencies.
        2. The other columns are selected with equal spacing at a proportion
           that reaches the desired acceleration rate taking into consideration
           the number of low frequencies. This ensures that the expected number
           of columns selected is equal to (N / acceleration)

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the EquispacedMaskFunc object is called.

    Note that this function may not give equispaced samples (documented in
    https://github.com/facebookresearch/fastMRI/issues/54), which will require
    modifications to standard GRAPPA approaches. Nonetheless, this aspect of
    the function has been preserved to match the public multicoil data.
    """

    def __init__(
        self,
        center_fractions,
        accelerations,
        skip_low_freqs = False,
        skip_around_low_freqs = False,
    ):
        """
        Args:
            center_fractions: Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is
                chosen uniformly each time.
            accelerations: Amount of under-sampling. This should have the same
                length as center_fractions. If multiple values are provided,
                then one of these is chosen uniformly each time.
            skip_low_freqs: Whether to skip already sampled low-frequency lines
                for the purposes of determining where equispaced lines should be.
                Set this `True` to guarantee the same number of sampled lines for
                all masks with a given (acceleration, center_fraction) setting.
            skip_around_low_freqs: Whether to also skip the two k-space lines right
                next to the already sampled low-frequency region. Used to guarantee
                that equispaced sampling doesn't extend the low-frequency region.
                This is mostly useful for VarNet, since it guarantees the same number
                of low-frequency lines are used for the sensitivity map calculation
                for all masks with a given (acceleration, center_fraction) setting.
                This argument has no effect when `skip_low_freqs` is `False`.
        """

        super().__init__(center_fractions, accelerations)
        self.skip_low_freqs = skip_low_freqs
        self.skip_around_low_freqs = skip_around_low_freqs

    def __call__(
        self, shape, seed = None
    ) -> torch.Tensor:
        """
        Args:
            shape: The shape of the mask to be created. The shape should have
                at least 3 dimensions. Samples are drawn along the second last
                dimension.
            seed: Seed for the random number generator. Setting the seed
                ensures the same mask is generated each time for the same
                shape. The random state is reset afterwards.

        Returns:
            A mask of the specified shape.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        center_fraction, acceleration = self.choose_acceleration()
        num_cols = shape[-2]
        num_low_freqs = int(round(num_cols * center_fraction))

        # create the mask
        mask = np.zeros(num_cols, dtype=np.float32)
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad : pad + num_low_freqs] = True

        # If everything has been sampled in the center: we don't need to sample anything else.
        if num_low_freqs * acceleration <= num_cols:
            if self.skip_low_freqs:
                buffer = 0
                if self.skip_around_low_freqs:
                    buffer = 2
                # Compute the adjusted acceleration according to having
                #  (num_low_freqs + buffer) center lines.
                adjusted_accel = (
                    acceleration * (num_low_freqs + buffer - num_cols)
                ) / (num_low_freqs * acceleration - num_cols)
                offset = self.rng.randint(0, round(adjusted_accel) - 1)

                # Select samples from the remaining columns
                accel_samples = np.arange(
                    offset, num_cols - num_low_freqs - buffer - 1, adjusted_accel
                )
                accel_samples = np.around(accel_samples).astype(np.uint)

                skip = (
                    num_low_freqs + buffer
                )  # Skip low freq AND optionally lines right next to it
                for sample in accel_samples:
                    if sample < pad - buffer // 2:
                        mask[sample] = True
                    else:  # sample is further than center, so skip low_freqs
                        mask[int(sample + skip)] = True
            else:  # Default behaviour
                # determine acceleration rate by adjusting for the number of low frequencies
                adjusted_accel = (acceleration * (num_low_freqs - num_cols)) / (
                    num_low_freqs * acceleration - num_cols
                )
                offset = self.rng.randint(0, round(adjusted_accel))

                accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
                accel_samples = np.around(accel_samples).astype(np.uint)
                mask[accel_samples] = True

            # reshape the mask
            mask_shape = [1 for _ in shape]
            mask_shape[-2] = num_cols
            mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

        return mask

class fastMRI_dataset(Dataset):
    def __init__(self, dirname, sequence='T2', batch_size=16):
        self.dirname = dirname
        self.batch_sz = batch_size
        allfiles = os.listdir(dirname)  # get all subjects
        allfiles = [s for s in allfiles if s[-3:]=='.h5'] # only keep '.h5' files

        if sequence: # only pick certain sequence
            self.datafiles = [s for s in allfiles if sequence in s]
        else: # else load all
            self.datafiles = allfiles
        #self.datafiles = self.datafiles[:2]

    def __len__(self):
        return len(self.datafiles)

    def sens_reduce(self, x, sens_maps):
        x = torch.fft.ifftshift(torch.fft.ifft2(x, norm='ortho'),dim=(-1,-2))
        sense_norm = torch.sum(sens_maps * torch.conj(sens_maps), axis=1)
        sense_norm[sense_norm == 0] = 1e-16
        reduced = torch.mul(x, torch.conj(sens_maps)).sum(dim=1)
        return complex_to_chan_dim(torch.div(reduced,sense_norm))

    def __getitem__(self, index):
        subj_file = self.datafiles[index]

        # Load data
        with h5py.File(self.dirname + subj_file, 'r') as fdset:
            #header = fdset['ismrmrd_header'][()].decode("utf-8") # Header file
            ksp = fdset['kspace'] # Raw fully sampled kspace data
            sliceindex = np.sort(np.random.choice(range(ksp.shape[0]), self.batch_sz, replace=False))
            ksp_sli = ksp[sliceindex] # Coils,H,W

        ksp_sli = torch.from_numpy(ksp_sli)

        rss_t = torch.sqrt(torch.sum(torch.abs(torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(ksp_sli, dim=(-1,-2)), norm='ortho'), dim=(-1,-2)))**2,1)).unsqueeze(1)
        mean = torch.mean(rss_t, (1,2,3), keepdim=True)
        std = torch.std(rss_t, (1,2,3), keepdim=True)
        rss_t = (rss_t-mean)/std

        mask_fc = EquispacedMaskFunc([0.08], [random.randint(1,10)])
        mask = mask_fc((1, ksp_sli.shape[-1], 1)).transpose(-1,-2)
        ksp_sli_masked = ksp_sli * mask

        rss_u = torch.sqrt(torch.sum(torch.abs(torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(ksp_sli_masked, dim=(-1,-2)), norm='ortho'), dim=(-1,-2)))**2,1))
        return rss_u.unsqueeze(1), rss_t.unsqueeze(1)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', type=int, nargs=1, default=100, help='number of epochs')
    parser.add_argument('--batch_sz', type=int, nargs=1, default=10, help='batch size')
    parser.add_argument('--lr', type=float, nargs=1, default=3e-4, help='initial learning rate')
    parser.add_argument('--R', type=int, nargs=1, default=8, help='Acceleration factor for k-space sampling')
    parser.add_argument('--train_datapath', type=str, default='./data/train/')
    parser.add_argument('--val_datapath', type=str, default='./data/validation/')
    parser.add_argument('--save_dir', type=str, default='./data/save/')

    args = parser.parse_args()
    name = '_CNNae_'
    Nx, Ny = 320, 320
    train_datapath = args.train_datapath
    val_datapath= args.val_datapath
    save_dir = args.save_dir
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Configure directory info
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Specify network
    net = AE()
    net.to(device)

    # Init Optimizer Adam
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)

    # Define loss
    loss_f = nn.MSELoss()

    # Create dataset
    train_dataset = fastMRI_dataset(train_datapath, sequence='T2', batch_size=args.batch_sz)
    train_dataloader  = data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)

    val_dataset = fastMRI_dataset(val_datapath, sequence='T2', batch_size=args.batch_sz)
    val_dataloader  = data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    print('Start Training...')
    for epoch in range(int(args.num_epoch)):
        # Training
        train_err = 0
        train_batches = 0
        net.train()
        sum_loss = 0
        for ix, batch in enumerate(train_dataloader):
            rss_u, target = batch
            rss_u, target = rss_u[0], target[0]
            optimizer.zero_grad()
            rec_rss, _ = net(rss_u.to(device))
            target, rec_rss = center_crop_to_smallest(target[:,0].to(device), rec_rss)
            loss = loss_f(rec_rss, target)
            sum_loss += loss.item()
            loss.backward()
            optimizer.step()

        print('====> epoch: {} Average loss: {:.4f}'.format(epoch, sum_loss))
        if epoch%5 == 0:
            # Validation
            net.eval()
            sum_loss = 0
            for ix, batch in enumerate(train_dataloader):
                rss_u, target = batch
                rss_u, target = rss_u[0], target[0]
                optimizer.zero_grad()
                # Forward pass
                rec_rss, _ = net(rss_u.float().to(device))
                target, rec_rss = center_crop_to_smallest(target[:,0].to(device), rec_rss)
                loss = loss_f(rec_rss, target)
                sum_loss += loss.item()
            print('====> VALIDATION epoch: {} Average Validation loss: {:.4f}'.format(epoch, sum_loss))

        # Save model
        if epoch % 10 == 0:
            path = save_dir + '/' + str(epoch) + '.pth'
            torch.save(net.state_dict(), path)
