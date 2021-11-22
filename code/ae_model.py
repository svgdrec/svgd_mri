import torch
from torch import nn

class Upsample(nn.Module):
    '''
    A class used for non-learnable upsampling of the images.
    For details please see the documentation of `nn.functional.interpolate`
    Parameters
    ----------
    scale_factor : int
        The scale for upsampling, default = 2
    mode : str
        The mode of upsampling, default = 'nearest'
    '''
    def __init__(self, scale_factor=3, mode='nearest'):
        super(Upsample, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.upsampling = Upsample(scale_factor=2, mode='nearest')

        #320
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.AvgPool2d(2, 2)
        ) 
        # /2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.AvgPool2d(2, 2)
        ) #/4

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.AvgPool2d(2, 2)
        ) #/8

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.AvgPool2d(2, 2)
        ) #/16

        self.conv_t4 = nn.Sequential(
            self.upsampling,
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU()
        )
        # 40

        self.conv_t3 = nn.Sequential(
            self.upsampling,
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU()
        )
        # 80

        self.conv_t2 = nn.Sequential(
            self.upsampling,
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU()
        )
        # 160

        self.conv_t1 = nn.Sequential(
            self.upsampling,
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU()
        )
        # 320
        self.conv_t0 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1, stride=1),
        )

    def encode(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

    def decode(self, x):
        x = self.conv_t4(x)
        x = self.conv_t3(x)
        x = self.conv_t2(x)
        x = self.conv_t1(x)
        x = self.conv_t0(x)
        return x

    def center_crop(self, data):
        if data.shape[-2] < 320:
            data_p = torch.ones((data.shape[0],320,data.shape[-1]), dtype=data.dtype, device=data.device) 
            data_p[:,:data.shape[-2],:] = data
            data = data_p

        if data.shape[-1] < 320:
            data_p = torch.ones((data.shape[0],data.shape[-2],320), dtype=data.dtype, device=data.device)
            data_p[:,:,:data.shape[-1]] = data
            data = data_p

        w_from = (data.shape[-2] // 2) - 160
        w_to = w_from + 320

        h_from = (data.shape[-1]  // 2) - 160
        h_to = h_from + 320

        ret = data[:,w_from:w_to, h_from:h_to]

        return ret

    def forward(self, x):
        rss = self.center_crop(x[:,0]).unsqueeze(1)
        mean = torch.mean(rss, (1,2,3), keepdim=True)
        std = torch.std(rss, (1,2,3), keepdim=True)
        rss = (rss-mean)/std
        z = self.encode(x)
        x_ret = self.decode(z)
        return x_ret, z
    

