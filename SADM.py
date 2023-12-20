import torch
import torch.nn as nn
import torch.nn.functional as F

import attention_blocks as ab

import math

affine_par = True


class FeatureL2Norm(torch.nn.Module):
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature,norm)
    
###############################################################################
"VGG mudule"
class VGG_feas(nn.Module):

    def __init__(self, features):
        super(VGG_feas, self).__init__()
        self.features = features
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():            
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class Convblock(nn.Module):
    def __init__(self, in_channels, out_channels, padding_, dilation_, batch_norm=False):
        super(Convblock,self).__init__()
        self.bnflag = batch_norm
        self.convb = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding_, dilation=dilation_)
        if self.bnflag:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.convb(x)
        if self.bnflag:
            x = self.bn(x)
        x = self.relu(x)
        return x

def make_layers(cfg, in_channels = 3, batch_norm=False):
    layers = []
    
    for v in cfg:
        if v == 'M2':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        elif v == 'M1':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        elif v == 'D512':
            cb = Convblock(in_channels, 512, padding_=2, dilation_ = 2)
            layers += [cb]
            in_channels = 512
        else:
            cb = Convblock(in_channels, v, padding_=1, dilation_ = 1)
            layers += [cb]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'vgg16-3': [64, 64, 'M2', 128, 128, 'M2', 256, 256, 256, 'M2'],
    'vgg16-4': [ 512, 512, 512, 'M1'],
    'vgg16-5': ['D512', 'D512', 'D512', 'M1'],
}

def vgg16_feas(block, cfg_flag, in_channels = 3, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG_feas(block(cfg_flag,in_channels), **kwargs)
    return model

###############################################################################
"the correlation layer"


class Correlation_Module(nn.Module):
    
    def __init__(self):
        super(Correlation_Module, self).__init__()

    
    def forward(self, x):
        
        [bs, c, h, w] = x.size()
        pixel_num = h * w
        
        x1 = torch.div(x.view(bs, c, pixel_num),c)
        x2 = x.view(bs, c, pixel_num).permute(0,2,1)
        
        x1_x2 = torch.bmm(x2,x1)
        x1_x2 = x1_x2.view(bs,pixel_num,h,w)
        
        return x1_x2
        
class Poolopt_on_Corrmat(nn.Module):
    
    def __init__(self, select_indices):
        super(Poolopt_on_Corrmat, self).__init__()
        self.select_indices = select_indices
        
    def forward(self, corr):
        sort_corr,sort_corr_idx = torch.sort(corr,1,descending=True)
        sort_indices = torch.tensor(self.select_indices,dtype=torch.long).cuda()
        sort_corr_pool = torch.index_select(sort_corr,1,sort_indices)
        return sort_corr_pool
    
        
# classify module
class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes,affine = affine_par)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


aspp_base_dim = 96
class Classify_Module(nn.Module):

    def __init__(self,rates,inputscale, NoLabels):
        super(Classify_Module, self).__init__()
        
        
        self.aspp1 = ASPP_module(inputscale, aspp_base_dim//2, rate=rates[0])
        self.aspp2 = ASPP_module(inputscale, aspp_base_dim//2, rate=rates[1])
        self.aspp3 = ASPP_module(inputscale, aspp_base_dim//2, rate=rates[2])
        self.aspp4 = ASPP_module(inputscale, aspp_base_dim//2, rate=rates[3])
        
        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inputscale, aspp_base_dim//2, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(aspp_base_dim//2),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(5 * aspp_base_dim//2, aspp_base_dim, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(aspp_base_dim)

        self.conv2 = nn.Conv2d(aspp_base_dim, aspp_base_dim//2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(aspp_base_dim//2)
        
        self.last_conv = nn.Sequential(nn.Conv2d(aspp_base_dim//2, aspp_base_dim//2, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(aspp_base_dim//2),
                                       nn.ReLU(),
                                       nn.Conv2d(aspp_base_dim//2, aspp_base_dim//2, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(aspp_base_dim//2),
                                       nn.ReLU(),
                                       nn.Conv2d(aspp_base_dim//2, NoLabels, kernel_size=1, stride=1))

                


    def forward(self, x, im_size):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.upsample(x, size=(int(math.ceil(im_size/4)),
                                int(math.ceil(im_size/4))), mode='bilinear', align_corners=True)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = F.upsample(x, size=(int(math.ceil(im_size/2)),
                                int(math.ceil(im_size/2))), mode='bilinear', align_corners=True)
        
        x = self.last_conv(x)
        x = F.upsample(x, size=(int(im_size),
                                int(im_size)), mode='bilinear', align_corners=True)

        return x
    
    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()




def corr_fun(x, Corr, poolopt_on_corrmat):
    corr = Corr(x)    
    corr = poolopt_on_corrmat(corr)    
    return corr



class SelfDM_VGG_Module(nn.Module):
    '''
    The SelfDM VGG class module
    init parameters:
        block: the vgg block function
        other parameters refer to the SelfDM_VGG function
    '''
    def __init__(self, block, NoLabels, sort_num = 48, normalize_features=True,normalize_matches=True):
        super(SelfDM_VGG_Module,self).__init__()
        self.Scale_3 = vgg16_feas(block, cfg['vgg16-3'], 3)
        self.Scale_4 = vgg16_feas(block, cfg['vgg16-4'], 256)
        self.Scale_5 = vgg16_feas(block, cfg['vgg16-5'], 512)        
        
        '''
        The module used to measure the correspondence of two tensors
        '''        
        self.Corr = Correlation_Module()
        
        '''
        The corr maps pooling modules
        '''
        self.sort_num = sort_num
        sort_indices = []
        for s_i in range(0,self.sort_num):
            sort_indices.append(s_i)
        self.poolopt_on_corrmat = Poolopt_on_Corrmat(sort_indices)
        
        self.FeatureL2Norm = FeatureL2Norm()
        self.normalize_features = normalize_features
        self.normalize_matches = normalize_matches
        self.ReLU = nn.ReLU(inplace=True)
        
        self.spatial_attention = 1
        if self.spatial_attention != 0:
            self.satten_3 = ab.Self_Attn(256)
            self.satten_4 = ab.Self_Attn(512)
            self.satten_5 = ab.Self_Attn(512)

        '''
        SelfDM function module
        '''
        self.classifier = self._make_pred_layer(Classify_Module, [1,6,12,18], self.sort_num*3, NoLabels)
    

    def forward(self,x):
        
        [bs, c, h, w] = x.size()
        
        x_3 = self.Scale_3(x)
        # normalize
        if self.normalize_features:
            x_3_ = self.FeatureL2Norm(x_3)
        else:
            x_3_ = x_3
        if self.spatial_attention != 0:
            x_3_,_ = self.satten_3(x_3_)
               
        c_3 = corr_fun(x_3_, self.Corr, self.poolopt_on_corrmat)
        
        x_4 = self.Scale_4(x_3)
        # normalize
        if self.normalize_features:
            x_4_ = self.FeatureL2Norm(x_4)
        else:
            x_4_ = x_4
        if self.spatial_attention != 0:
            x_4_,_ = self.satten_4(x_4_)
        
        c_4 = corr_fun(x_4_, self.Corr, self.poolopt_on_corrmat)
        
        x_5 = self.Scale_5(x_4)
        # normalize
        if self.normalize_features:
            x_5_ = self.FeatureL2Norm(x_5)
        else:
            x_5_ = x_5
        if self.spatial_attention != 0:
            x_5_,_ = self.satten_5(x_5_)
        
        c_5 = corr_fun(x_5_, self.Corr, self.poolopt_on_corrmat)
        
        # normalize
        if self.normalize_matches:
            c_3 = self.FeatureL2Norm(self.ReLU(c_3))            
            c_4 = self.FeatureL2Norm(self.ReLU(c_4))            
            c_5 = self.FeatureL2Norm(self.ReLU(c_5))
        
        c = torch.cat((c_3,c_4,c_5),1)
        
        x = self.classifier(c,h)
        
        return x
    
    def _make_pred_layer(self, block, rates, inputscale, NoLabels):
        return block(rates,inputscale,NoLabels)


def SelfDM_VGG(NoLabels, sort_num = 48, normalize_features=True,normalize_matches=True):
    '''The interface function of SelfDM. The user only needs to call this function for training or testing.
    INPUT:
        NoLabels: the number of output class. default 2
        gpu_idx: the gpu used
        dim: the input size of the image. default 256
    OUTPUT:
        SelfDM model
    '''
    model = SelfDM_VGG_Module(make_layers, NoLabels, sort_num, normalize_features,normalize_matches)
    return model
