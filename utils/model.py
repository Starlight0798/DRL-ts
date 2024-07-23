from torch import nn
import torch
import numpy as np
from torch.nn import functional as F
from loguru import logger

def initialize_weights(layer, init_type='orthogonal', nonlinearity='relu'):
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        if init_type == 'kaiming':                  # kaiming初始化，适合激活函数为ReLU, LeakyReLU, PReLU
            nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)
        elif init_type == 'xavier':
            nn.init.xavier_uniform_(layer.weight)   # xavier初始化, 适合激活函数为tanh和sigmoid
        elif init_type == 'orthogonal':
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))       # 正交初始化，适合激活函数为ReLU
        else:       
            raise ValueError(f"Unknown initialization type: {init_type}")
        
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
    return layer


# swish激活函数
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# 全连接层
class MLP(nn.Module):
    def __init__(self,
                 dim_list,
                 activation=nn.ReLU(inplace=True),
                 last_act=False,
                 use_norm=False,
                 linear=nn.Linear,
                 *args, **kwargs
                 ):
        super(MLP, self).__init__()
        assert dim_list, "Dim list can't be empty!"
        layers = []
        for i in range(len(dim_list) - 1):
            layer = initialize_weights(linear(dim_list[i], dim_list[i + 1], *args, **kwargs))
            layers.append(layer)
            if i < len(dim_list) - 2:
                if use_norm:
                    layers.append(nn.LayerNorm(dim_list[i + 1]))
                layers.append(activation)
        if last_act:
            if use_norm:
                layers.append(nn.LayerNorm(dim_list[-1]))
            layers.append(activation)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


# 微调层
class LoRALayer(nn.Module):
    def __init__(self, linear_layer, rank):
        super(LoRALayer, self).__init__()
        self.linear_layer = linear_layer
        self.rank = rank
        self.A = nn.Parameter(torch.randn(linear_layer.weight.size(0), rank))
        self.B = nn.Parameter(torch.randn(rank, linear_layer.weight.size(1)))

    def forward(self, x):
        W_adjusted = self.linear_layer.weight + torch.matmul(self.A, self.B)
        return nn.functional.linear(x, W_adjusted, self.linear_layer.bias)
    
    
# 稠密层(单层)
class DenseLayer(nn.Module):
    def __init__(self, in_features, growth_rate):
        super(DenseLayer, self).__init__()
        self.fc = MLP([in_features, growth_rate], last_act=True)

    def forward(self, x):
        return torch.cat([x, self.fc(x)], dim=-1)


# 稠密层
class DenseBlock(nn.Module):
    def __init__(self, in_features, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_features + i * growth_rate, growth_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    

# 残差连接层
class ResidualLayer(nn.Module):
    def __init__(self, layer_fn):
        super(ResidualLayer, self).__init__()
        self.layer = layer_fn

    def forward(self, x):
        identity = x
        out = self.layer(x)
        out += identity
        return out


# 残差连接块
class ResidualBlock(nn.Module):
    def __init__(self, in_features, num_layers):
        super(ResidualBlock, self).__init__()
        self.in_features = in_features
        layers = []
        for i in range(num_layers):
            layers.append(ResidualLayer(MLP([in_features, in_features])))
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
    
# 残差稠密层
class ResidualDenseLayer(nn.Module):
    def __init__(self, in_features, growth_rate, num_layers):
        super(ResidualDenseLayer, self).__init__()
        self.layer = ResidualLayer(
            nn.Sequential(
                DenseBlock(in_features, growth_rate, num_layers),
                MLP([in_features + growth_rate * num_layers, in_features])
            )
        )

    def forward(self, x):
        return self.layer(x)
    
    
# 残差稠密块
class ResidualDenseBlock(nn.Module):
    def __init__(self, in_features, growth_rate, num_layers, num_blocks):
        super(ResidualDenseBlock, self).__init__()
        self.layers = nn.Sequential(*[ResidualDenseLayer(in_features, growth_rate, num_layers) for _ in range(num_blocks)])

    def forward(self, x):
        return self.layers(x)
    

# 过渡层
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            MLP([channel, channel // reduction], bias=False),
            nn.ReLU(inplace=True),
            MLP([channel // reduction, channel], bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.size()
        y = self.avg_pool(x.unsqueeze(2)).view(b, c)  
        y = self.fc(y).view(b, c)  
        return x * y.expand_as(x)  



# 注意力层
class AttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(AttentionLayer, self).__init__()
        self.fc = MLP([in_features, out_features])
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        attention_weights = self.softmax(self.tanh(self.fc(x)))
        return torch.bmm(attention_weights.unsqueeze(2), x.unsqueeze(1)).squeeze(2)



# 带噪声的全连接层
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            self.reset_noise()
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)  # mul是对应元素相乘
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)

        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))  # 这里要除以out_features

    @staticmethod
    def scale_noise(size):
        x = torch.randn(size)  # torch.randn产生标准高斯分布
        x = x.sign().mul(x.abs().sqrt())
        return x

    def reset_noise(self):
        epsilon_i = self.scale_noise(self.in_features)
        epsilon_j = self.scale_noise(self.out_features)
        self.weight_epsilon.copy_(torch.ger(epsilon_j, epsilon_i))
        self.bias_epsilon.copy_(epsilon_j)
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features})"
    

# 一种兼顾宽度和深度的全连接层，提取信息效率更高
class PSCN(nn.Module):
    def __init__(self, input_dim, output_dim, linear=nn.Linear, use_norm=False):
        super(PSCN, self).__init__()
        assert output_dim >= 32 and output_dim % 8 == 0, "output_dim must be >= 32 and divisible by 8"
        self.hidden_dim = output_dim
        self.fc1 = MLP([input_dim, self.hidden_dim], last_act=True, linear=linear, use_norm=use_norm)
        self.fc2 = MLP([self.hidden_dim // 2, self.hidden_dim // 2], last_act=True, linear=linear)
        self.fc3 = MLP([self.hidden_dim // 4, self.hidden_dim // 4], last_act=True, linear=linear)
        self.fc4 = MLP([self.hidden_dim // 8, self.hidden_dim // 8], last_act=True, linear=linear)

    def forward(self, x):
        _shape = x.shape
        if len(_shape) > 2:
            x = x.view(-1, _shape[-1])
        
        x = self.fc1(x)

        x1 = x[:, :self.hidden_dim // 2]
        x = x[:, self.hidden_dim // 2:]
        x = self.fc2(x)

        x2 = x[:, :self.hidden_dim // 4]
        x = x[:, self.hidden_dim // 4:]
        x = self.fc3(x)

        x3 = x[:, :self.hidden_dim // 8]
        x = x[:, self.hidden_dim // 8:]
        x4 = self.fc4(x)

        out = torch.cat([x1, x2, x3, x4], dim=1)
        
        if len(_shape) > 2:
            out = out.view(_shape[0], _shape[1], -1)
        return out


# 将MLP和RNN以3:1的比例融合
class MLPRNN(nn.Module):
    def __init__(self, input_dim, output_dim, rnn=nn.GRU, *args, **kwargs):
        super(MLPRNN, self).__init__()
        assert output_dim % 4 == 0, "output_dim must be divisible by 4"
        self.rnn_size = output_dim // 4
        self.rnn_linear = MLP([input_dim, 3 * self.rnn_size])
        self.rnn = rnn(input_dim, self.rnn_size, *args, **kwargs)

    def forward(self, x, rnn_state: torch.Tensor):
        rnn_linear_out = self.rnn_linear(x)
        rnn_out, rnn_state = self.rnn(x, rnn_state)
        out = torch.cat([rnn_linear_out, rnn_out], dim=-1)
        return out, rnn_state
    
    
    
# 深度可分离卷积层，参数更少，效率比Conv2d更高
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=2):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.apply(initialize_weights)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# 卷积网络块
class ConvBlock(nn.Module):
    def __init__(self,
                 channels: list[tuple],
                 kernel_size: list[int],
                 stride: list[int],
                 padding: list[int],
                 output_dim,
                 input_shape=(3, 84, 84),
                 use_norm=False,
                 use_depthwise=False,
                 activation=nn.ReLU(inplace=True)
                 ):
        super(ConvBlock, self).__init__()
        self.conv_layers = nn.Sequential()
        for i, (in_channels, out_channels) in enumerate(channels):
            if use_depthwise:
                self.conv_layers.add_module(f'conv_dw_{i}', DepthwiseSeparableConv(in_channels,
                                                                                   out_channels,
                                                                                   kernel_size[i],
                                                                                   stride[i],
                                                                                   padding[i]))
            else:
                self.conv_layers.add_module(f'conv_{i}', nn.Conv2d(in_channels,
                                                                   out_channels,
                                                                   kernel_size[i],
                                                                   stride[i],
                                                                   padding[i]))
            if use_norm:
                self.conv_layers.add_module(f'bn_{i}', nn.BatchNorm2d(out_channels))
            self.conv_layers.add_module(f'act_{i}', activation)
            self.conv_layers.add_module(f'pool_{i}', nn.MaxPool2d(kernel_size=(2, 2)))
            
        self.output_dim = output_dim
        self._initialize_fc(input_shape, channels)

    def _initialize_fc(self, input_shape, channels):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            x = dummy_input
            for layer in self.conv_layers:
                x = layer(x)
            assert len(x.shape) == 4
            n_features = x.size(1) * x.size(2) * x.size(3)
            self.fc = MLP([n_features, self.output_dim], last_act=True)
            logger.info(f'ConvBlock output dim: {n_features}')


    def forward(self, x):
        features = self.conv_layers(x)
        flat = torch.flatten(features, 1) 
        out = self.fc(flat)
        return out
    



