## 简介

这是一个基于Tianshou框架的深度强化学习（DRL）实验项目，适用于Gymnasium、Pettingzoo和Atari等环境。该项目用于**个人学习和研究**。

## 安装

### Python版本要求

> 请使用`Python 3.11`版本，不要使用3.10或3.12。

最好安装Anaconda，使用如下命令创建和激活环境：

```
conda create -n drl python=3.11
conda activate drl
```

### 安装Tianshou和依赖项

1. 克隆Tianshou仓库并安装：

	```shell
	git clone https://github.com/thu-ml/tianshou.git
	cd tianshou
	conda activate drl  
	pip install .
	```

2. 安装基础依赖：

	```shell
	pip install -r requirements-base.txt
	```

3. 安装其他依赖：

	```shell
	pip install -r requirements.txt
	```

## 使用

本项目提供了一些示例代码，可以帮助你快速开始使用Tianshou框架进行DRL实验。

由于Tianshou在算法、训练方法等方面比较完善，目前我主要试验**不同神经网络**的开发，在tianshou框架下不同算法的训练效率以及水准等。

读者可以参照`/utils/model.py`，尝试以下神经网络进行**特征提取**：

```python
# MLP Concat
class PSCN(nn.Module):
    def __init__(self, input_dim, output_dim, linear=nn.Linear):
        super(PSCN, self).__init__()
        assert output_dim >= 32 and output_dim % 8 == 0, "output_dim must be >= 32 and divisible by 8"
        self.hidden_dim = output_dim
        self.fc1 = MLP([input_dim, self.hidden_dim], last_act=True, linear=linear)
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


# conv layer
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3, 
                 stride=1, 
                 padding=2):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, 
                                   in_channels, 
                                   kernel_size, 
                                   stride, 
                                   padding, 
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, 
                                   out_channels, 
                                   kernel_size=1)
        self.apply(initialize_weights)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
```

## 贡献

欢迎提交问题（Issues）和拉取请求（Pull Requests）以改进此项目。请确保在提交之前阅读并遵循贡献指南。

## 协议

本项目使用MIT协议。请参阅LICENSE文件以获取更多信息。
