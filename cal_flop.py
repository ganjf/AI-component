import torch
from model import densenet3d
from resnet import generate_model
from xception import Xception3d
from torchsummary import summary


def cal_param(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('Number of params: %.2fM' % (total / 1e6))

def cal_flops(model, input):
    multiply_adds = True # FLOPs include multiply ops and add ops
    list_conv=[]
    def conv_hook(module, input, output):
        batch_size, input_channels, input_z, input_height, input_width = input[0].size()
        output_channels, output_z, output_height, output_width = output[0].size()
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.kernel_size[2] * module.in_channels / module.groups * (2 if multiply_adds else 1)
        bias_ops = 1 if module.bias is not None else 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_z * output_height * output_width
        list_conv.append(flops)

    list_linear=[] 
    def linear_hook(module, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1
        weight_ops = module.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = module.bias.nelement()
        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn=[] 
    def bn_hook(module, input, output):
        list_bn.append(input[0].nelement())

    list_relu=[] 
    def relu_hook(module, input, output):
        list_relu.append(input[0].nelement())

    list_pooling=[]
    def pooling_hook(module, input, output):
        batch_size, input_channels, input_z, input_height, input_width = input[0].size()
        output_channels, output_z, output_height, output_width = output[0].size()
        kernel_ops = module.kernel_size * module.kernel_size * module.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_z * output_height * output_width
        list_pooling.append(flops)


    def register_flops_hook(net):
        for m in list(net.children()):
            if isinstance(m, torch.nn.Conv3d):
                m.register_forward_hook(conv_hook)
            elif isinstance(m, torch.nn.Linear):
                m.register_forward_hook(linear_hook)
            elif isinstance(m, torch.nn.BatchNorm3d):
                m.register_forward_hook(bn_hook)
            elif isinstance(m, torch.nn.ReLU):
                m.register_forward_hook(relu_hook)
            elif isinstance(m, torch.nn.MaxPool3d) or isinstance(m, torch.nn.AvgPool3d):
                m.register_forward_hook(pooling_hook)
            else:
                register_flops_hook(m)

    register_flops_hook(model)
    out = model(input)
    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))
    
    print('Number of FLOPs: %f' % (total_flops))
    print('Number of FLOPs: %.2fG' % (total_flops / 1e9))


if __name__=='__main__':
    # model = densenet3d().cuda()
    # model = generate_model(model_depth=50, n_classes=4).cuda()
    # model = generate_model(model_depth=101, n_classes=4).cuda()
    model = Xception3d(num_classes=4).cuda()
    input = torch.randn(1, 1, 16, 128, 128).cuda()
    summary(model, (1, 16, 128, 128))
    cal_param(model)
    cal_flops(model, input)