from torch_NN import *
import torch


model = CNN().to("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load('model_epoch_40.pth'))

dummy_input = torch.randn(1, 1, 28, 28).to("cuda" if torch.cuda.is_available() else "cpu")
# 导出模型
torch.onnx.export(model,  # 要导出的模型

                  dummy_input,  # 模型的输入数据

                  "model.onnx",  # 要保存的文件名

                  export_params=True,  # 是否导出模型参数

                  opset_version=11,  # ONNX版本

                  do_constant_folding=True,  # 是否执行常量折叠优化

                  input_names=['input'],  # 输入数据的名字

                  output_names=['output'],  # 输出数据的名字

                  dynamic_axes={'input': {0: 'batch_size'},  # 输入数据的动态轴

                                'output': {0: 'batch_size'}})