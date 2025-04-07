import torch

# 创建一个随机的三维张量作为输入数据
input_data = torch.randn((3, 1024, 3), dtype=torch.complex64)

# 进行三维傅里叶变换
fft_result = torch.fft.fftn(input_data)

# 进行逆三维傅里叶变换
ifft_result = torch.fft.ifftn(fft_result)

# 打印结果
print("原始数据:\n", input_data.shape)
print("\n三维傅里叶变换结果:\n", fft_result.shape)
print("\n逆三维傅里叶变换结果:\n", ifft_result.shape)
