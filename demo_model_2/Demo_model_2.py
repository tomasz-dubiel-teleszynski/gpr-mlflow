from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

def nonlinear_kernel(length_scale_nlk,noise_level_nlk):
    length_scale_bounds_nlk = (1e-5,1e5)
    kernel = ConstantKernel() \
                + RBF(length_scale=length_scale_nlk,length_scale_bounds=length_scale_bounds_nlk) \
                + WhiteKernel(noise_level=noise_level_nlk)
    return kernel
