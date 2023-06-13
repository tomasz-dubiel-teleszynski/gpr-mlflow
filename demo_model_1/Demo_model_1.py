from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, WhiteKernel

def linear_kernel(sigma_lk,noise_level_lk):
    sigma_bounds_lk = (1e-5,1e-5)
    kernel = ConstantKernel() \
        + DotProduct(sigma_0=sigma_lk,sigma_0_bounds=sigma_bounds_lk) \
        + WhiteKernel(noise_level=noise_level_lk)  
    return kernel