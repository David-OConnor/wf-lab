# Backup to `build.rs` to get clearer compile errors
nvcc src/cuda/cuda.cu -gencode "arch=compute_89,code=sm_89" -ptx