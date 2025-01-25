# Runs NVCC directly; it's a backup to `build.rs`'s automatic building that gives clearer compile errors
nvcc src/cuda/cuda.cu -gencode "arch=compute_89,code=sm_89" -ptx