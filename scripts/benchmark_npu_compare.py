# Run PyTorch on N windows
# Save expected reconstruction + anomaly scores
# Later compare against NPU output


# Run PyTorch model
pytorch_output = model(x_num, x_cat)

# Save input for NPU host
np.save("benchmark_inputs_x_num.npy", x_num_np)
np.save("benchmark_inputs_x_cat.npy", x_cat_np)

# Run IRON/NPU executable
subprocess.run([
    "./build/host.exe",
    "--xclbin", "build/final.xclbin",
    "--insts", "build/insts.bin",
    "--input", "benchmark_inputs.npy",
    "--output", "npu_output.npy",
])

# Load NPU result
npu_output = np.load("npu_output.npy")

# Compare
diff = np.abs(pytorch_output - npu_output)
