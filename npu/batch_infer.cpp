/*
Authors: Daniel Eyraud, Mary-Claire Ridgeway

Generic batched XRT host for FLAIR kernels with two inputs and one output.
Loads the design and shared parameter buffer once, then processes N windows
in groups of `batch`.

N must be a multiple of `batch`. Input and output bf16 values are transferred
as raw 16-bit words without host-side conversion.

SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
*/

#include "test_utils.h"

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using u16 = std::uint16_t; // bf16 storage (raw bits)

// Load a binary bf16 buffer without interpreting its values on the host.
static std::vector<u16> load_u16(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f)
    throw std::runtime_error("could not open " + path);
  std::streamsize n = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<u16> v(n / sizeof(u16));
  if (!f.read(reinterpret_cast<char *>(v.data()), n))
    throw std::runtime_error("could not read " + path);
  return v;
}

int main(int argc, char **argv) {
  if (argc < 11) {
    std::cerr << "usage: batch_infer <xclbin> <insts> <in1> <in2> <out> <N> "
                 "<batch> <in1_vol> <in2_vol> <out_vol> [kernel]\n";
    return 1;
  }
  std::string xclbin = argv[1], insts = argv[2];
  std::string in1_file = argv[3], in2_file = argv[4], out_file = argv[5];
  int N = std::stoi(argv[6]);
  int batch = std::stoi(argv[7]);
  int in1_vol = std::stoi(argv[8]);  // input bf16 elements per window
  int in2_vol = std::stoi(argv[9]);  // shared parameter elements
  int out_vol = std::stoi(argv[10]); // output bf16 elements per window
  std::string kernel_name = (argc > 11) ? argv[11] : "MLIR_AIE";

  if (batch <= 0 || N % batch != 0) {
    std::cerr << "N (" << N << ") must be a positive multiple of batch ("
              << batch << ")\n";
    return 1;
  }
  int G = N / batch; // dispatch groups
  int batch_in1_vol = batch * in1_vol;
  int batch_out_vol = batch * out_vol;

  std::vector<u16> in1 = load_u16(in1_file); // N per-window inputs
  std::vector<u16> in2 = load_u16(in2_file); // one shared parameter buffer
  if ((int)in1.size() < N * in1_vol || (int)in2.size() < in2_vol) {
    std::cerr << "input file smaller than expected (in1=" << in1.size()
              << " need " << (long)N * in1_vol << ", in2=" << in2.size()
              << " need " << in2_vol << ")\n";
    return 1;
  }
  std::vector<u16> out(static_cast<size_t>(N) * out_vol, 0);

  std::vector<uint32_t> instr_v = test_utils::load_instr_binary(insts);

  xrt::device device;
  xrt::kernel kernel;
  test_utils::init_xrt_load_kernel(device, kernel, 0, xclbin, kernel_name);

  // xrt_test_wrapper argument groups:
  //   1 = instructions, 3 = input, 4 = parameters, 5 = output,
  //   6 = control packets, 7 = trace.
  // Input and output BOs hold one dispatch; parameters remain shared.
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_in1 = xrt::bo(device, batch_in1_vol * sizeof(u16),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_in2 = xrt::bo(device, in2_vol * sizeof(u16), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(4));
  auto bo_out = xrt::bo(device, batch_out_vol * sizeof(u16),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
  auto bo_ctrl = xrt::bo(device, 8, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));
  auto bo_trace = xrt::bo(device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(7));

  // Instructions and shared parameters are transferred once before dispatch.
  memcpy(bo_instr.map<void *>(), instr_v.data(), instr_v.size() * sizeof(int));
  memcpy(bo_in2.map<void *>(), in2.data(), in2_vol * sizeof(u16));
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in2.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  u16 *map_in1 = bo_in1.map<u16 *>();
  u16 *map_out = bo_out.map<u16 *>();
  const unsigned opcode = 3;

  // Measure dispatch processing, including host copies, DMA sync, launch,
  // kernel completion, and output transfer.
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int g = 0; g < G; g++) {
    memcpy(map_in1, &in1[(size_t)g * batch_in1_vol],
           batch_in1_vol * sizeof(u16));
    bo_in1.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_in1, bo_in2, bo_out,
                      bo_ctrl, bo_trace);
    run.wait();

    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    memcpy(&out[(size_t)g * batch_out_vol], map_out,
           batch_out_vol * sizeof(u16));

    int windows_done = (g + 1) * batch;
    if ((windows_done % 100) < batch || g + 1 == G)
      std::cout << "  processed " << windows_done << "/" << N
                << " windows (" << (g + 1) << "/" << G << " dispatches)\r"
                << std::flush;
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                  .count() / 1000.0;
  std::cout << "\n" << N << " windows in " << G << " dispatches (batch="
            << batch << "), " << ms << " ms total (" << (ms * 1000.0 / N)
            << " us/window incl. host overhead)\n";

  std::ofstream fo(out_file, std::ios::binary);
  fo.write(reinterpret_cast<const char *>(out.data()),
           out.size() * sizeof(u16));
  std::cout << "wrote " << out_file << " (" << out.size() << " bf16)\n";
  return 0;
}
