// A BF16 GEMM built from CUTLASS 3.x collectives for Hopper (SM90a).
// Tile, cluster and pipeline stages are set at compile time, e.g.
//   make 10_cutlass_gemm TILE="-DTM=128 -DTN=256 -DTK=64 -DSTAGES=4"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "gemm_check.h"
using namespace cute;

#ifndef TM
#define TM 128
#endif
#ifndef TN
#define TN 256
#endif
#ifndef TK
#define TK 64
#endif
#ifndef CM
#define CM 2                                // cluster of CM x 1 thread blocks
#endif
#ifndef STAGES
#define STAGES 0                            // 0: as many as fit in shared memory
#endif
#ifndef SCHEDULE
#define SCHEDULE cutlass::gemm::collective::KernelScheduleAuto  // or cutlass::gemm::KernelTmaWarpSpecializedPingpong
#endif
#ifndef EPILOGUE
#define EPILOGUE cutlass::epilogue::collective::EpilogueScheduleAuto  // or cutlass::epilogue::TmaWarpSpecialized
#endif

using T = cutlass::bfloat16_t;
using RowMajor = cutlass::layout::RowMajor;   // A (M x K), B (K x N), C (M x N), all row-major
using TileShape = Shape<Int<TM>, Int<TN>, Int<TK>>;
using ClusterShape = Shape<Int<CM>, _1, _1>;

using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, float, float,
    T, RowMajor, 8, T, RowMajor, 8, EPILOGUE>::CollectiveOp;

using Stages = cute::conditional_t<STAGES == 0,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
    cutlass::gemm::collective::StageCount<STAGES>>;

using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, T, RowMajor, 8, T, RowMajor, 8, float,
    TileShape, ClusterShape, Stages, SCHEDULE>::CollectiveOp;

using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

int main(int argc, char **argv) {
  printf("tile %d x %d x %d, cluster %d x 1, %d stages, %d bytes of shared memory\n", TM, TN, TK, CM,
         (int)Mainloop::DispatchPolicy::Stages, (int)sizeof(typename Kernel::SharedStorage));
  for (int i = 1; i < argc; i++) {
    Problem p(atoi(argv[i]));
    int N = p.N;
    auto sA = cutlass::make_cute_packed_stride(typename Kernel::StrideA{}, {N, N, 1});
    auto sB = cutlass::make_cute_packed_stride(typename Kernel::StrideB{}, {N, N, 1});
    auto sC = cutlass::make_cute_packed_stride(typename Kernel::StrideC{}, {N, N, 1});
    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, {N, N, N, 1},
                                  {(T *)p.A, sA, (T *)p.B, sB},
                                  {{1.0f, 0.0f}, (T *)p.C, sC, (T *)p.C, sC}};
    Gemm gemm;
    void *work; cudaMalloc(&work, Gemm::get_workspace_size(args) + 1);
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) { printf("cannot implement N %d\n", N); continue; }
    gemm.initialize(args, work);
    double t = p.ms([&] { gemm.run(); });
    p.report("CUTLASS", t);
  }
}
