// RUN: buddy-opt %s -batch-matmul-vectorization-decode | FileCheck %s

// A transposed-B batch_matmul (reduction over B's innermost dim, as GPT-2's
// Q @ K^T emits) must be left alone by this pass; it is lowered by the generic
// linalg path, which handles arbitrary indexing maps correctly.

#mapA = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#mapB = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#mapC = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

func.func @batch_decode_transpose_b(%A: memref<12x1x64xf16>,
                                     %B: memref<12x1024x64xf16>,
                                     %C: memref<12x1x1024xf16>) {
  linalg.batch_matmul indexing_maps = [#mapA, #mapB, #mapC]
    ins(%A, %B : memref<12x1x64xf16>, memref<12x1024x64xf16>)
    outs(%C : memref<12x1x1024xf16>)
  return
}

// CHECK-LABEL: func.func @batch_decode_transpose_b
// CHECK:       linalg.batch_matmul
// CHECK-NOT:   vector.load
// CHECK-NOT:   vector.fma
// CHECK-NOT:   scf.parallel
// CHECK:       return
