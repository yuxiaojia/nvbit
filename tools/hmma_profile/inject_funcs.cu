/*
 * SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdint.h>
#include <stdio.h>
#include "utils/utils.h"
#include "nvbit_reg_rw.h"

extern "C" __device__ __noinline__ void count_instrs(int predicate,
                                                     int count_warp_level,
                                                     uint64_t pcounter) {
    /* all the active threads will compute the active mask */
    const int active_mask = __ballot_sync(__activemask(), 1);

    /* compute the predicate mask */
    const int predicate_mask = __ballot_sync(__activemask(), predicate);

    /* each thread will get a lane id (get_lane_id is implemented in
     * utils/utils.h) */
    const int laneid = get_laneid();

    /* get the id of the first active thread */
    const int first_laneid = __ffs(active_mask) - 1;

    /* count all the active thread */
    const int num_threads = __popc(predicate_mask);

    /* only the first active thread will perform the atomic */
    if (first_laneid == laneid) {
        if (count_warp_level) {
            /* num threads can be zero when accounting for predicates off */
            if (num_threads > 0) {
                atomicAdd((unsigned long long*)pcounter, 1);
            }
        } else {
            atomicAdd((unsigned long long*)pcounter, num_threads);
        }
    }
}

/* NEW: Runtime non-zero selector with atomic batch claiming */
// extern "C" __device__ __noinline__ void select_nonzero_instr(int predicate,
//                                                             int reg_dst_num,
//                                                             uint32_t batch_id,
//                                                             uint32_t kernel_id,
//                                                             uint32_t instr_idx,
//                                                             int* batch_flags,    // Array of flags per batch
//                                                             int* total_selected) {
//     if (!predicate) return;

//     /* Check if this batch already has a selection */
//     if (batch_flags[batch_id] != 0) return;  // Batch already selected, skip

//     /* Read destination register */
//     int dest_value = nvbit_read_reg(reg_dst_num);
    
//     /* Only proceed if non-zero */
//     if (dest_value != 0) {
//         /* Try to atomically claim this batch - only first thread with non-zero succeeds */
//         if (atomicExch(&batch_flags[batch_id], 1) == 0) {
//             /* This thread won the race - it's the selected instruction for this batch */
//             int selection_id = atomicAdd(total_selected, 1);
            
//             /* Print only from first thread to avoid spam */
//             if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 &&
//                 blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
//                 printf("==> RUNTIME SELECTED: Batch %d, Kernel %d, Instruction %d, Reg R%d = %d (Selection #%d)\n",
//                        batch_id, kernel_id, instr_idx, reg_dst_num, dest_value, selection_id);
//             }
//         }
//     }
//     /* If dest_value == 0, do nothing - keep looking for non-zero in this batch */
// }

// extern "C" __device__ __noinline__ void check_nonzero_simple(int pred, int reg_dst_num, uint32_t batch_id, uint32_t kernel_id, uint32_t instr_idx) {
//     if (!pred) return;
    
//     int dest_value = nvbit_read_reg(reg_dst_num);
//     if (dest_value != 0 && threadIdx.x == 0 && blockIdx.x == 0) {
//         printf("NONZERO FOUND: Batch %d, Kernel %d, Instruction %d, Reg R%d = %d\n",
//                batch_id, kernel_id, instr_idx, reg_dst_num, dest_value);
//     }
// }

extern "C" __device__ __noinline__ void check_nonzero_simple(int reg_dst_num, 
                                                             uint32_t batch_id, 
                                                             uint32_t kernel_id, 
                                                             uint32_t instr_idx) {
    // Filter to prevent spam (like NVBITFI does)
    uint32_t smid;
    asm("mov.u32 %0, %smid;" :"=r"(smid));
    uint32_t laneid;
    asm("mov.u32 %0, %laneid;" :"=r"(laneid));
    
    // Only print from SM 0, Lane 0
    if (smid != 0 || laneid != 0) return;
    
    int raw_bits = nvbit_read_reg(reg_dst_num);
    if (raw_bits != 0) {
        // Interpret as FP16 (same as reference code)
        half original_half;
        *reinterpret_cast<unsigned short*>(&original_half) = (unsigned short)(raw_bits & 0xFFFF);
        
        // Convert to float for decimal display
        float decimal_value = __half2float(original_half);
        
        printf("NONZERO: Kernel %d, Instr %d, Reg R%d = %.6f (bits:0x%04x, raw:0x%x)\n",
               kernel_id, instr_idx, reg_dst_num, 
               decimal_value, (unsigned short)(raw_bits & 0xFFFF), raw_bits);
    }
}
