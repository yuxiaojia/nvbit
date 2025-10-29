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

/* Tensor instruction selector - only selects instructions with non-zero destination registers */
extern "C" __device__ __noinline__ void check_tensor_instr(int predicate,
                                                          int reg_dst_num,
                                                          uint64_t batch_selected_ptr,
                                                          uint32_t current_batch_id,
                                                          uint64_t total_selected_ptr,
                                                          uint32_t instr_idx,
                                                          uint64_t kernel_id) {
    if (!predicate) return;

    /* Read the destination register value */
    int dest_value = nvbit_read_reg(reg_dst_num);
    
    /* Only proceed if destination register is not zero */
    if (dest_value != 0) {
        /* Cast pointers to access managed variables */
        bool* batch_selected = (bool*)batch_selected_ptr;
        uint32_t* total_selected = (uint32_t*)total_selected_ptr;
        
        /* Check if this batch already has a selection */
        if (!batch_selected[current_batch_id]) {
            /* Try to claim this batch atomically */
            if (!atomicExch((int*)&batch_selected[current_batch_id], 1)) {
                /* Successfully claimed - this is the selected instruction */
                uint32_t selection_id = atomicAdd(total_selected, 1);
                
                /* Print selection info (only from first thread to avoid spam) */
                if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 &&
                    blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
                    printf("==> RUNTIME SELECTED: Batch %d, Kernel %llu, Instruction %d, Reg R%d = %d (Selection #%d)\n",
                           current_batch_id, kernel_id, instr_idx, reg_dst_num, dest_value, selection_id);
                }
            }
        }
    }
}
