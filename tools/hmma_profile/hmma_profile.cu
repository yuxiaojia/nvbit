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

#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <unordered_set>
#include <algorithm>
#include <string>

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* nvbit utility functions */
#include "utils/utils.h"

/* kernel id counter, maintained in system memory */
uint32_t kernel_id = 0;

/* total instruction counter, maintained in system memory, incremented by
 * "counter" every time a kernel completes  */
uint64_t tot_app_instrs = 0;

/* kernel instruction counter, updated by the GPU */
__managed__ uint64_t counter = 0;

/* batch boundary tracking - ADDED */
int batch_boundaries[100];
int num_batches = 0;
int current_batch = -1;  // ADDED: track current batch
__managed__ int batch_selected[100] = {0};  // int instead of bool
__managed__ int total_selected = 0;         // int instead of uint32_t

/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
uint32_t start_grid_num = 0;
uint32_t end_grid_num = UINT32_MAX;
int verbose = 0;
int count_warp_level = 1;
int exclude_pred_off = 0;
int active_from_start = 1;
bool mangled = false;

/* used to select region of insterest when active from start is off */
bool active_region = true;

/* a pthread mutex, used to prevent multiple kernels to run concurrently and
 * therefore to "corrupt" the counter variable */
pthread_mutex_t mutex;

/* ADDED: Check if instruction is tensor core related */
bool is_tensor_instruction(Instr *instr) {
    std::string opcode = instr->getOpcode();
    return opcode.find("HMMA") != std::string::npos || 
           opcode.find("WMMA") != std::string::npos ||
           opcode.find("MMA") != std::string::npos;
}

/* ADDED: Check if kernel name suggests tensor operations */
bool is_tensor_kernel(CUcontext ctx, CUfunction func) {
    const char *kernel_name = nvbit_get_func_name(ctx, func);
    if (!kernel_name) return false;
    
    // Use same patterns as working code
    if (strstr(kernel_name, "implicit_gemm") != NULL) return true;
    if (strstr(kernel_name, "implicit_convolve_sgemm") != NULL) return true;
    if (strstr(kernel_name, "sm75_xmma_fprop_implicit_gemm") != NULL) return true;
    
    return false;
}

/* ADDED: Check if kernel is in batch and print batch starts */
// bool is_in_batch(uint32_t kernel_id) {
//     for (int i = 0; i < num_batches; i++) {
//         uint32_t start = (i == 0) ? 753 : batch_boundaries[i-1] + 1;
//         if (kernel_id >= start && kernel_id <= (uint32_t)batch_boundaries[i]) {
//             if (current_batch != i) {
//                 current_batch = i;
//                 printf("*** BATCH %d STARTS: kernels %d-%d ***\n", i, start, batch_boundaries[i]);
//             }
//             return true;
//         }
//     }
//     return false;
// }

bool is_in_batch(uint32_t kernel_id) {
    for (int i = 0; i < num_batches; i++) {
        uint32_t start = (i == 0) ? 0 : batch_boundaries[i-1] + 1;  // Change 753 to 0
        if (kernel_id >= start && kernel_id <= (uint32_t)batch_boundaries[i]) {
            if (current_batch != i) {
                current_batch = i;
                printf("*** BATCH %d STARTS: kernels %d-%d ***\n", i, start, batch_boundaries[i]);
            }
            return true;
        }
    }
    return false;
}

/* nvbit_at_init() is executed as soon as the nvbit tool is loaded. We typically
 * do initializations in this call. In this case for instance we get some
 * environment variables values which we use as input arguments to the tool */
void nvbit_at_init() {
    /* just make sure all managed variables are allocated on GPU */
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);

    /* ADDED: Initialize batch boundaries */
    batch_boundaries[0] = 49;  // Batch 0 ends at kernel 49 (kernels 0-49)
    num_batches = 1;
    int current_boundary = 49;
    while (current_boundary < 16791 && num_batches < 100) {
        current_boundary += 50;
        batch_boundaries[num_batches++] = current_boundary;
    }

    /* we get some environment variables that are going to be use to selectively
     * instrument (within a interval of kernel indexes and instructions). By
     * default we instrument everything. */
    GET_VAR_INT(
        instr_begin_interval, "INSTR_BEGIN", 0,
        "Beginning of the instruction interval where to apply instrumentation");
    GET_VAR_INT(
        instr_end_interval, "INSTR_END", UINT32_MAX,
        "End of the instruction interval where to apply instrumentation");
    GET_VAR_INT(start_grid_num, "START_GRID_NUM", 0,
                "Beginning of the kernel gird launch interval where to apply "
                "instrumentation");
    GET_VAR_INT(
        end_grid_num, "END_GRID_NUM", UINT32_MAX,
        "End of the kernel launch interval where to apply instrumentation");
    GET_VAR_INT(count_warp_level, "COUNT_WARP_LEVEL", 1,
                "Count warp level or thread level instructions");
    GET_VAR_INT(exclude_pred_off, "EXCLUDE_PRED_OFF", 0,
                "Exclude predicated off instruction from count");
    GET_VAR_INT(
        active_from_start, "ACTIVE_FROM_START", 1,
        "Start instruction counting from start or wait for cuProfilerStart "
        "and cuProfilerStop");
    GET_VAR_INT(mangled, "MANGLED_NAMES", 1,
                "Print kernel names mangled or not");

    GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
    if (active_from_start == 0) {
        active_region = false;
    }

    std::string pad(100, '-');
    printf("%s\n", pad.c_str());
}

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    /* Only process tensor kernels */
    if (!is_tensor_kernel(ctx, func)) {
        return;
    }

    /* Get related functions of the kernel (device function that can be
     * called by the kernel) */
    std::vector<CUfunction> related_functions =
        nvbit_get_related_functions(ctx, func);

    /* add kernel itself to the related function vector */
    related_functions.push_back(func);

    /* iterate on function */
    for (auto f : related_functions) {
        /* "recording" function was instrumented, if set insertion failed
         * we have already encountered this function */
        if (!already_instrumented.insert(f).second) {
            continue;
        }

        /* Get the vector of instruction composing the loaded CUFunction "f" */
        const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);

        /* If verbose we print function name and number of" static" instructions
         */
        if (verbose) {
            printf("inspecting tensor kernel %s - num instrs %ld\n",
                   nvbit_get_func_name(ctx, f), instrs.size());
        }

        /* We iterate on the vector of instruction */
        for (auto i : instrs) {
            /* Check if the instruction falls in the interval where we want to
             * instrument */
            if (i->getIdx() >= instr_begin_interval &&
                i->getIdx() < instr_end_interval) {
                
                /* ADDED: Only instrument tensor core instructions */
                if (is_tensor_instruction(i)) {
                    /* If verbose we print which instruction we are instrumenting
                     * (both offset in the function and SASS string) */
                    if (verbose == 1) {
                        i->print();
                    } else if (verbose == 2) {
                        i->printDecoded();
                    }

                    /* Extract destination register number from SASS */
                    int num_opnds = i->getNumOperands();
                    if (num_opnds == 0) continue;

                    const InstrType::operand_t *op0 = i->getOperand(0);
                    if (op0->type != InstrType::OperandType::REG) continue;

                    int dst_reg = op0->u.reg.num;

                    /* Insert a call to "check_tensor_instr" before the instruction */
                    // nvbit_insert_call(i, "select_nonzero_instr", IPOINT_BEFORE);
                    nvbit_insert_call(i, "check_nonzero_simple", IPOINT_AFTER);

                    // if (exclude_pred_off) {
                    //     nvbit_add_call_arg_guard_pred_val(i);
                    // } else {
                    //     nvbit_add_call_arg_const_val32(i, 1);
                    // }
                    
                    nvbit_add_call_arg_const_val32(i, dst_reg);              // Register number
                    nvbit_add_call_arg_const_val32(i, current_batch);        // Batch ID
                    nvbit_add_call_arg_const_val32(i, kernel_id);            // Kernel ID
                    nvbit_add_call_arg_const_val32(i, i->getIdx());          // Instruction index
                    
                    // nvbit_add_call_arg_const_val32(i, dst_reg);                    // register number
                    // nvbit_add_call_arg_const_val32(i, current_batch);              // batch ID
                    // nvbit_add_call_arg_const_val32(i, kernel_id);                  // kernel ID  
                    // nvbit_add_call_arg_const_val32(i, i->getIdx());                // instruction index
                    // nvbit_add_call_arg_const_val64(i, (uint64_t)&batch_selected);  // batch flags array
                    // nvbit_add_call_arg_const_val64(i, (uint64_t)&total_selected);  // total counter
                    
                }
            }
        }
    }
}

/* This call-back is triggered every time a CUDA driver call is encountered.
 * Here we can look for a particular CUDA driver call by checking at the
 * call back ids  which are defined in tools_cuda_api_meta.h.
 * This call back is triggered bith at entry and at exit of each CUDA driver
 * call, is_exit=0 is entry, is_exit=1 is exit.
 * */
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
    /* Identify all the possible CUDA launch events */
    if (cbid == API_CUDA_cuLaunch || cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchGrid || cbid == API_CUDA_cuLaunchGridAsync ||
        cbid == API_CUDA_cuLaunchKernel ||
        cbid == API_CUDA_cuLaunchKernelEx ||
        cbid == API_CUDA_cuLaunchKernelEx_ptsz) {
        /* cast params to launch parameter based on cbid since if we are here
         * we know these are the right parameters types */
        CUfunction func;
        if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
            cbid == API_CUDA_cuLaunchKernelEx) {
            cuLaunchKernelEx_params* p = (cuLaunchKernelEx_params*)params;
            func = p->f;
        } else {
            cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;
            func = p->f;
        }

        if (!is_exit) {
            /* if we are entering in a kernel launch:
             * 1. Lock the mutex to prevent multiple kernels to run concurrently
             * (overriding the counter) in case the user application does that
             * 2. Instrument the function if needed
             * 3. Select if we want to run the instrumented or original
             * version of the kernel
             * 4. Reset the kernel instruction counter */

            pthread_mutex_lock(&mutex);
            instrument_function_if_needed(ctx, func);

            if (active_from_start) {
                if (kernel_id >= start_grid_num && kernel_id < end_grid_num) {
                    active_region = true;
                } else {
                    active_region = false;
                }
            }

            /* MODIFIED: Only instrument if in batch */
            if (active_region && is_in_batch(kernel_id)) {
                nvbit_enable_instrumented(ctx, func, true);
            } else {
                nvbit_enable_instrumented(ctx, func, false);
            }

            counter = 0;
        } else {
            /* if we are exiting a kernel launch:
             * 1. Wait until the kernel is completed using
             * cudaDeviceSynchronize()
             * 2. Get number of thread blocks in the kernel
             * 3. Print the thread instruction counters
             * 4. Release the lock*/
            CUDA_SAFECALL(cudaDeviceSynchronize());
            
            /* MODIFIED: Only count if in batch */
            if (is_in_batch(kernel_id)) {
                tot_app_instrs += counter;
            }
            
            int num_ctas = 0;
            if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
                cbid == API_CUDA_cuLaunchKernel) {
                cuLaunchKernel_params *p2 = (cuLaunchKernel_params *)params;
                num_ctas = p2->gridDimX * p2->gridDimY * p2->gridDimZ;
            } else if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
                cbid == API_CUDA_cuLaunchKernelEx) {
                cuLaunchKernelEx_params *p2 = (cuLaunchKernelEx_params *)params;
                num_ctas = p2->config->gridDimX * p2->config->gridDimY *
                    p2->config->gridDimZ;
            }
            printf(
                "\nkernel %d - %s - #thread-blocks %d,  kernel "
                "instructions %ld, total instructions %ld\n",
                kernel_id++, nvbit_get_func_name(ctx, func, mangled), num_ctas,
                counter, tot_app_instrs);
            pthread_mutex_unlock(&mutex);
        }
    } else if (cbid == API_CUDA_cuProfilerStart && is_exit) {
        if (!active_from_start) {
            active_region = true;
        }
    } else if (cbid == API_CUDA_cuProfilerStop && is_exit) {
        if (!active_from_start) {
            active_region = false;
        }
    }
}

void nvbit_at_term() {
    printf("Total app instructions: %ld\n", tot_app_instrs);
    printf("Tensor core selections: %d out of %d batches\n", total_selected, num_batches);
}