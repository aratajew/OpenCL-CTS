//
// Copyright (c) 2017 The Khronos Group Inc.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#include "procs.h"
#include "subhelpers.h"
#include "workgroup_kernel_sources.h"
#include "harness/typeWrappers.h"


int
test_work_group_functions_clustered_reduce(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    int error = false;
    std::vector<std::string> required_extensions;
    required_extensions = { "cl_khr_subgroup_clustered_reduce" };
    error |= test<cl_int, RED_CLU<cl_int, 0>, G, L>::run(device, context, queue, num_elements, "test_redadd_clustered", redadd_clustered_source, 0, required_extensions);
    error |= test<cl_uint, RED_CLU<cl_uint, 0>, G, L>::run(device, context, queue, num_elements, "test_redadd_clustered", redadd_clustered_source, 0, required_extensions);
    error |= test<cl_long, RED_CLU<cl_long, 0>, G, L>::run(device, context, queue, num_elements, "test_redadd_clustered", redadd_clustered_source, 0, required_extensions);
    error |= test<cl_ulong, RED_CLU<cl_ulong, 0>, G, L>::run(device, context, queue, num_elements, "test_redadd_clustered", redadd_clustered_source, 0, required_extensions);
    error |= test<cl_short, RED_CLU<cl_short, 0>, G, L>::run(device, context, queue, num_elements, "test_redadd_clustered", redadd_clustered_source, 0, required_extensions);
    error |= test<cl_ushort, RED_CLU<cl_ushort, 0>, G, L>::run(device, context, queue, num_elements, "test_redadd_clustered", redadd_clustered_source, 0, required_extensions);
    error |= test<cl_char, RED_CLU<cl_char, 0>, G, L>::run(device, context, queue, num_elements, "test_redadd_clustered", redadd_clustered_source, 0, required_extensions);
    error |= test<cl_uchar, RED_CLU<cl_uchar, 0>, G, L>::run(device, context, queue, num_elements, "test_redadd_clustered", redadd_clustered_source, 0, required_extensions);
    error |= test<cl_float, RED_CLU<cl_float, 0>, G, L>::run(device, context, queue, num_elements, "test_redadd_clustered", redadd_clustered_source, 0, required_extensions);
    error |= test<cl_double, RED_CLU<cl_double, 0>, G, L>::run(device, context, queue, num_elements, "test_redadd_clustered", redadd_clustered_source, 0, required_extensions);
    //error |= test<subgroups::cl_half, RED_CLU<subgroups::cl_half, 0>, G, L>::run(device, context, queue, num_elements, "test_redadd_clustered", redadd_clustered_source, 0, required_extensions);

    error |= test<cl_int, RED_CLU<cl_int, 1>, G, L>::run(device, context, queue, num_elements, "test_redmax_clustered", redmax_clustered_source, 0, required_extensions);
    error |= test<cl_uint, RED_CLU<cl_uint, 1>, G, L>::run(device, context, queue, num_elements, "test_redmax_clustered", redmax_clustered_source, 0, required_extensions);
    error |= test<cl_long, RED_CLU<cl_long, 1>, G, L>::run(device, context, queue, num_elements, "test_redmax_clustered", redmax_clustered_source, 0, required_extensions);
    error |= test<cl_ulong, RED_CLU<cl_ulong, 1>, G, L>::run(device, context, queue, num_elements, "test_redmax_clustered", redmax_clustered_source, 0, required_extensions);
    error |= test<cl_short, RED_CLU<cl_short, 1>, G, L>::run(device, context, queue, num_elements, "test_redmax_clustered", redmax_clustered_source, 0, required_extensions);
    error |= test<cl_ushort, RED_CLU<cl_ushort, 1>, G, L>::run(device, context, queue, num_elements, "test_redmax_clustered", redmax_clustered_source, 0, required_extensions);
    error |= test<cl_char, RED_CLU<cl_char, 1>, G, L>::run(device, context, queue, num_elements, "test_redmax_clustered", redmax_clustered_source, 0, required_extensions);
    error |= test<cl_uchar, RED_CLU<cl_uchar, 1>, G, L>::run(device, context, queue, num_elements, "test_redmax_clustered", redmax_clustered_source, 0, required_extensions);
    error |= test<cl_float, RED_CLU<cl_float, 1>, G, L>::run(device, context, queue, num_elements, "test_redmax_clustered", redmax_clustered_source, 0, required_extensions);
    error |= test<cl_double, RED_CLU<cl_double, 1>, G, L>::run(device, context, queue, num_elements, "test_redmax_clustered", redmax_clustered_source, 0, required_extensions);
    //error |= test<subgroups::cl_half, RED_CLU<subgroups::cl_half, 1>, G, L>::run(device, context, queue, num_elements, "test_redmax_clustered", redmax_clustered_source, 0, required_extensions);

    error |= test<cl_int, RED_CLU<cl_int, 2>, G, L>::run(device, context, queue, num_elements, "test_redmin_clustered", redmin_clustered_source, 0, required_extensions);
    error |= test<cl_uint, RED_CLU<cl_uint, 2>, G, L>::run(device, context, queue, num_elements, "test_redmin_clustered", redmin_clustered_source, 0, required_extensions);
    error |= test<cl_long, RED_CLU<cl_long, 2>, G, L>::run(device, context, queue, num_elements, "test_redmin_clustered", redmin_clustered_source, 0, required_extensions);
    error |= test<cl_ulong, RED_CLU<cl_ulong, 2>, G, L>::run(device, context, queue, num_elements, "test_redmin_clustered", redmin_clustered_source, 0, required_extensions);
    error |= test<cl_short, RED_CLU<cl_short, 2>, G, L>::run(device, context, queue, num_elements, "test_redmin_clustered", redmin_clustered_source, 0, required_extensions);
    error |= test<cl_ushort, RED_CLU<cl_ushort, 2>, G, L>::run(device, context, queue, num_elements, "test_redmin_clustered", redmin_clustered_source, 0, required_extensions);
    error |= test<cl_char, RED_CLU<cl_char, 2>, G, L>::run(device, context, queue, num_elements, "test_redmin_clustered", redmin_clustered_source, 0, required_extensions);
    error |= test<cl_uchar, RED_CLU<cl_uchar, 2>, G, L>::run(device, context, queue, num_elements, "test_redmin_clustered", redmin_clustered_source, 0, required_extensions);
    error |= test<cl_float, RED_CLU<cl_float, 2>, G, L>::run(device, context, queue, num_elements, "test_redmin_clustered", redmin_clustered_source, 0, required_extensions);
    error |= test<cl_double, RED_CLU<cl_double, 2>, G, L>::run(device, context, queue, num_elements, "test_redmin_clustered", redmin_clustered_source, 0, required_extensions);
    //error |= test<subgroups::cl_half, RED_CLU<subgroups::cl_half, 2>, G, L>::run(device, context, queue, num_elements, "test_redmin_clustered", redmin_clustered_source, 0, required_extensions);

    error |= test<cl_int, RED_CLU<cl_int, 3>, G, L>::run(device, context, queue, num_elements, "test_redmul_clustered", redmul_clustered_source, 0, required_extensions);
    error |= test<cl_uint, RED_CLU<cl_uint, 3>, G, L>::run(device, context, queue, num_elements, "test_redmul_clustered", redmul_clustered_source, 0, required_extensions);
    error |= test<cl_long, RED_CLU<cl_long, 3>, G, L>::run(device, context, queue, num_elements, "test_redmul_clustered", redmul_clustered_source, 0, required_extensions);
    error |= test<cl_ulong, RED_CLU<cl_ulong, 3>, G, L>::run(device, context, queue, num_elements, "test_redmul_clustered", redmul_clustered_source, 0, required_extensions);
    error |= test<cl_short, RED_CLU<cl_short, 3>, G, L>::run(device, context, queue, num_elements, "test_redmul_clustered", redmul_clustered_source, 0, required_extensions);
    error |= test<cl_ushort, RED_CLU<cl_ushort, 3>, G, L>::run(device, context, queue, num_elements, "test_redmul_clustered", redmul_clustered_source, 0, required_extensions);
    error |= test<cl_char, RED_CLU<cl_char, 3>, G, L>::run(device, context, queue, num_elements, "test_redmul_clustered", redmul_clustered_source, 0, required_extensions);
    error |= test<cl_uchar, RED_CLU<cl_uchar, 3>, G, L>::run(device, context, queue, num_elements, "test_redmul_clustered", redmul_clustered_source, 0, required_extensions);
    error |= test<cl_float, RED_CLU<cl_float, 3>, G, L>::run(device, context, queue, num_elements, "test_redmul_clustered", redmul_clustered_source, 0, required_extensions);
    error |= test<cl_double, RED_CLU<cl_double, 3>, G, L>::run(device, context, queue, num_elements, "test_redmul_clustered", redmul_clustered_source, 0, required_extensions);
    //error |= test<subgroups::cl_half, RED_CLU<subgroups::cl_half, 3>, G, L>::run(device, context, queue, num_elements, "test_redmul_clustered", redmul_clustered_source, 0, required_extensions);

    error |= test<cl_int, RED_CLU<cl_int, 4>, G, L>::run(device, context, queue, num_elements, "test_redand_clustered", redand_clustered_source, 0, required_extensions);
    error |= test<cl_uint, RED_CLU<cl_uint, 4>, G, L>::run(device, context, queue, num_elements, "test_redand_clustered", redand_clustered_source, 0, required_extensions);
    error |= test<cl_long, RED_CLU<cl_long, 4>, G, L>::run(device, context, queue, num_elements, "test_redand_clustered", redand_clustered_source, 0, required_extensions);
    error |= test<cl_ulong, RED_CLU<cl_ulong, 4>, G, L>::run(device, context, queue, num_elements, "test_redand_clustered", redand_clustered_source, 0, required_extensions);
    error |= test<cl_short, RED_CLU<cl_short, 4>, G, L>::run(device, context, queue, num_elements, "test_redand_clustered", redand_clustered_source, 0, required_extensions);
    error |= test<cl_ushort, RED_CLU<cl_ushort, 4>, G, L>::run(device, context, queue, num_elements, "test_redand_clustered", redand_clustered_source, 0, required_extensions);
    error |= test<cl_char, RED_CLU<cl_char, 4>, G, L>::run(device, context, queue, num_elements, "test_redand_clustered", redand_clustered_source, 0, required_extensions);
    error |= test<cl_uchar, RED_CLU<cl_uchar, 4>, G, L>::run(device, context, queue, num_elements, "test_redand_clustered", redand_clustered_source, 0, required_extensions);

    error |= test<cl_int, RED_CLU<cl_int, 5>, G, L>::run(device, context, queue, num_elements, "test_redor_clustered", redor_clustered_source, 0, required_extensions);
    error |= test<cl_uint, RED_CLU<cl_uint, 5>, G, L>::run(device, context, queue, num_elements, "test_redor_clustered", redor_clustered_source, 0, required_extensions);
    error |= test<cl_long, RED_CLU<cl_long, 5>, G, L>::run(device, context, queue, num_elements, "test_redor_clustered", redor_clustered_source, 0, required_extensions);
    error |= test<cl_ulong, RED_CLU<cl_ulong, 5>, G, L>::run(device, context, queue, num_elements, "test_redor_clustered", redor_clustered_source, 0, required_extensions);
    error |= test<cl_short, RED_CLU<cl_short, 5>, G, L>::run(device, context, queue, num_elements, "test_redor_clustered", redor_clustered_source, 0, required_extensions);
    error |= test<cl_ushort, RED_CLU<cl_ushort, 5>, G, L>::run(device, context, queue, num_elements, "test_redor_clustered", redor_clustered_source, 0, required_extensions);
    error |= test<cl_char, RED_CLU<cl_char, 5>, G, L>::run(device, context, queue, num_elements, "test_redor_clustered", redor_clustered_source, 0, required_extensions);
    error |= test<cl_uchar, RED_CLU<cl_uchar, 5>, G, L>::run(device, context, queue, num_elements, "test_redor_clustered", redor_clustered_source, 0, required_extensions);

    error |= test<cl_int, RED_CLU<cl_int, 6>, G, L>::run(device, context, queue, num_elements, "test_redxor_clustered", redxor_clustered_source, 0, required_extensions);
    error |= test<cl_uint, RED_CLU<cl_uint, 6>, G, L>::run(device, context, queue, num_elements, "test_redxor_clustered", redxor_clustered_source, 0, required_extensions);
    error |= test<cl_long, RED_CLU<cl_long, 6>, G, L>::run(device, context, queue, num_elements, "test_redxor_clustered", redxor_clustered_source, 0, required_extensions);
    error |= test<cl_ulong, RED_CLU<cl_ulong, 6>, G, L>::run(device, context, queue, num_elements, "test_redxor_clustered", redxor_clustered_source, 0, required_extensions);
    error |= test<cl_short, RED_CLU<cl_short, 6>, G, L>::run(device, context, queue, num_elements, "test_redxor_clustered", redxor_clustered_source, 0, required_extensions);
    error |= test<cl_ushort, RED_CLU<cl_ushort, 6>, G, L>::run(device, context, queue, num_elements, "test_redxor_clustered", redxor_clustered_source, 0, required_extensions);
    error |= test<cl_char, RED_CLU<cl_char, 6>, G, L>::run(device, context, queue, num_elements, "test_redxor_clustered", redxor_clustered_source, 0, required_extensions);
    error |= test<cl_uchar, RED_CLU<cl_uchar, 6>, G, L>::run(device, context, queue, num_elements, "test_redxor_clustered", redxor_clustered_source, 0, required_extensions);

    error |= test<cl_int, RED_CLU<cl_int, 7>, G, L>::run(device, context, queue, num_elements, "test_redand_clustered_logical", redand_clustered_logical_source, 0, required_extensions);
    error |= test<cl_int, RED_CLU<cl_int, 8>, G, L>::run(device, context, queue, num_elements, "test_redor_clustered_logical", redor_clustered_logical_source, 0, required_extensions);
    error |= test<cl_int, RED_CLU<cl_int, 9>, G, L>::run(device, context, queue, num_elements, "test_redxor_clustered_logical", redxor_clustered_logical_source, 0, required_extensions);

    return error;
}

