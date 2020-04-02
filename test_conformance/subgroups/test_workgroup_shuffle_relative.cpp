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
#include "harness/conversions.h"
#include "harness/typeWrappers.h"

int
test_work_group_functions_shuffle_relative(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    int error = false;
    std::vector<std::string> required_extensions;
    required_extensions = { "cl_khr_subgroup_shuffle_relative" };
    error |= test<cl_int, SHF<cl_int, 1>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle_up", shuffle_up_source, 0, required_extensions);
    error |= test<cl_uint, SHF<cl_uint, 1>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle_up", shuffle_up_source, 0, required_extensions);
    error |= test<cl_long, SHF<cl_long, 1>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle_up", shuffle_up_source, 0, required_extensions);
    error |= test<cl_ulong, SHF<cl_ulong, 1>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle_up", shuffle_up_source, 0, required_extensions);
    error |= test<cl_short, SHF<cl_short, 1>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle_up", shuffle_up_source, 0, required_extensions);
    error |= test<cl_ushort, SHF<cl_ushort, 1>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle_up", shuffle_up_source, 0, required_extensions);
    error |= test<cl_char, SHF<cl_char, 1>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle_up", shuffle_up_source, 0, required_extensions);
    error |= test<cl_uchar, SHF<cl_uchar, 1>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle_up", shuffle_up_source, 0, required_extensions);
    error |= test<cl_float, SHF<cl_float, 1>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle_up", shuffle_up_source, 0, required_extensions);
    error |= test<cl_double, SHF<cl_double, 1>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle_up", shuffle_up_source, 0, required_extensions);
    error |= test<subgroups::cl_half, SHF<subgroups::cl_half, 1>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle_up", shuffle_up_source, 0, required_extensions);

    error |= test<cl_int, SHF<cl_int, 2>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle_down", shuffle_down_source, 0, required_extensions);
    error |= test<cl_uint, SHF<cl_uint, 2>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle_down", shuffle_down_source, 0, required_extensions);
    error |= test<cl_long, SHF<cl_long, 2>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle_down", shuffle_down_source, 0, required_extensions);
    error |= test<cl_ulong, SHF<cl_ulong, 2>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle_down", shuffle_down_source, 0, required_extensions);
    error |= test<cl_short, SHF<cl_short, 2>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle_down", shuffle_down_source, 0, required_extensions);
    error |= test<cl_ushort, SHF<cl_ushort, 2>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle_down", shuffle_down_source, 0, required_extensions);
    error |= test<cl_char, SHF<cl_char, 2>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle_down", shuffle_down_source, 0, required_extensions);
    error |= test<cl_uchar, SHF<cl_uchar, 2>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle_down", shuffle_down_source, 0, required_extensions);
    error |= test<cl_float, SHF<cl_float, 2>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle_down", shuffle_down_source, 0, required_extensions);
    error |= test<cl_double, SHF<cl_double, 2>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle_down", shuffle_down_source, 0, required_extensions);
    error |= test<subgroups::cl_half, SHF<subgroups::cl_half, 2>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle_down", shuffle_down_source, 0, required_extensions);

    return error;
}

