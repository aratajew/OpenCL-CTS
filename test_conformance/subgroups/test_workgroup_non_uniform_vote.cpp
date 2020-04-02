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
test_work_group_functions_non_uniform_vote(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    int error = false;
    std::vector<std::string> required_extensions;
    required_extensions = { "cl_khr_subgroup_non_uniform_vote" };
    error |= test<cl_int, ELECT, G, L>::run(device, context, queue, num_elements, "test_elect", elect_source, 0, required_extensions);
    error |= test<int, AAN<int, 0>, G, L>::run(device, context, queue, num_elements, "test_non_uniform_any", non_uniform_any_source, 0, required_extensions);
    error |= test<int, AAN<int, 1>, G, L>::run(device, context, queue, num_elements, "test_non_uniform_all", non_uniform_all_source, 0, required_extensions);
    error |= test<cl_char, AAN<cl_char, 2>, G, L>::run(device, context, queue, num_elements, "test_non_uniform_all_equal", non_uniform_all_equal_source, 0, required_extensions);
    error |= test<cl_uchar, AAN<cl_uchar, 2>, G, L>::run(device, context, queue, num_elements, "test_non_uniform_all_equal", non_uniform_all_equal_source, 0, required_extensions);
    error |= test<cl_short, AAN<cl_short, 2>, G, L>::run(device, context, queue, num_elements, "test_non_uniform_all_equal", non_uniform_all_equal_source, 0, required_extensions);
    error |= test<cl_ushort, AAN<cl_ushort, 2>, G, L>::run(device, context, queue, num_elements, "test_non_uniform_all_equal", non_uniform_all_equal_source, 0, required_extensions);
    error |= test<cl_int, AAN<cl_int, 2>, G, L>::run(device, context, queue, num_elements, "test_non_uniform_all_equal", non_uniform_all_equal_source, 0, required_extensions);
    error |= test<cl_uint, AAN<cl_uint, 2>, G, L>::run(device, context, queue, num_elements, "test_non_uniform_all_equal", non_uniform_all_equal_source, 0, required_extensions);
    error |= test<cl_long, AAN<cl_long, 2>, G, L>::run(device, context, queue, num_elements, "test_non_uniform_all_equal", non_uniform_all_equal_source, 0, required_extensions);
    error |= test<cl_ulong, AAN<cl_ulong, 2>, G, L>::run(device, context, queue, num_elements, "test_non_uniform_all_equal", non_uniform_all_equal_source, 0, required_extensions);
    error |= test<cl_float, AAN<cl_float, 2>, G, L>::run(device, context, queue, num_elements, "test_non_uniform_all_equal", non_uniform_all_equal_source, 0, required_extensions);
    error |= test<cl_double, AAN<cl_double, 2>, G, L>::run(device, context, queue, num_elements, "test_non_uniform_all_equal", non_uniform_all_equal_source, 0, required_extensions);

    return error;
}

