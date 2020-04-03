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
#include "workgroup_common_templates.h"
#include "harness/typeWrappers.h"
#include <bitset>

static const char * shuffle_xor_source =
"__kernel void test_sub_group_shuffle_xor(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    Type x = in[gid];\n"
"    out[gid] = sub_group_shuffle_xor(x, xy[gid].z);"
"}\n";

static const char * shuffle_source =
"__kernel void test_sub_group_shuffle(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    Type x = in[gid];\n"
"    out[gid] = sub_group_shuffle(x, xy[gid].z);"
"}\n";

int
test_work_group_functions_shuffle(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    int error = false;
    std::vector<std::string> required_extensions;
    required_extensions = { "cl_khr_subgroup_shuffle" };
    error |= test<cl_int, SHF<cl_int, 0>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle", shuffle_source, 0, required_extensions);
    error |= test<cl_uint, SHF<cl_uint, 0>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle", shuffle_source, 0, required_extensions);
    error |= test<cl_long, SHF<cl_long, 0>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle", shuffle_source, 0, required_extensions);
    error |= test<cl_ulong, SHF<cl_ulong, 0>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle", shuffle_source, 0, required_extensions);
    error |= test<cl_short, SHF<cl_short, 0>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle", shuffle_source, 0, required_extensions);
    error |= test<cl_ushort, SHF<cl_ushort, 0>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle", shuffle_source, 0, required_extensions);
    error |= test<cl_char, SHF<cl_char, 0>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle", shuffle_source, 0, required_extensions);
    error |= test<cl_uchar, SHF<cl_uchar, 0>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle", shuffle_source, 0, required_extensions);
    error |= test<cl_float, SHF<cl_float, 0>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle", shuffle_source, 0, required_extensions);
    error |= test<cl_double, SHF<cl_double, 0>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle", shuffle_source, 0, required_extensions);
    error |= test<subgroups::cl_half, SHF<subgroups::cl_half, 0>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle", shuffle_source, 0, required_extensions);

    error |= test<cl_int, SHF<cl_int, 3>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle_xor", shuffle_xor_source, 0, required_extensions);
    error |= test<cl_uint, SHF<cl_uint, 3>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle_xor", shuffle_xor_source, 0, required_extensions);
    error |= test<cl_long, SHF<cl_long, 3>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle_xor", shuffle_xor_source, 0, required_extensions);
    error |= test<cl_ulong, SHF<cl_ulong, 3>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle_xor", shuffle_xor_source, 0, required_extensions);
    error |= test<cl_short, SHF<cl_short, 3>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle_xor", shuffle_xor_source, 0, required_extensions);
    error |= test<cl_ushort, SHF<cl_ushort, 3>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle_xor", shuffle_xor_source, 0, required_extensions);
    error |= test<cl_char, SHF<cl_char, 3>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle_xor", shuffle_xor_source, 0, required_extensions);
    error |= test<cl_uchar, SHF<cl_uchar, 3>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle_xor", shuffle_xor_source, 0, required_extensions);
    error |= test<cl_float, SHF<cl_float, 3>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle_xor", shuffle_xor_source, 0, required_extensions);
    error |= test<cl_double, SHF<cl_double, 3>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle_xor", shuffle_xor_source, 0, required_extensions);
    error |= test<subgroups::cl_half, SHF<subgroups::cl_half, 3>, G, L>::run(device, context, queue, num_elements, "test_sub_group_shuffle_xor", shuffle_xor_source, 0, required_extensions);

    return error;
}

