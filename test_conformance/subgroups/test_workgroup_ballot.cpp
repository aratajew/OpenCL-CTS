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
#include <bitset>


int
test_work_group_functions_ballot(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    int error = false;
    std::vector<std::string> required_extensions;
    required_extensions = { "cl_khr_subgroup_ballot" };
    error |= test<cl_int, BC<cl_int, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_int2, BC<cl_int2, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<subgroups::cl_int3, BC<subgroups::cl_int3, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_int4, BC<cl_int4, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_int8, BC<cl_int8, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_int16, BC<cl_int16, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_uint, BC<cl_uint, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_uint2, BC<cl_uint2, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<subgroups::cl_uint3, BC<subgroups::cl_uint3, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_uint4, BC<cl_uint4, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_uint8, BC<cl_uint8, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_uint16, BC<cl_uint16, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_long, BC<cl_long, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_long2, BC<cl_long2, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<subgroups::cl_long3, BC<subgroups::cl_long3, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_long4, BC<cl_long4, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_long8, BC<cl_long8, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_long16, BC<cl_long16, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_ulong, BC<cl_ulong, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_ulong2, BC<cl_ulong2, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<subgroups::cl_ulong3, BC<subgroups::cl_ulong3, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_ulong4, BC<cl_ulong4, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_ulong8, BC<cl_ulong8, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_ulong16, BC<cl_ulong16, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_double, BC<cl_double, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_double2, BC<cl_double2, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<subgroups::cl_double3, BC<subgroups::cl_double3, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_double4, BC<cl_double4, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_double8, BC<cl_double8, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_double16, BC<cl_double16, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<subgroups::cl_half, BC<subgroups::cl_half, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<subgroups::cl_half2, BC<subgroups::cl_half2, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<subgroups::cl_half3, BC<subgroups::cl_half3, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<subgroups::cl_half4, BC<subgroups::cl_half4, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<subgroups::cl_half8, BC<subgroups::cl_half8, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<subgroups::cl_half16, BC<subgroups::cl_half16, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_float, BC<cl_float, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_float2, BC<cl_float2, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<subgroups::cl_float3, BC<subgroups::cl_float3, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_float4, BC<cl_float4, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_float8, BC<cl_float8, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_float16, BC<cl_float16, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_short, BC<cl_short, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_short2, BC<cl_short2, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<subgroups::cl_short3, BC<subgroups::cl_short3, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_short4, BC<cl_short4, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_short8, BC<cl_short8, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_short16, BC<cl_short16, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_ushort, BC<cl_ushort, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_ushort2, BC<cl_ushort2, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<subgroups::cl_ushort3, BC<subgroups::cl_ushort3, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_ushort4, BC<cl_ushort4, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_ushort8, BC<cl_ushort8, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_ushort16, BC<cl_ushort16, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_char, BC<cl_char, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_char2, BC<cl_char2, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<subgroups::cl_char3, BC<subgroups::cl_char3, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_char4, BC<cl_char4, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_char8, BC<cl_char8, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_char16, BC<cl_char16, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_uchar, BC<cl_uchar, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_uchar2, BC<cl_uchar2, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<subgroups::cl_uchar3, BC<subgroups::cl_uchar3, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_uchar4, BC<cl_uchar4, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_uchar8, BC<cl_uchar8, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_uchar16, BC<cl_uchar16, 2>, G, L>::run(device, context, queue, num_elements, "test_bcast_non_uniform", bcast_non_uniform_source, 0, required_extensions);
    error |= test<cl_int, BC<cl_int, 1>, G, L>::run(device, context, queue, num_elements, "test_bcast_first", bcast_first_source, 0, required_extensions);
    error |= test<cl_uint, BC<cl_uint, 1>, G, L>::run(device, context, queue, num_elements, "test_bcast_first", bcast_first_source, 0, required_extensions);
    error |= test<cl_long, BC<cl_long, 1>, G, L>::run(device, context, queue, num_elements, "test_bcast_first", bcast_first_source, 0, required_extensions);
    error |= test<cl_ulong, BC<cl_ulong, 1>, G, L>::run(device, context, queue, num_elements, "test_bcast_first", bcast_first_source, 0, required_extensions);
    error |= test<cl_float, BC<cl_float, 1>, G, L>::run(device, context, queue, num_elements, "test_bcast_first", bcast_first_source, 0, required_extensions);
    error |= test<cl_short, BC<cl_short, 1>, G, L>::run(device, context, queue, num_elements, "test_bcast_first", bcast_first_source, 0, required_extensions);
    error |= test<cl_ushort, BC<cl_ushort, 1>, G, L>::run(device, context, queue, num_elements, "test_bcast_first", bcast_first_source, 0, required_extensions);
    error |= test<cl_char, BC<cl_char, 1>, G, L>::run(device, context, queue, num_elements, "test_bcast_first", bcast_first_source, 0, required_extensions);
    error |= test<cl_uchar, BC<cl_uchar, 1>, G, L>::run(device, context, queue, num_elements, "test_bcast_first", bcast_first_source, 0, required_extensions);
    error |= test<cl_double, BC<cl_double, 1>, G, L>::run(device, context, queue, num_elements, "test_bcast_first", bcast_first_source, 0, required_extensions);
    error |= test<subgroups::cl_half, BC<subgroups::cl_half, 1>, G, L>::run(device, context, queue, num_elements, "test_bcast_first", bcast_first_source, 0, required_extensions);
    error |= test<cl_uint4, SMASK<cl_uint4, 0>, G, L>::run(device, context, queue, num_elements, "test_get_sub_group_eq_mask", get_subgroup_eq_mask_source, 0, required_extensions);
    error |= test<cl_uint4, SMASK<cl_uint4, 1>, G, L>::run(device, context, queue, num_elements, "test_get_sub_group_ge_mask", get_subgroup_ge_mask_source, 0, required_extensions);
    error |= test<cl_uint4, SMASK<cl_uint4, 2>, G, L>::run(device, context, queue, num_elements, "test_get_sub_group_gt_mask", get_subgroup_gt_mask_source, 0, required_extensions);
    error |= test<cl_uint4, SMASK<cl_uint4, 3>, G, L>::run(device, context, queue, num_elements, "test_get_sub_group_le_mask", get_subgroup_le_mask_source, 0, required_extensions);
    error |= test<cl_uint4, SMASK<cl_uint4, 4>, G, L>::run(device, context, queue, num_elements, "test_get_sub_group_lt_mask", get_subgroup_lt_mask_source, 0, required_extensions);
    error |= test<cl_uint4, BALLOT<cl_uint4>, G, L>::run(device, context, queue, num_elements, "test_sub_group_ballot", ballot_source, 0, required_extensions);
    error |= test<cl_uint4, BALLOT2<cl_uint4, 0>, G, L>::run(device, context, queue, num_elements, "test_sub_group_inverse_ballot", inverse_ballot_source, 0, required_extensions);
    error |= test<cl_uint4, BALLOT2<cl_uint4, 1>, G, L>::run(device, context, queue, num_elements, "test_sub_group_ballot_bit_extract", ballot_bit_extract_source, 0, required_extensions);
    error |= test<cl_uint4, BALLOT3<cl_uint4, 0>, G, L>::run(device, context, queue, num_elements, "test_sub_group_ballot_bit_count", ballot_bit_count_source, 0, required_extensions);
    error |= test<cl_uint4, BALLOT3<cl_uint4, 1>, G, L>::run(device, context, queue, num_elements, "test_sub_group_ballot_inclusive_scan", ballot_inclusive_scan_source, 0, required_extensions);
    error |= test<cl_uint4, BALLOT3<cl_uint4, 2>, G, L>::run(device, context, queue, num_elements, "test_sub_group_ballot_exclusive_scan", ballot_exclusive_scan_source, 0, required_extensions);
    error |= test<cl_uint4, BALLOT3<cl_uint4, 3>, G, L>::run(device, context, queue, num_elements, "test_sub_group_ballot_find_lsb", ballot_find_lsb_source, 0, required_extensions);
    error |= test<cl_uint4, BALLOT3<cl_uint4, 4>, G, L>::run(device, context, queue, num_elements, "test_sub_group_ballot_find_msb", ballot_find_msb_source, 0, required_extensions);

    return error;
}

