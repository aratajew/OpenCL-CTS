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
test_work_group_functions_extended_types(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    int error = false;
    std::vector<std::string> required_extensions;
    required_extensions = {"cl_khr_subgroup_extended_types" };
    error |= test<cl_double2, BC<cl_double2>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<subgroups::cl_double3, BC<subgroups::cl_double3>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_double4, BC<cl_double4>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_double8, BC<cl_double8>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_double16, BC<cl_double16>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<subgroups::cl_half2, BC<subgroups::cl_half2>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<subgroups::cl_half3, BC<subgroups::cl_half3>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<subgroups::cl_half4, BC<subgroups::cl_half4>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<subgroups::cl_half8, BC<subgroups::cl_half8>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<subgroups::cl_half16, BC<subgroups::cl_half16>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_int2, BC<cl_int2>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<subgroups::cl_int3, BC<subgroups::cl_int3>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_int4, BC<cl_int4>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_int8, BC<cl_int8>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_int16, BC<cl_int16>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_uint2, BC<cl_uint2>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<subgroups::cl_uint3, BC<subgroups::cl_uint3>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_uint4, BC<cl_uint4>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_uint8, BC<cl_uint8>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_uint16, BC<cl_uint16>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_long2, BC<cl_long2>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<subgroups::cl_long3, BC<subgroups::cl_long3>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_long4, BC<cl_long4>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_long8, BC<cl_long8>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_long16, BC<cl_long16>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_ulong2, BC<cl_ulong2>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<subgroups::cl_ulong3, BC<subgroups::cl_ulong3>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_ulong4, BC<cl_ulong4>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_ulong8, BC<cl_ulong8>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_ulong16, BC<cl_ulong16>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_float2, BC<cl_float2>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<subgroups::cl_float3, BC<subgroups::cl_float3>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_float4, BC<cl_float4>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_float8, BC<cl_float8>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_float16, BC<cl_float16>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_short, BC<cl_short>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_short2, BC<cl_short2>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<subgroups::cl_short3, BC<subgroups::cl_short3>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_short4, BC<cl_short4>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_short8, BC<cl_short8>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_short16, BC<cl_short16>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_ushort, BC<cl_ushort>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_ushort2, BC<cl_ushort2>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<subgroups::cl_ushort3, BC<subgroups::cl_ushort3>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_ushort4, BC<cl_ushort4>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_ushort8, BC<cl_ushort8>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_ushort16, BC<cl_ushort16>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_char, BC<cl_char>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_char2, BC<cl_char2>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<subgroups::cl_char3, BC<subgroups::cl_char3>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_char4, BC<cl_char4>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_char8, BC<cl_char8>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_char16, BC<cl_char16>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_uchar, BC<cl_uchar>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_uchar2, BC<cl_uchar2>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<subgroups::cl_uchar3, BC<subgroups::cl_uchar3>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_uchar4, BC<cl_uchar4>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_uchar8, BC<cl_uchar8>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_uchar16, BC<cl_uchar16>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);
    error |= test<cl_short, RED<cl_short, 0>, G, L>::run(device, context, queue, num_elements, "test_redadd", redadd_source, 0, required_extensions);
    error |= test<cl_ushort, RED<cl_ushort, 0>, G, L>::run(device, context, queue, num_elements, "test_redadd", redadd_source, 0, required_extensions);
    error |= test<cl_char, RED<cl_char, 0>, G, L>::run(device, context, queue, num_elements, "test_redadd", redadd_source, 0, required_extensions);
    error |= test<cl_uchar, RED<cl_uchar, 0>, G, L>::run(device, context, queue, num_elements, "test_redadd", redadd_source, 0, required_extensions);
    error |= test<cl_short, RED<cl_short, 1>, G, L>::run(device, context, queue, num_elements, "test_redmax", redmax_source, 0, required_extensions);
    error |= test<cl_ushort, RED<cl_ushort, 1>, G, L>::run(device, context, queue, num_elements, "test_redmax", redmax_source, 0, required_extensions);
    error |= test<cl_char, RED<cl_char, 1>, G, L>::run(device, context, queue, num_elements, "test_redmax", redmax_source, 0, required_extensions);
    error |= test<cl_uchar, RED<cl_uchar, 1>, G, L>::run(device, context, queue, num_elements, "test_redmax", redmax_source, 0, required_extensions);
    error |= test<cl_short, RED<cl_short, 2>, G, L>::run(device, context, queue, num_elements, "test_redmin", redmin_source, 0, required_extensions);
    error |= test<cl_ushort, RED<cl_ushort, 2>, G, L>::run(device, context, queue, num_elements, "test_redmin", redmin_source, 0, required_extensions);
    error |= test<cl_char, RED<cl_char, 2>, G, L>::run(device, context, queue, num_elements, "test_redmin", redmin_source, 0, required_extensions);
    error |= test<cl_uchar, RED<cl_uchar, 2>, G, L>::run(device, context, queue, num_elements, "test_redmin", redmin_source, 0, required_extensions);
    error |= test<cl_short, SCIN<cl_short, 0>, G, L>::run(device, context, queue, num_elements, "test_scinadd", scinadd_source, 0, required_extensions);
    error |= test<cl_ushort, SCIN<cl_ushort, 0>, G, L>::run(device, context, queue, num_elements, "test_scinadd", scinadd_source, 0, required_extensions);
    error |= test<cl_char, SCIN<cl_char, 0>, G, L>::run(device, context, queue, num_elements, "test_scinadd", scinadd_source, 0, required_extensions);
    error |= test<cl_uchar, SCIN<cl_uchar, 0>, G, L>::run(device, context, queue, num_elements, "test_scinadd", scinadd_source, 0, required_extensions);
    error |= test<cl_short, SCIN<cl_short, 1>, G, L>::run(device, context, queue, num_elements, "test_scinmax", scinmax_source, 0, required_extensions);
    error |= test<cl_ushort, SCIN<cl_ushort, 1>, G, L>::run(device, context, queue, num_elements, "test_scinmax", scinmax_source, 0, required_extensions);
    error |= test<cl_char, SCIN<cl_char, 1>, G, L>::run(device, context, queue, num_elements, "test_scinmax", scinmax_source, 0, required_extensions);
    error |= test<cl_uchar, SCIN<cl_uchar, 1>, G, L>::run(device, context, queue, num_elements, "test_scinmax", scinmax_source, 0, required_extensions);
    error |= test<cl_short, SCIN<cl_short, 2>, G, L>::run(device, context, queue, num_elements, "test_scinmin", scinmin_source, 0, required_extensions);
    error |= test<cl_ushort, SCIN<cl_ushort, 2>, G, L>::run(device, context, queue, num_elements, "test_scinmin", scinmin_source, 0, required_extensions);
    error |= test<cl_char, SCIN<cl_char, 2>, G, L>::run(device, context, queue, num_elements, "test_scinmin", scinmin_source, 0, required_extensions);
    error |= test<cl_uchar, SCIN<cl_uchar, 2>, G, L>::run(device, context, queue, num_elements, "test_scinmin", scinmin_source, 0, required_extensions);
    error |= test<cl_short, SCEX<cl_short, 0>, G, L>::run(device, context, queue, num_elements, "test_scexadd", scexadd_source, 0, required_extensions);
    error |= test<cl_ushort, SCEX<cl_ushort, 0>, G, L>::run(device, context, queue, num_elements, "test_scexadd", scexadd_source, 0, required_extensions);
    error |= test<cl_char, SCEX<cl_char, 0>, G, L>::run(device, context, queue, num_elements, "test_scexadd", scexadd_source, 0, required_extensions);
    error |= test<cl_uchar, SCEX<cl_uchar, 0>, G, L>::run(device, context, queue, num_elements, "test_scexadd", scexadd_source, 0, required_extensions);
    error |= test<cl_short, SCEX<cl_short, 1>, G, L>::run(device, context, queue, num_elements, "test_scexmax", scexmax_source, 0, required_extensions);
    error |= test<cl_ushort, SCEX<cl_ushort, 1>, G, L>::run(device, context, queue, num_elements, "test_scexmax", scexmax_source, 0, required_extensions);
    error |= test<cl_char, SCEX<cl_char, 1>, G, L>::run(device, context, queue, num_elements, "test_scexmax", scexmax_source, 0, required_extensions);
    error |= test<cl_uchar, SCEX<cl_uchar, 1>, G, L>::run(device, context, queue, num_elements, "test_scexmax", scexmax_source, 0, required_extensions);
    error |= test<cl_short, SCEX<cl_short, 2>, G, L>::run(device, context, queue, num_elements, "test_scexmin", scexmin_source, 0, required_extensions);
    error |= test<cl_ushort, SCEX<cl_ushort, 2>, G, L>::run(device, context, queue, num_elements, "test_scexmin", scexmin_source, 0, required_extensions);
    error |= test<cl_char, SCEX<cl_char, 2>, G, L>::run(device, context, queue, num_elements, "test_scexmin", scexmin_source, 0, required_extensions);
    error |= test<cl_uchar, SCEX<cl_uchar, 2>, G, L>::run(device, context, queue, num_elements, "test_scexmin", scexmin_source, 0, required_extensions);

      //*************************************************************************

    return error;
}

