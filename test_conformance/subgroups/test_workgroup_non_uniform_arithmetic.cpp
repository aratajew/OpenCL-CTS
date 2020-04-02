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
test_work_group_functions_non_uniform_arithmetic(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    int error = false;
    std::vector<std::string> required_extensions;
    required_extensions = { "cl_khr_subgroup_non_uniform_arithmetic" };
    error |= test<cl_int, SCIN_NU<cl_int, 0>, G, L>::run(device, context, queue, num_elements, "test_scinadd_non_uniform", scinadd_non_uniform_source, 0, required_extensions);
    error |= test<cl_uint, SCIN_NU<cl_uint, 0>, G, L>::run(device, context, queue, num_elements, "test_scinadd_non_uniform", scinadd_non_uniform_source, 0, required_extensions);
    error |= test<cl_long, SCIN_NU<cl_long, 0>, G, L>::run(device, context, queue, num_elements, "test_scinadd_non_uniform", scinadd_non_uniform_source, 0, required_extensions);
    error |= test<cl_ulong, SCIN_NU<cl_ulong, 0>, G, L>::run(device, context, queue, num_elements, "test_scinadd_non_uniform", scinadd_non_uniform_source, 0, required_extensions);
    error |= test<cl_short, SCIN_NU<cl_short, 0>, G, L>::run(device, context, queue, num_elements, "test_scinadd_non_uniform", scinadd_non_uniform_source, 0, required_extensions);
    error |= test<cl_ushort, SCIN_NU<cl_ushort, 0>, G, L>::run(device, context, queue, num_elements, "test_scinadd_non_uniform", scinadd_non_uniform_source, 0, required_extensions);
    error |= test<cl_char, SCIN_NU<cl_char, 0>, G, L>::run(device, context, queue, num_elements, "test_scinadd_non_uniform", scinadd_non_uniform_source, 0, required_extensions);
    error |= test<cl_uchar, SCIN_NU<cl_uchar, 0>, G, L>::run(device, context, queue, num_elements, "test_scinadd_non_uniform", scinadd_non_uniform_source, 0, required_extensions);
    error |= test<cl_float, SCIN_NU<cl_float, 0>, G, L>::run(device, context, queue, num_elements, "test_scinadd_non_uniform", scinadd_non_uniform_source, 0, required_extensions);
    error |= test<cl_double, SCIN_NU<cl_double, 0>, G, L>::run(device, context, queue, num_elements, "test_scinadd_non_uniform", scinadd_non_uniform_source, 0, required_extensions);
    //error |= test<subgroups::cl_half, SCIN_NU<subgroups::cl_half, 0>, G, L>::run(device, context, queue, num_elements, "test_scinadd_non_uniform", scinadd_non_uniform_source, 0, required_extensions);

    error |= test<cl_int, SCIN_NU<cl_int, 1>, G, L>::run(device, context, queue, num_elements, "test_scinmax_non_uniform", scinmax_non_uniform_source, 0, required_extensions);
    error |= test<cl_uint, SCIN_NU<cl_uint, 1>, G, L>::run(device, context, queue, num_elements, "test_scinmax_non_uniform", scinmax_non_uniform_source, 0, required_extensions);
    error |= test<cl_long, SCIN_NU<cl_long, 1>, G, L>::run(device, context, queue, num_elements, "test_scinmax_non_uniform", scinmax_non_uniform_source, 0, required_extensions);
    error |= test<cl_ulong, SCIN_NU<cl_ulong, 1>, G, L>::run(device, context, queue, num_elements, "test_scinmax_non_uniform", scinmax_non_uniform_source, 0, required_extensions);
    error |= test<cl_short, SCIN_NU<cl_short, 1>, G, L>::run(device, context, queue, num_elements, "test_scinmax_non_uniform", scinmax_non_uniform_source, 0, required_extensions);
    error |= test<cl_ushort, SCIN_NU<cl_ushort, 1>, G, L>::run(device, context, queue, num_elements, "test_scinmax_non_uniform", scinmax_non_uniform_source, 0, required_extensions);
    error |= test<cl_char, SCIN_NU<cl_char, 1>, G, L>::run(device, context, queue, num_elements, "test_scinmax_non_uniform", scinmax_non_uniform_source, 0, required_extensions);
    error |= test<cl_uchar, SCIN_NU<cl_uchar, 1>, G, L>::run(device, context, queue, num_elements, "test_scinmax_non_uniform", scinmax_non_uniform_source, 0, required_extensions);
    error |= test<cl_float, SCIN_NU<cl_float, 1>, G, L>::run(device, context, queue, num_elements, "test_scinmax_non_uniform", scinmax_non_uniform_source, 0, required_extensions);
    error |= test<cl_double, SCIN_NU<cl_double, 1>, G, L>::run(device, context, queue, num_elements, "test_scinmax_non_uniform", scinmax_non_uniform_source, 0, required_extensions);
    //error |= test<subgroups::cl_half, SCIN_NU<subgroups::cl_half, 1>, G, L>::run(device, context, queue, num_elements, "test_scinmax_non_uniform", scinmax_non_uniform_source, 0, required_extensions);

    error |= test<cl_int, SCIN_NU<cl_int, 2>, G, L>::run(device, context, queue, num_elements, "test_scinmin_non_uniform", scinmin_non_uniform_source, 0, required_extensions);
    error |= test<cl_uint, SCIN_NU<cl_uint, 2>, G, L>::run(device, context, queue, num_elements, "test_scinmin_non_uniform", scinmin_non_uniform_source, 0, required_extensions);
    error |= test<cl_long, SCIN_NU<cl_long, 2>, G, L>::run(device, context, queue, num_elements, "test_scinmin_non_uniform", scinmin_non_uniform_source, 0, required_extensions);
    error |= test<cl_ulong, SCIN_NU<cl_ulong, 2>, G, L>::run(device, context, queue, num_elements, "test_scinmin_non_uniform", scinmin_non_uniform_source, 0, required_extensions);
    error |= test<cl_short, SCIN_NU<cl_short, 2>, G, L>::run(device, context, queue, num_elements, "test_scinmin_non_uniform", scinmin_non_uniform_source, 0, required_extensions);
    error |= test<cl_ushort, SCIN_NU<cl_ushort, 2>, G, L>::run(device, context, queue, num_elements, "test_scinmin_non_uniform", scinmin_non_uniform_source, 0, required_extensions);
    error |= test<cl_char, SCIN_NU<cl_char, 2>, G, L>::run(device, context, queue, num_elements, "test_scinmin_non_uniform", scinmin_non_uniform_source, 0, required_extensions);
    error |= test<cl_uchar, SCIN_NU<cl_uchar, 2>, G, L>::run(device, context, queue, num_elements, "test_scinmin_non_uniform", scinmin_non_uniform_source, 0, required_extensions);
    error |= test<cl_float, SCIN_NU<cl_float, 2>, G, L>::run(device, context, queue, num_elements, "test_scinmin_non_uniform", scinmin_non_uniform_source, 0, required_extensions);
    error |= test<cl_double, SCIN_NU<cl_double, 2>, G, L>::run(device, context, queue, num_elements, "test_scinmin_non_uniform", scinmin_non_uniform_source, 0, required_extensions);
    //error |= test<subgroups::cl_half, SCIN_NU<subgroups::cl_half, 2>, G, L>::run(device, context, queue, num_elements, "test_scinmin_non_uniform", scinmin_non_uniform_source, 0, required_extensions);

    error |= test<cl_int, SCIN_NU<cl_int, 3>, G, L>::run(device, context, queue, num_elements, "test_scinmul_non_uniform", scinmul_non_uniform_source, 0, required_extensions);
    error |= test<cl_uint, SCIN_NU<cl_uint, 3>, G, L>::run(device, context, queue, num_elements, "test_scinmul_non_uniform", scinmul_non_uniform_source, 0, required_extensions);
    error |= test<cl_long, SCIN_NU<cl_long, 3>, G, L>::run(device, context, queue, num_elements, "test_scinmul_non_uniform", scinmul_non_uniform_source, 0, required_extensions);
    error |= test<cl_ulong, SCIN_NU<cl_ulong, 3>, G, L>::run(device, context, queue, num_elements, "test_scinmul_non_uniform", scinmul_non_uniform_source, 0, required_extensions);
    error |= test<cl_short, SCIN_NU<cl_short, 3>, G, L>::run(device, context, queue, num_elements, "test_scinmul_non_uniform", scinmul_non_uniform_source, 0, required_extensions);
    error |= test<cl_ushort, SCIN_NU<cl_ushort, 3>, G, L>::run(device, context, queue, num_elements, "test_scinmul_non_uniform", scinmul_non_uniform_source, 0, required_extensions);
    error |= test<cl_char, SCIN_NU<cl_char, 3>, G, L>::run(device, context, queue, num_elements, "test_scinmul_non_uniform", scinmul_non_uniform_source, 0, required_extensions);
    error |= test<cl_uchar, SCIN_NU<cl_uchar, 3>, G, L>::run(device, context, queue, num_elements, "test_scinmul_non_uniform", scinmul_non_uniform_source, 0, required_extensions);
    error |= test<cl_float, SCIN_NU<cl_float, 3>, G, L>::run(device, context, queue, num_elements, "test_scinmul_non_uniform", scinmul_non_uniform_source, 0, required_extensions);
    error |= test<cl_double, SCIN_NU<cl_double, 3>, G, L>::run(device, context, queue, num_elements, "test_scinmul_non_uniform", scinmul_non_uniform_source, 0, required_extensions);
    //error |= test<subgroups::cl_half, SCIN_NU<subgroups::cl_half, 3>, G, L>::run(device, context, queue, num_elements, "test_scinmul_non_uniform", scinmul_non_uniform_source, 0, required_extensions);

    error |= test<cl_int, SCIN_NU<cl_int, 4>, G, L>::run(device, context, queue, num_elements, "test_scinand_non_uniform", scinand_non_uniform_source, 0, required_extensions);
    error |= test<cl_uint, SCIN_NU<cl_uint, 4>, G, L>::run(device, context, queue, num_elements, "test_scinand_non_uniform", scinand_non_uniform_source, 0, required_extensions);
    error |= test<cl_long, SCIN_NU<cl_long, 4>, G, L>::run(device, context, queue, num_elements, "test_scinand_non_uniform", scinand_non_uniform_source, 0, required_extensions);
    error |= test<cl_ulong, SCIN_NU<cl_ulong, 4>, G, L>::run(device, context, queue, num_elements, "test_scinand_non_uniform", scinand_non_uniform_source, 0, required_extensions);
    error |= test<cl_short, SCIN_NU<cl_short, 4>, G, L>::run(device, context, queue, num_elements, "test_scinand_non_uniform", scinand_non_uniform_source, 0, required_extensions);
    error |= test<cl_ushort, SCIN_NU<cl_ushort, 4>, G, L>::run(device, context, queue, num_elements, "test_scinand_non_uniform", scinand_non_uniform_source, 0, required_extensions);
    error |= test<cl_char, SCIN_NU<cl_char, 4>, G, L>::run(device, context, queue, num_elements, "test_scinand_non_uniform", scinand_non_uniform_source, 0, required_extensions);
    error |= test<cl_uchar, SCIN_NU<cl_uchar, 4>, G, L>::run(device, context, queue, num_elements, "test_scinand_non_uniform", scinand_non_uniform_source, 0, required_extensions);

    error |= test<cl_int, SCIN_NU<cl_int, 5>, G, L>::run(device, context, queue, num_elements, "test_scinor_non_uniform", scinor_non_uniform_source, 0, required_extensions);
    error |= test<cl_uint, SCIN_NU<cl_uint, 5>, G, L>::run(device, context, queue, num_elements, "test_scinor_non_uniform", scinor_non_uniform_source, 0, required_extensions);
    error |= test<cl_long, SCIN_NU<cl_long, 5>, G, L>::run(device, context, queue, num_elements, "test_scinor_non_uniform", scinor_non_uniform_source, 0, required_extensions);
    error |= test<cl_ulong, SCIN_NU<cl_ulong, 5>, G, L>::run(device, context, queue, num_elements, "test_scinor_non_uniform", scinor_non_uniform_source, 0, required_extensions);
    error |= test<cl_short, SCIN_NU<cl_short, 5>, G, L>::run(device, context, queue, num_elements, "test_scinor_non_uniform", scinor_non_uniform_source, 0, required_extensions);
    error |= test<cl_ushort, SCIN_NU<cl_ushort, 5>, G, L>::run(device, context, queue, num_elements, "test_scinor_non_uniform", scinor_non_uniform_source, 0, required_extensions);
    error |= test<cl_char, SCIN_NU<cl_char, 5>, G, L>::run(device, context, queue, num_elements, "test_scinor_non_uniform", scinor_non_uniform_source, 0, required_extensions);
    error |= test<cl_uchar, SCIN_NU<cl_uchar, 5>, G, L>::run(device, context, queue, num_elements, "test_scinor_non_uniform", scinor_non_uniform_source, 0, required_extensions);

    error |= test<cl_int, SCIN_NU<cl_int, 6>, G, L>::run(device, context, queue, num_elements, "test_scinxor_non_uniform", scinxor_non_uniform_source, 0, required_extensions);
    error |= test<cl_uint, SCIN_NU<cl_uint, 6>, G, L>::run(device, context, queue, num_elements, "test_scinxor_non_uniform", scinxor_non_uniform_source, 0, required_extensions);
    error |= test<cl_long, SCIN_NU<cl_long, 6>, G, L>::run(device, context, queue, num_elements, "test_scinxor_non_uniform", scinxor_non_uniform_source, 0, required_extensions);
    error |= test<cl_ulong, SCIN_NU<cl_ulong, 6>, G, L>::run(device, context, queue, num_elements, "test_scinxor_non_uniform", scinxor_non_uniform_source, 0, required_extensions);
    error |= test<cl_short, SCIN_NU<cl_short, 6>, G, L>::run(device, context, queue, num_elements, "test_scinxor_non_uniform", scinxor_non_uniform_source, 0, required_extensions);
    error |= test<cl_ushort, SCIN_NU<cl_ushort, 6>, G, L>::run(device, context, queue, num_elements, "test_scinxor_non_uniform", scinxor_non_uniform_source, 0, required_extensions);
    error |= test<cl_char, SCIN_NU<cl_char, 6>, G, L>::run(device, context, queue, num_elements, "test_scinxor_non_uniform", scinxor_non_uniform_source, 0, required_extensions);
    error |= test<cl_uchar, SCIN_NU<cl_uchar, 6>, G, L>::run(device, context, queue, num_elements, "test_scinxor_non_uniform", scinxor_non_uniform_source, 0, required_extensions);

    error |= test<cl_int, SCIN_NU<cl_int, 7>, G, L>::run(device, context, queue, num_elements, "test_scinand_non_uniform_logical", scinand_non_uniform_logical_source, 0, required_extensions);
    error |= test<cl_int, SCIN_NU<cl_int, 8>, G, L>::run(device, context, queue, num_elements, "test_scinor_non_uniform_logical", scinor_non_uniform_logical_source, 0, required_extensions);
    error |= test<cl_int, SCIN_NU<cl_int, 9>, G, L>::run(device, context, queue, num_elements, "test_scinxor_non_uniform_logical", scinxor_non_uniform_logical_source, 0, required_extensions);

    error |= test<cl_int, SCEX_NU<cl_int, 0>, G, L>::run(device, context, queue, num_elements, "test_scexadd_non_uniform", scexadd_non_uniform_source, 0, required_extensions);
    error |= test<cl_uint, SCEX_NU<cl_uint, 0>, G, L>::run(device, context, queue, num_elements, "test_scexadd_non_uniform", scexadd_non_uniform_source, 0, required_extensions);
    error |= test<cl_long, SCEX_NU<cl_long, 0>, G, L>::run(device, context, queue, num_elements, "test_scexadd_non_uniform", scexadd_non_uniform_source, 0, required_extensions);
    error |= test<cl_ulong, SCEX_NU<cl_ulong, 0>, G, L>::run(device, context, queue, num_elements, "test_scexadd_non_uniform", scexadd_non_uniform_source, 0, required_extensions);
    error |= test<cl_short, SCEX_NU<cl_short, 0>, G, L>::run(device, context, queue, num_elements, "test_scexadd_non_uniform", scexadd_non_uniform_source, 0, required_extensions);
    error |= test<cl_ushort, SCEX_NU<cl_ushort, 0>, G, L>::run(device, context, queue, num_elements, "test_scexadd_non_uniform", scexadd_non_uniform_source, 0, required_extensions);
    error |= test<cl_char, SCEX_NU<cl_char, 0>, G, L>::run(device, context, queue, num_elements, "test_scexadd_non_uniform", scexadd_non_uniform_source, 0, required_extensions);
    error |= test<cl_uchar, SCEX_NU<cl_uchar, 0>, G, L>::run(device, context, queue, num_elements, "test_scexadd_non_uniform", scexadd_non_uniform_source, 0, required_extensions);
    error |= test<cl_float, SCEX_NU<cl_float, 0>, G, L>::run(device, context, queue, num_elements, "test_scexadd_non_uniform", scexadd_non_uniform_source, 0, required_extensions);
    error |= test<cl_double, SCEX_NU<cl_double, 0>, G, L>::run(device, context, queue, num_elements, "test_scexadd_non_uniform", scexadd_non_uniform_source, 0, required_extensions);
    //error |= test<subgroups::cl_half, SCEX_NU<subgroups::cl_half, 0>, G, L>::run(device, context, queue, num_elements, "test_scexadd_non_uniform", scexadd_non_uniform_source, 0, required_extensions);

    error |= test<cl_int, SCEX_NU<cl_int, 1>, G, L>::run(device, context, queue, num_elements, "test_scexmax_non_uniform", scexmax_non_uniform_source, 0, required_extensions);
    error |= test<cl_uint, SCEX_NU<cl_uint, 1>, G, L>::run(device, context, queue, num_elements, "test_scexmax_non_uniform", scexmax_non_uniform_source, 0, required_extensions);
    error |= test<cl_long, SCEX_NU<cl_long, 1>, G, L>::run(device, context, queue, num_elements, "test_scexmax_non_uniform", scexmax_non_uniform_source, 0, required_extensions);
    error |= test<cl_ulong, SCEX_NU<cl_ulong, 1>, G, L>::run(device, context, queue, num_elements, "test_scexmax_non_uniform", scexmax_non_uniform_source, 0, required_extensions);
    error |= test<cl_short, SCEX_NU<cl_short, 1>, G, L>::run(device, context, queue, num_elements, "test_scexmax_non_uniform", scexmax_non_uniform_source, 0, required_extensions);
    error |= test<cl_ushort, SCEX_NU<cl_ushort, 1>, G, L>::run(device, context, queue, num_elements, "test_scexmax_non_uniform", scexmax_non_uniform_source, 0, required_extensions);
    error |= test<cl_char, SCEX_NU<cl_char, 1>, G, L>::run(device, context, queue, num_elements, "test_scexmax_non_uniform", scexmax_non_uniform_source, 0, required_extensions);
    error |= test<cl_uchar, SCEX_NU<cl_uchar, 1>, G, L>::run(device, context, queue, num_elements, "test_scexmax_non_uniform", scexmax_non_uniform_source, 0, required_extensions);
    error |= test<cl_float, SCEX_NU<cl_float, 1>, G, L>::run(device, context, queue, num_elements, "test_scexmax_non_uniform", scexmax_non_uniform_source, 0, required_extensions);
    error |= test<cl_double, SCEX_NU<cl_double, 1>, G, L>::run(device, context, queue, num_elements, "test_scexmax_non_uniform", scexmax_non_uniform_source, 0, required_extensions);
   //error |= test<subgroups::cl_half, SCEX_NU<subgroups::cl_half, 1>, G, L>::run(device, context, queue, num_elements, "test_scexmax_non_uniform", scexmax_non_uniform_source, 0, required_extensions);

    error |= test<cl_int, SCEX_NU<cl_int, 2>, G, L>::run(device, context, queue, num_elements, "test_scexmin_non_uniform", scexmin_non_uniform_source, 0, required_extensions);
    error |= test<cl_uint, SCEX_NU<cl_uint, 2>, G, L>::run(device, context, queue, num_elements, "test_scexmin_non_uniform", scexmin_non_uniform_source, 0, required_extensions);
    error |= test<cl_long, SCEX_NU<cl_long, 2>, G, L>::run(device, context, queue, num_elements, "test_scexmin_non_uniform", scexmin_non_uniform_source, 0, required_extensions);
    error |= test<cl_ulong, SCEX_NU<cl_ulong, 2>, G, L>::run(device, context, queue, num_elements, "test_scexmin_non_uniform", scexmin_non_uniform_source, 0, required_extensions);
    error |= test<cl_short, SCEX_NU<cl_short, 2>, G, L>::run(device, context, queue, num_elements, "test_scexmin_non_uniform", scexmin_non_uniform_source, 0, required_extensions);
    error |= test<cl_ushort, SCEX_NU<cl_ushort, 2>, G, L>::run(device, context, queue, num_elements, "test_scexmin_non_uniform", scexmin_non_uniform_source, 0, required_extensions);
    error |= test<cl_char, SCEX_NU<cl_char, 2>, G, L>::run(device, context, queue, num_elements, "test_scexmin_non_uniform", scexmin_non_uniform_source, 0, required_extensions);
    error |= test<cl_uchar, SCEX_NU<cl_uchar, 2>, G, L>::run(device, context, queue, num_elements, "test_scexmin_non_uniform", scexmin_non_uniform_source, 0, required_extensions);
    error |= test<cl_float, SCEX_NU<cl_float, 2>, G, L>::run(device, context, queue, num_elements, "test_scexmin_non_uniform", scexmin_non_uniform_source, 0, required_extensions);
    error |= test<cl_double, SCEX_NU<cl_double, 2>, G, L>::run(device, context, queue, num_elements, "test_scexmin_non_uniform", scexmin_non_uniform_source, 0, required_extensions);
   //error |= test<subgroups::cl_half, SCEX_NU<subgroups::cl_half, 2>, G, L>::run(device, context, queue, num_elements, "test_scexmin_non_uniform", scexmin_non_uniform_source, 0, required_extensions);

    error |= test<cl_int, SCEX_NU<cl_int, 3>, G, L>::run(device, context, queue, num_elements, "test_scexmul_non_uniform", scexmul_non_uniform_source, 0, required_extensions);
    error |= test<cl_uint, SCEX_NU<cl_uint, 3>, G, L>::run(device, context, queue, num_elements, "test_scexmul_non_uniform", scexmul_non_uniform_source, 0, required_extensions);
    error |= test<cl_long, SCEX_NU<cl_long, 3>, G, L>::run(device, context, queue, num_elements, "test_scexmul_non_uniform", scexmul_non_uniform_source, 0, required_extensions);
    error |= test<cl_ulong, SCEX_NU<cl_ulong, 3>, G, L>::run(device, context, queue, num_elements, "test_scexmul_non_uniform", scexmul_non_uniform_source, 0, required_extensions);
    error |= test<cl_short, SCEX_NU<cl_short, 3>, G, L>::run(device, context, queue, num_elements, "test_scexmul_non_uniform", scexmul_non_uniform_source, 0, required_extensions);
    error |= test<cl_ushort, SCEX_NU<cl_ushort, 3>, G, L>::run(device, context, queue, num_elements, "test_scexmul_non_uniform", scexmul_non_uniform_source, 0, required_extensions);
    error |= test<cl_char, SCEX_NU<cl_char, 3>, G, L>::run(device, context, queue, num_elements, "test_scexmul_non_uniform", scexmul_non_uniform_source, 0, required_extensions);
    error |= test<cl_uchar, SCEX_NU<cl_uchar, 3>, G, L>::run(device, context, queue, num_elements, "test_scexmul_non_uniform", scexmul_non_uniform_source, 0, required_extensions);
    error |= test<cl_float, SCEX_NU<cl_float, 3>, G, L>::run(device, context, queue, num_elements, "test_scexmul_non_uniform", scexmul_non_uniform_source, 0, required_extensions);
    error |= test<cl_double, SCEX_NU<cl_double, 3>, G, L>::run(device, context, queue, num_elements, "test_scexmul_non_uniform", scexmul_non_uniform_source, 0, required_extensions);
   //error |= test<subgroups::cl_half, SCEX_NU<subgroups::cl_half, 3>, G, L>::run(device, context, queue, num_elements, "test_scexmul_non_uniform", scexmul_non_uniform_source, 0, required_extensions);

    error |= test<cl_int, SCEX_NU<cl_int, 4>, G, L>::run(device, context, queue, num_elements, "test_scexand_non_uniform", scexand_non_uniform_source, 0, required_extensions);
    error |= test<cl_uint, SCEX_NU<cl_uint, 4>, G, L>::run(device, context, queue, num_elements, "test_scexand_non_uniform", scexand_non_uniform_source, 0, required_extensions);
    error |= test<cl_long, SCEX_NU<cl_long, 4>, G, L>::run(device, context, queue, num_elements, "test_scexand_non_uniform", scexand_non_uniform_source, 0, required_extensions);
    error |= test<cl_ulong, SCEX_NU<cl_ulong, 4>, G, L>::run(device, context, queue, num_elements, "test_scexand_non_uniform", scexand_non_uniform_source, 0, required_extensions);
    error |= test<cl_short, SCEX_NU<cl_short, 4>, G, L>::run(device, context, queue, num_elements, "test_scexand_non_uniform", scexand_non_uniform_source, 0, required_extensions);
    error |= test<cl_ushort, SCEX_NU<cl_ushort, 4>, G, L>::run(device, context, queue, num_elements, "test_scexand_non_uniform", scexand_non_uniform_source, 0, required_extensions);
    error |= test<cl_char, SCEX_NU<cl_char, 4>, G, L>::run(device, context, queue, num_elements, "test_scexand_non_uniform", scexand_non_uniform_source, 0, required_extensions);
    error |= test<cl_uchar, SCEX_NU<cl_uchar, 4>, G, L>::run(device, context, queue, num_elements, "test_scexand_non_uniform", scexand_non_uniform_source, 0, required_extensions);

    error |= test<cl_int, SCEX_NU<cl_int, 5>, G, L>::run(device, context, queue, num_elements, "test_scexor_non_uniform", scexor_non_uniform_source, 0, required_extensions);
    error |= test<cl_uint, SCEX_NU<cl_uint, 5>, G, L>::run(device, context, queue, num_elements, "test_scexor_non_uniform", scexor_non_uniform_source, 0, required_extensions);
    error |= test<cl_long, SCEX_NU<cl_long, 5>, G, L>::run(device, context, queue, num_elements, "test_scexor_non_uniform", scexor_non_uniform_source, 0, required_extensions);
    error |= test<cl_ulong, SCEX_NU<cl_ulong, 5>, G, L>::run(device, context, queue, num_elements, "test_scexor_non_uniform", scexor_non_uniform_source, 0, required_extensions);
    error |= test<cl_short, SCEX_NU<cl_short, 5>, G, L>::run(device, context, queue, num_elements, "test_scexor_non_uniform", scexor_non_uniform_source, 0, required_extensions);
    error |= test<cl_ushort, SCEX_NU<cl_ushort, 5>, G, L>::run(device, context, queue, num_elements, "test_scexor_non_uniform", scexor_non_uniform_source, 0, required_extensions);
    error |= test<cl_char, SCEX_NU<cl_char, 5>, G, L>::run(device, context, queue, num_elements, "test_scexor_non_uniform", scexor_non_uniform_source, 0, required_extensions);
    error |= test<cl_uchar, SCEX_NU<cl_uchar, 5>, G, L>::run(device, context, queue, num_elements, "test_scexor_non_uniform", scexor_non_uniform_source, 0, required_extensions);

    error |= test<cl_int, SCEX_NU<cl_int, 6>, G, L>::run(device, context, queue, num_elements, "test_scexxor_non_uniform", scexxor_non_uniform_source, 0, required_extensions);
    error |= test<cl_uint, SCEX_NU<cl_uint, 6>, G, L>::run(device, context, queue, num_elements, "test_scexxor_non_uniform", scexxor_non_uniform_source, 0, required_extensions);
    error |= test<cl_long, SCEX_NU<cl_long, 6>, G, L>::run(device, context, queue, num_elements, "test_scexxor_non_uniform", scexxor_non_uniform_source, 0, required_extensions);
    error |= test<cl_ulong, SCEX_NU<cl_ulong, 6>, G, L>::run(device, context, queue, num_elements, "test_scexxor_non_uniform", scexxor_non_uniform_source, 0, required_extensions);
    error |= test<cl_short, SCEX_NU<cl_short, 6>, G, L>::run(device, context, queue, num_elements, "test_scexxor_non_uniform", scexxor_non_uniform_source, 0, required_extensions);
    error |= test<cl_ushort, SCEX_NU<cl_ushort, 6>, G, L>::run(device, context, queue, num_elements, "test_scexxor_non_uniform", scexxor_non_uniform_source, 0, required_extensions);
    error |= test<cl_char, SCEX_NU<cl_char, 6>, G, L>::run(device, context, queue, num_elements, "test_scexxor_non_uniform", scexxor_non_uniform_source, 0, required_extensions);
    error |= test<cl_uchar, SCEX_NU<cl_uchar, 6>, G, L>::run(device, context, queue, num_elements, "test_scexxor_non_uniform", scexxor_non_uniform_source, 0, required_extensions);

    error |= test<cl_int, SCEX_NU<cl_int, 7>, G, L>::run(device, context, queue, num_elements, "test_scexand_non_uniform_logical", scexand_non_uniform_logical_source, 0, required_extensions);
    error |= test<cl_int, SCEX_NU<cl_int, 8>, G, L>::run(device, context, queue, num_elements, "test_scexor_non_uniform_logical", scexor_non_uniform_logical_source, 0, required_extensions);
    error |= test<cl_int, SCEX_NU<cl_int, 9>, G, L>::run(device, context, queue, num_elements, "test_scexxor_non_uniform_logical", scexxor_non_uniform_logical_source, 0, required_extensions);

    error |= test<cl_int, RED_NU<cl_int, 0>, G, L>::run(device, context, queue, num_elements, "test_redadd_non_uniform", redadd_non_uniform_source, 0, required_extensions);
    error |= test<cl_uint, RED_NU<cl_uint, 0>, G, L>::run(device, context, queue, num_elements, "test_redadd_non_uniform", redadd_non_uniform_source, 0, required_extensions);
    error |= test<cl_long, RED_NU<cl_long, 0>, G, L>::run(device, context, queue, num_elements, "test_redadd_non_uniform", redadd_non_uniform_source, 0, required_extensions);
    error |= test<cl_ulong, RED_NU<cl_ulong, 0>, G, L>::run(device, context, queue, num_elements, "test_redadd_non_uniform", redadd_non_uniform_source, 0, required_extensions);
    error |= test<cl_short, RED_NU<cl_short, 0>, G, L>::run(device, context, queue, num_elements, "test_redadd_non_uniform", redadd_non_uniform_source, 0, required_extensions);
    error |= test<cl_ushort, RED_NU<cl_ushort, 0>, G, L>::run(device, context, queue, num_elements, "test_redadd_non_uniform", redadd_non_uniform_source, 0, required_extensions);
    error |= test<cl_char, RED_NU<cl_char, 0>, G, L>::run(device, context, queue, num_elements, "test_redadd_non_uniform", redadd_non_uniform_source, 0, required_extensions);
    error |= test<cl_uchar, RED_NU<cl_uchar, 0>, G, L>::run(device, context, queue, num_elements, "test_redadd_non_uniform", redadd_non_uniform_source, 0, required_extensions);
    error |= test<cl_float, RED_NU<cl_float, 0>, G, L>::run(device, context, queue, num_elements, "test_redadd_non_uniform", redadd_non_uniform_source, 0, required_extensions);
    error |= test<cl_double, RED_NU<cl_double, 0>, G, L>::run(device, context, queue, num_elements, "test_redadd_non_uniform", redadd_non_uniform_source, 0, required_extensions);
    //error |= test<subgroups::cl_half, RED_NU<subgroups::cl_half, 0>, G, L>::run(device, context, queue, num_elements, "test_redadd_non_uniform", redadd_non_uniform_source, 0, required_extensions);

    error |= test<cl_int, RED_NU<cl_int, 1>, G, L>::run(device, context, queue, num_elements, "test_redmax_non_uniform", redmax_non_uniform_source, 0, required_extensions);
    error |= test<cl_uint, RED_NU<cl_uint, 1>, G, L>::run(device, context, queue, num_elements, "test_redmax_non_uniform", redmax_non_uniform_source, 0, required_extensions);
    error |= test<cl_long, RED_NU<cl_long, 1>, G, L>::run(device, context, queue, num_elements, "test_redmax_non_uniform", redmax_non_uniform_source, 0, required_extensions);
    error |= test<cl_ulong, RED_NU<cl_ulong, 1>, G, L>::run(device, context, queue, num_elements, "test_redmax_non_uniform", redmax_non_uniform_source, 0, required_extensions);
    error |= test<cl_short, RED_NU<cl_short, 1>, G, L>::run(device, context, queue, num_elements, "test_redmax_non_uniform", redmax_non_uniform_source, 0, required_extensions);
    error |= test<cl_ushort, RED_NU<cl_ushort, 1>, G, L>::run(device, context, queue, num_elements, "test_redmax_non_uniform", redmax_non_uniform_source, 0, required_extensions);
    error |= test<cl_char, RED_NU<cl_char, 1>, G, L>::run(device, context, queue, num_elements, "test_redmax_non_uniform", redmax_non_uniform_source, 0, required_extensions);
    error |= test<cl_uchar, RED_NU<cl_uchar, 1>, G, L>::run(device, context, queue, num_elements, "test_redmax_non_uniform", redmax_non_uniform_source, 0, required_extensions);
    error |= test<cl_float, RED_NU<cl_float, 1>, G, L>::run(device, context, queue, num_elements, "test_redmax_non_uniform", redmax_non_uniform_source, 0, required_extensions);
    error |= test<cl_double, RED_NU<cl_double, 1>, G, L>::run(device, context, queue, num_elements, "test_redmax_non_uniform", redmax_non_uniform_source, 0, required_extensions);
    //error |= test<subgroups::cl_half, RED_NU<subgroups::cl_half, 1>, G, L>::run(device, context, queue, num_elements, "test_redmax_non_uniform", redmax_non_uniform_source, 0, required_extensions);

    error |= test<cl_int, RED_NU<cl_int, 2>, G, L>::run(device, context, queue, num_elements, "test_redmin_non_uniform", redmin_non_uniform_source, 0, required_extensions);
    error |= test<cl_uint, RED_NU<cl_uint, 2>, G, L>::run(device, context, queue, num_elements, "test_redmin_non_uniform", redmin_non_uniform_source, 0, required_extensions);
    error |= test<cl_long, RED_NU<cl_long, 2>, G, L>::run(device, context, queue, num_elements, "test_redmin_non_uniform", redmin_non_uniform_source, 0, required_extensions);
    error |= test<cl_ulong, RED_NU<cl_ulong, 2>, G, L>::run(device, context, queue, num_elements, "test_redmin_non_uniform", redmin_non_uniform_source, 0, required_extensions);
    error |= test<cl_short, RED_NU<cl_short, 2>, G, L>::run(device, context, queue, num_elements, "test_redmin_non_uniform", redmin_non_uniform_source, 0, required_extensions);
    error |= test<cl_ushort, RED_NU<cl_ushort, 2>, G, L>::run(device, context, queue, num_elements, "test_redmin_non_uniform", redmin_non_uniform_source, 0, required_extensions);
    error |= test<cl_char, RED_NU<cl_char, 2>, G, L>::run(device, context, queue, num_elements, "test_redmin_non_uniform", redmin_non_uniform_source, 0, required_extensions);
    error |= test<cl_uchar, RED_NU<cl_uchar, 2>, G, L>::run(device, context, queue, num_elements, "test_redmin_non_uniform", redmin_non_uniform_source, 0, required_extensions);
    error |= test<cl_float, RED_NU<cl_float, 2>, G, L>::run(device, context, queue, num_elements, "test_redmin_non_uniform", redmin_non_uniform_source, 0, required_extensions);
    error |= test<cl_double, RED_NU<cl_double, 2>, G, L>::run(device, context, queue, num_elements, "test_redmin_non_uniform", redmin_non_uniform_source, 0, required_extensions);
    //error |= test<subgroups::cl_half, RED_NU<subgroups::cl_half, 2>, G, L>::run(device, context, queue, num_elements, "test_redmin_non_uniform", redmin_non_uniform_source, 0, required_extensions);

    error |= test<cl_int, RED_NU<cl_int, 3>, G, L>::run(device, context, queue, num_elements, "test_redmul_non_uniform", redmul_non_uniform_source, 0, required_extensions);
    error |= test<cl_uint, RED_NU<cl_uint, 3>, G, L>::run(device, context, queue, num_elements, "test_redmul_non_uniform", redmul_non_uniform_source, 0, required_extensions);
    error |= test<cl_long, RED_NU<cl_long, 3>, G, L>::run(device, context, queue, num_elements, "test_redmul_non_uniform", redmul_non_uniform_source, 0, required_extensions);
    error |= test<cl_ulong, RED_NU<cl_ulong, 3>, G, L>::run(device, context, queue, num_elements, "test_redmul_non_uniform", redmul_non_uniform_source, 0, required_extensions);
    error |= test<cl_short, RED_NU<cl_short, 3>, G, L>::run(device, context, queue, num_elements, "test_redmul_non_uniform", redmul_non_uniform_source, 0, required_extensions);
    error |= test<cl_ushort, RED_NU<cl_ushort, 3>, G, L>::run(device, context, queue, num_elements, "test_redmul_non_uniform", redmul_non_uniform_source, 0, required_extensions);
    error |= test<cl_char, RED_NU<cl_char, 3>, G, L>::run(device, context, queue, num_elements, "test_redmul_non_uniform", redmul_non_uniform_source, 0, required_extensions);
    error |= test<cl_uchar, RED_NU<cl_uchar, 3>, G, L>::run(device, context, queue, num_elements, "test_redmul_non_uniform", redmul_non_uniform_source, 0, required_extensions);
    error |= test<cl_float, RED_NU<cl_float, 3>, G, L>::run(device, context, queue, num_elements, "test_redmul_non_uniform", redmul_non_uniform_source, 0, required_extensions);
    error |= test<cl_double, RED_NU<cl_double, 3>, G, L>::run(device, context, queue, num_elements, "test_redmul_non_uniform", redmul_non_uniform_source, 0, required_extensions);
    //error |= test<subgroups::cl_half, RED_NU<subgroups::cl_half, 3>, G, L>::run(device, context, queue, num_elements, "test_redmul_non_uniform", redmul_non_uniform_source, 0, required_extensions);

    error |= test<cl_int, RED_NU<cl_int, 4>, G, L>::run(device, context, queue, num_elements, "test_redand_non_uniform", redand_non_uniform_source, 0, required_extensions);
    error |= test<cl_uint, RED_NU<cl_uint, 4>, G, L>::run(device, context, queue, num_elements, "test_redand_non_uniform", redand_non_uniform_source, 0, required_extensions);
    error |= test<cl_long, RED_NU<cl_long, 4>, G, L>::run(device, context, queue, num_elements, "test_redand_non_uniform", redand_non_uniform_source, 0, required_extensions);
    error |= test<cl_ulong, RED_NU<cl_ulong, 4>, G, L>::run(device, context, queue, num_elements, "test_redand_non_uniform", redand_non_uniform_source, 0, required_extensions);
    error |= test<cl_short, RED_NU<cl_short, 4>, G, L>::run(device, context, queue, num_elements, "test_redand_non_uniform", redand_non_uniform_source, 0, required_extensions);
    error |= test<cl_ushort, RED_NU<cl_ushort, 4>, G, L>::run(device, context, queue, num_elements, "test_redand_non_uniform", redand_non_uniform_source, 0, required_extensions);
    error |= test<cl_char, RED_NU<cl_char, 4>, G, L>::run(device, context, queue, num_elements, "test_redand_non_uniform", redand_non_uniform_source, 0, required_extensions);
    error |= test<cl_uchar, RED_NU<cl_uchar, 4>, G, L>::run(device, context, queue, num_elements, "test_redand_non_uniform", redand_non_uniform_source, 0, required_extensions);

    error |= test<cl_int, RED_NU<cl_int, 5>, G, L>::run(device, context, queue, num_elements, "test_redor_non_uniform", redor_non_uniform_source, 0, required_extensions);
    error |= test<cl_uint, RED_NU<cl_uint, 5>, G, L>::run(device, context, queue, num_elements, "test_redor_non_uniform", redor_non_uniform_source, 0, required_extensions);
    error |= test<cl_long, RED_NU<cl_long, 5>, G, L>::run(device, context, queue, num_elements, "test_redor_non_uniform", redor_non_uniform_source, 0, required_extensions);
    error |= test<cl_ulong, RED_NU<cl_ulong, 5>, G, L>::run(device, context, queue, num_elements, "test_redor_non_uniform", redor_non_uniform_source, 0, required_extensions);
    error |= test<cl_short, RED_NU<cl_short, 5>, G, L>::run(device, context, queue, num_elements, "test_redor_non_uniform", redor_non_uniform_source, 0, required_extensions);
    error |= test<cl_ushort, RED_NU<cl_ushort, 5>, G, L>::run(device, context, queue, num_elements, "test_redor_non_uniform", redor_non_uniform_source, 0, required_extensions);
    error |= test<cl_char, RED_NU<cl_char, 5>, G, L>::run(device, context, queue, num_elements, "test_redor_non_uniform", redor_non_uniform_source, 0, required_extensions);
    error |= test<cl_uchar, RED_NU<cl_uchar, 5>, G, L>::run(device, context, queue, num_elements, "test_redor_non_uniform", redor_non_uniform_source, 0, required_extensions);

    error |= test<cl_int, RED_NU<cl_int, 6>, G, L>::run(device, context, queue, num_elements, "test_redxor_non_uniform", redxor_non_uniform_source, 0, required_extensions);
    error |= test<cl_uint, RED_NU<cl_uint, 6>, G, L>::run(device, context, queue, num_elements, "test_redxor_non_uniform", redxor_non_uniform_source, 0, required_extensions);
    error |= test<cl_long, RED_NU<cl_long, 6>, G, L>::run(device, context, queue, num_elements, "test_redxor_non_uniform", redxor_non_uniform_source, 0, required_extensions);
    error |= test<cl_ulong, RED_NU<cl_ulong, 6>, G, L>::run(device, context, queue, num_elements, "test_redxor_non_uniform", redxor_non_uniform_source, 0, required_extensions);
    error |= test<cl_short, RED_NU<cl_short, 6>, G, L>::run(device, context, queue, num_elements, "test_redxor_non_uniform", redxor_non_uniform_source, 0, required_extensions);
    error |= test<cl_ushort, RED_NU<cl_ushort, 6>, G, L>::run(device, context, queue, num_elements, "test_redxor_non_uniform", redxor_non_uniform_source, 0, required_extensions);
    error |= test<cl_char, RED_NU<cl_char, 6>, G, L>::run(device, context, queue, num_elements, "test_redxor_non_uniform", redxor_non_uniform_source, 0, required_extensions);
    error |= test<cl_uchar, RED_NU<cl_uchar, 6>, G, L>::run(device, context, queue, num_elements, "test_redxor_non_uniform", redxor_non_uniform_source, 0, required_extensions);

    error |= test<cl_int, RED_NU<cl_int, 7>, G, L>::run(device, context, queue, num_elements, "test_redand_non_uniform_logical", redand_non_uniform_logical_source, 0, required_extensions);
    error |= test<cl_int, RED_NU<cl_int, 8>, G, L>::run(device, context, queue, num_elements, "test_redor_non_uniform_logical", redor_non_uniform_logical_source, 0, required_extensions);
    error |= test<cl_int, RED_NU<cl_int, 9>, G, L>::run(device, context, queue, num_elements, "test_redxor_non_uniform_logical", redxor_non_uniform_logical_source, 0, required_extensions);

    return error;
}

