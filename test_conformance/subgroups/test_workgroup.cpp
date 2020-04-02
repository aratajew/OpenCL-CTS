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
#include "harness/conversions.h"
#include "harness/typeWrappers.h"
#include <bitset>

#define CLUSTER_SIZE 4
#define CLUSTER_SIZE_STR "4"

static const char * ballot_source =
"__kernel void test_sub_group_ballot(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    Type x = in[gid];\n"
"    uint4 value = sub_group_ballot(x.s0);\n"
"    out[gid] = value;\n"
"}\n";
static const char * inverse_ballot_source =
"__kernel void test_sub_group_inverse_ballot(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    Type x = in[gid];\n"
"    uint4 value = (uint4)(10,0,0,0);\n"
"    if (sub_group_inverse_ballot(x)) {\n"
"       value = (uint4)(1,0,0,0);\n"
"    } else {\n"
"       value = (uint4)(0,0,0,0);\n"
"    }\n"
"    out[gid] = value;\n"
"}\n";
static const char * ballot_bit_extract_source =
"__kernel void test_sub_group_ballot_bit_extract(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    Type x = in[gid];\n"
"    uint index = xy[gid].z;\n"
"    uint4 value = (uint4)(10,0,0,0);\n"
"    if (sub_group_ballot_bit_extract(x, index)) {\n"
"       value = (uint4)(1,0,0,0);\n"
"    } else {\n"
"       value = (uint4)(0,0,0,0);\n"
"    }\n"
"    out[gid] = value;\n"
"}\n";
static const char * ballot_bit_count_source =
"__kernel void test_sub_group_ballot_bit_count(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    Type x = in[gid];\n"
"    uint4 value = (uint4)(0,0,0,0);\n"
"    value = (uint4)(sub_group_ballot_bit_count(x),0,0,0);\n"
"    out[gid] = value;\n"
"}\n";
static const char * ballot_inclusive_scan_source =
"__kernel void test_sub_group_ballot_inclusive_scan(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    Type x = in[gid];\n"
"    uint4 value = (uint4)(0,0,0,0);\n"
"    value = (uint4)(sub_group_ballot_inclusive_scan(x),0,0,0);\n"
"    out[gid] = value;\n"
"}\n";
static const char * ballot_exclusive_scan_source =
"__kernel void test_sub_group_ballot_exclusive_scan(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    Type x = in[gid];\n"
"    uint4 value = (uint4)(0,0,0,0);\n"
"    value = (uint4)(sub_group_ballot_exclusive_scan(x),0,0,0);\n"
"    out[gid] = value;\n"
"}\n";
static const char * ballot_find_lsb_source =
"__kernel void test_sub_group_ballot_find_lsb(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    Type x = in[gid];\n"
"    uint4 value = (uint4)(0,0,0,0);\n"
"    value = (uint4)(sub_group_ballot_find_lsb(x),0,0,0);\n"
"    out[gid] = value;\n"
"}\n";
static const char * ballot_find_msb_source =
"__kernel void test_sub_group_ballot_find_msb(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    Type x = in[gid];\n"
"    uint4 value = (uint4)(0,0,0,0);"
"    value = (uint4)(sub_group_ballot_find_msb(x),0,0,0);"
"    out[gid] = value ;"
"}\n";
static const char * shuffle_xor_source =
"__kernel void test_sub_group_shuffle_xor(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    Type x = in[gid];\n"
"    out[gid] = sub_group_shuffle_xor(x, xy[gid].z);"
"}\n";
static const char * shuffle_down_source =
"__kernel void test_sub_group_shuffle_down(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    Type x = in[gid];\n"
"    out[gid] = sub_group_shuffle_down(x, xy[gid].z);"
"}\n";
static const char * shuffle_up_source =
"__kernel void test_sub_group_shuffle_up(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    Type x = in[gid];\n"
"    out[gid] = sub_group_shuffle_up(x, xy[gid].z);"
"}\n";
static const char * shuffle_source =
"__kernel void test_sub_group_shuffle(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    Type x = in[gid];\n"
"    out[gid] = sub_group_shuffle(x, xy[gid].z);"
"}\n";
static const char * get_subgroup_ge_mask_source =
"__kernel void test_get_sub_group_ge_mask(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    xy[gid].z = get_max_sub_group_size();\n"
"    Type x = in[gid];\n"
"    uint4 mask = get_sub_group_ge_mask();"
"    out[gid] = mask;\n"
"}\n";
static const char * get_subgroup_gt_mask_source =
"__kernel void test_get_sub_group_gt_mask(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    xy[gid].z = get_max_sub_group_size();\n"
"    Type x = in[gid];\n"
"    uint4 mask = get_sub_group_gt_mask();"
"    out[gid] = mask;\n"
"}\n";
static const char * get_subgroup_le_mask_source =
"__kernel void test_get_sub_group_le_mask(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    xy[gid].z = get_max_sub_group_size();\n"
"    Type x = in[gid];\n"
"    uint4 mask = get_sub_group_le_mask();"
"    out[gid] = mask;\n"
"}\n";
static const char * get_subgroup_lt_mask_source =
"__kernel void test_get_sub_group_lt_mask(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    xy[gid].z = get_max_sub_group_size();\n"
"    Type x = in[gid];\n"
"    uint4 mask = get_sub_group_lt_mask();"
"    out[gid] = mask;\n"
"}\n";
static const char * get_subgroup_eq_mask_source =
"__kernel void test_get_sub_group_eq_mask(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    xy[gid].z = get_max_sub_group_size();\n"
"    Type x = in[gid];\n"
"    uint4 mask = get_sub_group_eq_mask();"
"    out[gid] = mask;\n"
"}\n";
static const char * elect_source =
"__kernel void test_elect(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    Type x = in[gid];\n"
"    int am_i_elected = sub_group_elect();\n"
"    out[gid] = am_i_elected;\n" //one in subgroup true others false.
"}\n";
static const char * any_source =
"__kernel void test_any(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    out[gid] = sub_group_any(in[gid]);\n"
"}\n";

static const char * all_source =
"__kernel void test_all(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    out[gid] = sub_group_all(in[gid]);\n"
"}\n";

static const char * non_uniform_any_source =
"__kernel void test_non_uniform_any(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    if (xy[gid].x < NON_UNIFORM) {\n"
"        out[gid] = sub_group_non_uniform_any(in[gid]);\n"
"    }\n"
"}\n";
static const char * non_uniform_all_source =
"__kernel void test_non_uniform_all(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    if (xy[gid].x < NON_UNIFORM) {"
"        out[gid] = sub_group_non_uniform_all(in[gid]);\n"
"    }"
"}\n";
static const char * non_uniform_all_equal_source =
"__kernel void test_non_uniform_all_equal(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
" if (xy[gid].x < NON_UNIFORM) {"
"    out[gid] = sub_group_non_uniform_all_equal(in[gid]);\n"
"}"
"}\n";
static const char * bcast_source =
"__kernel void test_bcast(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    Type x = in[gid];\n"
"    out[gid] = sub_group_broadcast(x, xy[gid].z);\n"

"}\n";
static const char * bcast_non_uniform_source =
"__kernel void test_bcast_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    Type x = in[gid];\n"
" if (xy[gid].x < NON_UNIFORM) {\n" // broadcast 4 values , other values are 0
"    out[gid] = sub_group_broadcast(x, xy[gid].z);\n"
" }\n"
"}\n";
static const char * bcast_first_source =
"__kernel void test_bcast_first(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    Type x = in[gid];\n"
"    out[gid] = sub_group_broadcast_first(x);\n"
"}\n";

static const char * redadd_source =
"__kernel void test_redadd(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    out[gid] = sub_group_reduce_add(in[gid]);\n"
"}\n";

static const char * redmax_source =
"__kernel void test_redmax(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    out[gid] = sub_group_reduce_max(in[gid]);\n"
"}\n";

static const char * redmin_source =
"__kernel void test_redmin(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    out[gid] = sub_group_reduce_min(in[gid]);\n"
"}\n";

static const char * scinadd_source =
"__kernel void test_scinadd(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    out[gid] = sub_group_scan_inclusive_add(in[gid]);\n"
"}\n";

static const char * scinmax_source =
"__kernel void test_scinmax(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    out[gid] = sub_group_scan_inclusive_max(in[gid]);\n"
"}\n";

static const char * scinmin_source =
"__kernel void test_scinmin(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    out[gid] = sub_group_scan_inclusive_min(in[gid]);\n"
"}\n";

static const char * scinadd_non_uniform_source =
"__kernel void test_scinadd_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
" if (xy[gid].x < NON_UNIFORM) {"
"    out[gid] = sub_group_non_uniform_scan_inclusive_add(in[gid]);\n"
" }"
//"printf(\"gid = %d, sub group local id = %d, sub group id = %d, x form in = %d, new_set = %d, out[gid] = %d\\n\",gid,xy[gid].x, xy[gid].y, x, xy[gid].z, out[gid]);"
"}\n";
static const char * scinmax_non_uniform_source =
"__kernel void test_scinmax_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
" if (xy[gid].x < NON_UNIFORM) {"
"    out[gid] = sub_group_non_uniform_scan_inclusive_max(in[gid]);\n"
" }"
"}\n";
static const char * scinmin_non_uniform_source =
"__kernel void test_scinmin_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
" if (xy[gid].x < NON_UNIFORM) {"
"    out[gid] = sub_group_non_uniform_scan_inclusive_min(in[gid]);\n"
" }"
"}\n";
static const char * scinmul_non_uniform_source =
"__kernel void test_scinmul_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
" if (xy[gid].x < NON_UNIFORM) {"
"    out[gid] = sub_group_non_uniform_scan_inclusive_mul(in[gid]);\n"
" }"
"}\n";
static const char * scinand_non_uniform_source =
"__kernel void test_scinand_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
" if (xy[gid].x < NON_UNIFORM) {"
"    out[gid] = sub_group_non_uniform_scan_inclusive_and(in[gid]);\n"
" }"
"}\n";
static const char * scinor_non_uniform_source =
"__kernel void test_scinor_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
" if (xy[gid].x < NON_UNIFORM) {"
"    out[gid] = sub_group_non_uniform_scan_inclusive_or(in[gid]);\n"
" }"
"}\n";
static const char * scinxor_non_uniform_source =
"__kernel void test_scinxor_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
" if (xy[gid].x < NON_UNIFORM) {"
"    out[gid] = sub_group_non_uniform_scan_inclusive_xor(in[gid]);\n"
" }"
"}\n";
static const char * scinand_non_uniform_logical_source =
"__kernel void test_scinand_non_uniform_logical(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
" if (xy[gid].x < NON_UNIFORM) {"
"    out[gid] = sub_group_non_uniform_scan_inclusive_logical_and(in[gid]);\n"
" }"
"}\n";
static const char * scinor_non_uniform_logical_source =
"__kernel void test_scinor_non_uniform_logical(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
" if (xy[gid].x < NON_UNIFORM) {"
"    out[gid] = sub_group_non_uniform_scan_inclusive_logical_or(in[gid]);\n"
" }"
"}\n";
static const char * scinxor_non_uniform_logical_source =
"__kernel void test_scinxor_non_uniform_logical(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
" if (xy[gid].x < NON_UNIFORM) {"
"    out[gid] = sub_group_non_uniform_scan_inclusive_logical_xor(in[gid]);\n"
" }"
"}\n";
static const char * scexadd_source =
"__kernel void test_scexadd(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    out[gid] = sub_group_scan_exclusive_add(in[gid]);\n"
"}\n";

static const char * scexmax_source =
"__kernel void test_scexmax(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    out[gid] = sub_group_scan_exclusive_max(in[gid]);\n"
"}\n";

static const char * scexmin_source =
"__kernel void test_scexmin(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    out[gid] = sub_group_scan_exclusive_min(in[gid]);\n"
"}\n";

static const char * scexadd_non_uniform_source =
"__kernel void test_scexadd_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
" if (xy[gid].x < NON_UNIFORM) {"
"    out[gid] = sub_group_non_uniform_scan_exclusive_add(in[gid]);\n"
" }"
//"printf(\"gid = %d, sub group local id = %d, sub group id = %d, x form in = %d, new_set = %d, out[gid] = %d , x = %d\\n\",gid,xy[gid].x, xy[gid].y, x, xy[gid].z, out[gid]);"
"}\n";

static const char * scexmax_non_uniform_source =
"__kernel void test_scexmax_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
" if (xy[gid].x < NON_UNIFORM) {"
"    out[gid] = sub_group_non_uniform_scan_exclusive_max(in[gid]);\n"
" }"
"}\n";

static const char * scexmin_non_uniform_source =
"__kernel void test_scexmin_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
" if (xy[gid].x < NON_UNIFORM) {"
"    out[gid] = sub_group_non_uniform_scan_exclusive_min(in[gid]);\n"
" }"
"}\n";

static const char * scexmul_non_uniform_source =
"__kernel void test_scexmul_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
" if (xy[gid].x < NON_UNIFORM) {"
"    out[gid] = sub_group_non_uniform_scan_exclusive_mul(in[gid]);\n"
" }"
"}\n";

static const char * scexand_non_uniform_source =
"__kernel void test_scexand_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
" if (xy[gid].x < NON_UNIFORM) {"
"    out[gid] = sub_group_non_uniform_scan_exclusive_and(in[gid]);\n"
" }"
"}\n";

static const char * scexor_non_uniform_source =
"__kernel void test_scexor_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
" if (xy[gid].x < NON_UNIFORM) {"
"    out[gid] = sub_group_non_uniform_scan_exclusive_or(in[gid]);\n"
" }"
"}\n";

static const char * scexxor_non_uniform_source =
"__kernel void test_scexxor_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
" if (xy[gid].x < NON_UNIFORM) {"
"    out[gid] = sub_group_non_uniform_scan_exclusive_xor(in[gid]);\n"
" }"
"}\n";

static const char * scexand_non_uniform_logical_source =
"__kernel void test_scexand_non_uniform_logical(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
" if (xy[gid].x < NON_UNIFORM) {"
"    out[gid] = sub_group_non_uniform_scan_exclusive_logical_and(in[gid]);\n"
" }"
"}\n";

static const char * scexor_non_uniform_logical_source =
"__kernel void test_scexor_non_uniform_logical(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
" if (xy[gid].x < NON_UNIFORM) {"
"    out[gid] = sub_group_non_uniform_scan_exclusive_logical_or(in[gid]);\n"
" }"
"}\n";

static const char * scexxor_non_uniform_logical_source =
"__kernel void test_scexxor_non_uniform_logical(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
" if (xy[gid].x < NON_UNIFORM) {"
"    out[gid] = sub_group_non_uniform_scan_exclusive_logical_xor(in[gid]);\n"
" }"
"}\n";

static const char * redadd_non_uniform_source =
"__kernel void test_redadd_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
" if (xy[gid].x < NON_UNIFORM) {"
"    out[gid] = sub_group_non_uniform_reduce_add(in[gid]);\n"
" }"
//"printf(\"gid = %d, sub group local id = %d, sub group id = %d, x form in = %d, new_set = %d, out[gid] = %d\\n\",gid,xy[gid].x, xy[gid].y, x, xy[gid].z, out[gid]);"
"}\n";

static const char * redmax_non_uniform_source =
"__kernel void test_redmax_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
" if (xy[gid].x < NON_UNIFORM) {"
"    out[gid] = sub_group_non_uniform_reduce_max(in[gid]);\n"
" }"
"}\n";

static const char * redmin_non_uniform_source =
"__kernel void test_redmin_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
" if (xy[gid].x < NON_UNIFORM) {"
"    out[gid] = sub_group_non_uniform_reduce_min(in[gid]);\n"
" }"
"}\n";

static const char * redmul_non_uniform_source =
"__kernel void test_redmul_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
" if (xy[gid].x < NON_UNIFORM) {"
"    out[gid] = sub_group_non_uniform_reduce_mul(in[gid]);\n"
" }"
"}\n";

static const char * redand_non_uniform_source =
"__kernel void test_redand_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
" if (xy[gid].x < NON_UNIFORM) {"
"    out[gid] = sub_group_non_uniform_reduce_and(in[gid]);\n"
" }"
"}\n";

static const char * redor_non_uniform_source =
"__kernel void test_redor_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
" if (xy[gid].x < NON_UNIFORM) {"
"    out[gid] = sub_group_non_uniform_reduce_or(in[gid]);\n"
" }"
"}\n";

static const char * redxor_non_uniform_source =
"__kernel void test_redxor_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
" if (xy[gid].x < NON_UNIFORM) {"
"    out[gid] = sub_group_non_uniform_reduce_xor(in[gid]);\n"
" }"
"}\n";

static const char * redand_non_uniform_logical_source =
"__kernel void test_redand_non_uniform_logical(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
" if (xy[gid].x < NON_UNIFORM) {"
"    out[gid] = sub_group_non_uniform_reduce_logical_and(in[gid]);\n"
" }"
"}\n";

static const char * redor_non_uniform_logical_source =
"__kernel void test_redor_non_uniform_logical(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
" if (xy[gid].x < NON_UNIFORM) {"
"    out[gid] = sub_group_non_uniform_reduce_logical_or(in[gid]);\n"
" }"
"}\n";

static const char * redxor_non_uniform_logical_source =
"__kernel void test_redxor_non_uniform_logical(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
" if (xy[gid].x < NON_UNIFORM) {"
"    out[gid] = sub_group_non_uniform_reduce_logical_xor(in[gid]);\n"
" }"
"}\n";

static const char * redadd_clustered_source =
"__kernel void test_redadd_clustered(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    out[gid] = sub_group_clustered_reduce_add(in[gid], " CLUSTER_SIZE_STR ");\n"
//"printf(\"gid = %d, sub group local id = %d, sub group id = %d, x form in = %d, new_set = %d, out[gid] = %d\\n\", gid, xy[gid].x, xy[gid].y, in[gid], xy[gid].z, out[gid]);"
"}\n";

static const char * redmax_clustered_source =
"__kernel void test_redmax_clustered(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    out[gid] = sub_group_clustered_reduce_max(in[gid], " CLUSTER_SIZE_STR ");\n"
"}\n";

static const char * redmin_clustered_source =
"__kernel void test_redmin_clustered(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    out[gid] = sub_group_clustered_reduce_min(in[gid], " CLUSTER_SIZE_STR ");\n"
"}\n";

static const char * redmul_clustered_source =
"__kernel void test_redmul_clustered(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    out[gid] = sub_group_clustered_reduce_mul(in[gid], " CLUSTER_SIZE_STR ");\n"
"}\n";

static const char * redand_clustered_source =
"__kernel void test_redand_clustered(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    out[gid] = sub_group_clustered_reduce_and(in[gid], " CLUSTER_SIZE_STR ");\n"
"}\n";

static const char * redor_clustered_source =
"__kernel void test_redor_clustered(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    out[gid] = sub_group_clustered_reduce_or(in[gid], " CLUSTER_SIZE_STR ");\n"
"}\n";

static const char * redxor_clustered_source =
"__kernel void test_redxor_clustered(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    out[gid] = sub_group_clustered_reduce_xor(in[gid], " CLUSTER_SIZE_STR ");\n"
"}\n";

static const char * redand_clustered_logical_source =
"__kernel void test_redand_clustered_logical(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    out[gid] = sub_group_clustered_reduce_logical_and(in[gid], " CLUSTER_SIZE_STR ");\n"
"}\n";

static const char * redor_clustered_logical_source =
"__kernel void test_redor_clustered_logical(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    out[gid] = sub_group_clustered_reduce_logical_or(in[gid], " CLUSTER_SIZE_STR ");\n"
"}\n";

static const char * redxor_clustered_logical_source =
"__kernel void test_redxor_clustered_logical(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    out[gid] = sub_group_clustered_reduce_logical_xor(in[gid], " CLUSTER_SIZE_STR ");\n"
"}\n";

// These need to stay in sync with the kernel source below
#define NUM_LOC 49
#define INST_LOC_MASK 0x7f
#define INST_OP_SHIFT 0
#define INST_OP_MASK 0xf
#define INST_LOC_SHIFT 4
#define INST_VAL_SHIFT 12
#define INST_VAL_MASK 0x7ffff
#define INST_END 0x0
#define INST_STORE 0x1
#define INST_WAIT 0x2
#define INST_COUNT 0x3

static const char * ifp_source =
"#define NUM_LOC 49\n"
"#define INST_LOC_MASK 0x7f\n"
"#define INST_OP_SHIFT 0\n"
"#define INST_OP_MASK 0xf\n"
"#define INST_LOC_SHIFT 4\n"
"#define INST_VAL_SHIFT 12\n"
"#define INST_VAL_MASK 0x7ffff\n"
"#define INST_END 0x0\n"
"#define INST_STORE 0x1\n"
"#define INST_WAIT 0x2\n"
"#define INST_COUNT 0x3\n"
"\n"
"__kernel void\n"
"test_ifp(const __global int *in, __global int4 *xy, __global int *out)\n"
"{\n"
"    __local atomic_int loc[NUM_LOC];\n"
"\n"
"    // Don't run if there is only one sub group\n"
"    if (get_num_sub_groups() == 1)\n"
"        return;\n"
"\n"
"    // First initialize loc[]\n"
"    int lid = (int)get_local_id(0);\n"
"\n"
"    if (lid < NUM_LOC)\n"
"        atomic_init(loc+lid, 0);\n"
"\n"
"    work_group_barrier(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"    // Compute pointer to this sub group's \"instructions\"\n"
"    const __global int *pc = in +\n"
"        ((int)get_group_id(0)*(int)get_enqueued_num_sub_groups() +\n"
"         (int)get_sub_group_id()) *\n"
"        (NUM_LOC+1);\n"
"\n"
"    // Set up to \"run\"\n"
"    bool ok = (int)get_sub_group_local_id() == 0;\n"
"    bool run = true;\n"
"\n"
"    while (run) {\n"
"        int inst = *pc++;\n"
"        int iop = (inst >> INST_OP_SHIFT) & INST_OP_MASK;\n"
"        int iloc = (inst >> INST_LOC_SHIFT) & INST_LOC_MASK;\n"
"        int ival = (inst >> INST_VAL_SHIFT) & INST_VAL_MASK;\n"
"\n"
"        switch (iop) {\n"
"        case INST_STORE:\n"
"            if (ok)\n"
"                atomic_store(loc+iloc, ival);\n"
"            break;\n"
"        case INST_WAIT:\n"
"            if (ok) {\n"
"                while (atomic_load(loc+iloc) != ival)\n"
"                    ;\n"
"            }\n"
"            break;\n"
"        case INST_COUNT:\n"
"            if (ok) {\n"
"                int i;\n"
"                for (i=0;i<ival;++i)\n"
"                    atomic_fetch_add(loc+iloc, 1);\n"
"            }\n"
"            break;\n"
"        case INST_END:\n"
"            run = false;\n"
"            break;\n"
"        }\n"
"\n"
"        sub_group_barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"\n"
"    work_group_barrier(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"    // Save this group's result\n"
"    __global int *op = out + (int)get_group_id(0)*NUM_LOC;\n"
"    if (lid < NUM_LOC)\n"
"        op[lid] = atomic_load(loc+lid);\n"
"}\n";

cl_uint set_bit(cl_uint bit_value, cl_uint number, cl_uint position) {
    number ^= (-(bit_value) ^ number) & (1UL << position);
    return number;
}
cl_uint4 generate_bit_mask(cl_uint subgroup_local_id, std::string mask_type, cl_uint max_sub_group_size) {
    typedef std::bitset<128> bs128;
    bs128 mask128;
    cl_uint4 mask;
    cl_uint pos = subgroup_local_id;
    if (mask_type == "eq")
        mask128.set(pos);
    if (mask_type == "le" || mask_type == "lt") {
        for (cl_uint i = 0; i <= pos; i++)
            mask128.set(i);
        if (mask_type == "lt")
            mask128.reset(pos);
    }
    if (mask_type == "ge" || mask_type == "gt") {
        for (cl_uint i = pos; i < max_sub_group_size; i++)
            mask128.set(i);
        if (mask_type == "gt")
            mask128.reset(pos);
    }

    // convert std::bitset<128> to uint4
    auto const uint_mask = bs128{ static_cast<unsigned long>(-1) };
    mask.s0 = (mask128 & uint_mask).to_ulong();
    mask128 >>= 32;
    mask.s1 = (mask128 & uint_mask).to_ulong();
    mask128 >>= 32;
    mask.s2 = (mask128 & uint_mask).to_ulong();
    mask128 >>= 32;
    mask.s3 = (mask128 & uint_mask).to_ulong();

    return mask;
}
template <typename Ty, int Which = 0>
struct SHF {
    static void gen(Ty *x, Ty *t, cl_int *m, int ns, int nw, int ng)
    {
        int i, ii, j, k, l, n, delta;
        int nj = (nw + ns - 1) / ns;
        int d = ns > 100 ? 100 : ns;
        ii = 0;
        for (k = 0; k < ng; ++k) {      // for each work_group
            for (j = 0; j < nj; ++j) {  // for each subgroup
                ii = j * ns;
                n = ii + ns > nw ? nw - ii : ns;
                for (i = 0; i < n; ++i) {
                    int midx = 4 * ii + 4 * i + 2;
                    l = (int)(genrand_int32(gMTdata) & 0x7fffffff) % (d > n ? n : d);
                    if (Which == 0 || Which == 3) {
                        m[midx] = (cl_int)l; // storing information about shuffle index
                    }

                    if (Which == 1) {
                        delta = l; // calculate delta for shuffle up
                        if (i - delta < 0) {
                            delta = i;
                        }
                        m[midx] = (cl_int)delta;
                    }

                    if (Which == 2) {
                        delta = l; // calculate delta for shuffle down
                        if (i + delta >= n) {
                            delta = n - 1 - i;
                        }
                        m[midx] = (cl_int)delta;
                    }

                    cl_uint number;
                    number = (int)(genrand_int32(gMTdata) & 0x7fffffff); // calculate value for shuffle function
                    set_value(t[ii + i], number);
                    //log_info("wg = %d ,sg = %d, inside sg = %d, number == %d, l = %d, midx = %d\n", k, j, i, number, l, midx);
                }
            }
            // Now map into work group using map from device
            for (j = 0; j < nw; ++j) {              // for each element in work_group
                i = m[4 * j + 1] * ns + m[4 * j];   // calculate index as number of subgroup plus subgroup local id
                x[j] = t[i];
            }
            x += nw;
            m += 4 * nw;
        }
    }

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m, int ns, int nw, int ng)
    {
        int ii, i, j, k, l, n;
        int nj = (nw + ns - 1) / ns;
        Ty tr, rr;


        log_info("  sub_group_shuffle%s(%s)...\n", Which == 0 ? "" : (Which == 1 ? "_up" : Which == 2 ? "_down" : "_xor"), TypeName<Ty>::val());

        for (k = 0; k < ng; ++k) {      // for each work_group
            for (j = 0; j < nw; ++j) {  // inside the work_group
                i = m[4 * j + 1] * ns + m[4 * j];
                mx[i] = x[j];           // read host inputs for work_group
                my[i] = y[j];           // read device outputs for work_group
            }

            for (j = 0; j < nj; ++j) {  // for each subgroup
                ii = j * ns;
                n = ii + ns > nw ? nw - ii : ns;

                for (i = 0; i < n; ++i) {  // inside the subgroup
                    int midx = 4 * ii + 4 * i + 2; // shuffle index storage
                    l = (int)m[midx];
                    rr = my[ii + i];
                    if (Which == 0) {
                        tr = mx[ii + l];        //shuffle basic - treat l as index
                    }
                    if (Which == 1) {
                        tr = mx[ii + i - l];    //shuffle up - treat l as delta
                    }
                    if (Which == 2) {
                        tr = mx[ii + i + l];    //shuffle down - treat l as delta
                    }
                    if (Which == 3) {
                        tr = mx[ii + (i ^ l)];  //shuffle xor - treat l as mask
                    }

                    if (!compare(rr, tr)) {
                        log_error("ERROR: sub_group_shuffle%s(%s) mismatch for local id %d in sub group %d in group %d\n", Which == 0 ? "" : (Which == 1 ? "_up" : (Which == 2 ? "_down" : "_xor")),
                            TypeName<Ty>::val(), i, j, k);
                        return -1;
                    }
                }
            }
            x += nw;
            y += nw;
            m += 4 * nw;
        }
        return 0;
    }
};

// test mask functions 0 means equal
// DESCRIPTION :
// test mask functions : 0 - equal, 1 - ge, 2 - gt, 3 - le, 4 - lt

static const char smask_names[][3] = { "eq", "ge", "gt", "le", "lt" };

template <typename Ty, int Which>
struct SMASK {
    static void gen(Ty *x, Ty *t, cl_int *m, int ns, int nw, int ng)
    {
        int i, ii, j, k, n, l;
        int nj = (nw + ns - 1) / ns;
        int d = ns > 100 ? 100 : ns;
        int e;

        ii = 0;
        for (k = 0; k < ng; ++k) {      // for each work_group
            for (j = 0; j < nj; ++j) {  // for each subgroup
                ii = j * ns;
                n = ii + ns > nw ? nw - ii : ns;
                // Produce expected masks for each work item in the subgroup
                for (i = 0; i < n; ++i) {
                    int midx = 4 * ii + 4 * i;
                    cl_uint max_sub_group_size = m[midx+2];
                    cl_uint4 expected_mask = { 0 };
                    if (Which == 0) {
                        expected_mask = generate_bit_mask(i, "eq", max_sub_group_size);
                    }
                    if (Which == 1) {
                        expected_mask = generate_bit_mask(i, "ge", max_sub_group_size);
                    }
                    if (Which == 2) {
                        expected_mask = generate_bit_mask(i, "gt", max_sub_group_size);
                    }
                    if (Which == 3) {
                        expected_mask = generate_bit_mask(i, "le", max_sub_group_size);
                    }
                    if (Which == 4) {
                        expected_mask = generate_bit_mask(i, "lt", max_sub_group_size);
                    }
                    set_value(t[ii + i], expected_mask);
                }
            }

            // Now map into work group using map from device
            for (j = 0; j < nw; ++j) {
                i = m[4 * j + 1] * ns + m[4 * j];
                x[j] = t[i];
            }
            x += nw;
            m += 4 * nw;
        }
    }

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m, int ns, int nw, int ng)
    {
        int ii, i, j, k, n;
        int nj = (nw + ns - 1) / ns;
        Ty taa, raa;

        log_info("  get_sub_group_%s_mask...\n", smask_names[Which]);

        for (k = 0; k < ng; ++k) {      // for each work_group
            for (j = 0; j < nw; ++j) {  // inside the work_group
                i = m[4 * j + 1] * ns + m[4 * j];
                mx[i] = x[j];           // read host inputs for work_group
                my[i] = y[j];           // read device outputs for work_group
            }

            for (j = 0; j < nj; ++j) {
                ii = j * ns;
                n = ii + ns > nw ? nw - ii : ns;

                // Check result
                for (i = 0; i < n; ++i) { // inside the subgroup
                    taa = mx[ii + i];     // read host input for subgroup
                    raa = my[ii + i];     // read device outputs for subgroup
                    if (!compare(raa, taa)) {
                        log_error("ERROR:  get_sub_group_%s_mask... mismatch for local id %d in sub group %d in group %d, obtained %d, expected %d\n", smask_names[Which], i, j, k, raa, taa);
                        return -1;
                    }
                }
            }
            x += nw;
            y += nw;
            m += 4 * nw;
        }
        return 0;
    }
};

// DESCRIPTION :
// Test any/all/all_equal non uniform test functions non uniform:
// Which 0 - any, 1 - all, 2 - all equal
template <typename Ty, int Which>
struct AAN {
    static void gen(Ty *x, Ty *t, cl_int *m, int ns, int nw, int ng)
    {
        int i, ii, j, k, n;
        int nj = (nw + ns - 1) / ns;
        int e;

        ii = 0;
        for (k = 0; k < ng; ++k) {      // for each work_group
            for (j = 0; j < nj; ++j) {  // for each subgroup
                ii = j * ns;
                n = ii + ns > nw ? nw - ii : ns;
                e = (int)(genrand_int32(gMTdata) % 3);

                // Initialize data matrix indexed by local id and sub group id
                switch (e) {
                case 0:
                    memset(&t[ii], 0, n * sizeof(Ty));
                    break;
                case 1:
                    memset(&t[ii], 0, n * sizeof(Ty));
                    i = (int)(genrand_int32(gMTdata) % (cl_uint)n);
                    t[ii + i] = 41;
                    break;
                case 2:
                    memset(&t[ii], 0xff, n * sizeof(Ty));
                    break;
                }
            }
            // Now map into work group using map from device
            for (j = 0; j < nw; ++j) {
                i = m[4 * j + 1] * ns + m[4 * j];
                x[j] = t[i];
            }
            x += nw;
            m += 4 * nw;
        }
    }

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m, int ns, int nw, int ng)
    {
        int ii, i, j, k, n;
        int nj = (nw + ns - 1) / ns;
        cl_int taa, raa;

        if (Which == 0)
            log_info("  sub_group_non_uniform_any...\n");
        else if (Which == 1)
            log_info("  sub_group_non_uniform_all...\n");
        else if (Which == 2)
            log_info("  sub_group_non_uniform_all_equal(%s)...\n", TypeName<Ty>::val());

        for (k = 0; k < ng; ++k) {      // for each work_group
            // Map to array indexed to array indexed by local ID and sub group
            for (j = 0; j < nw; ++j) {  // inside the work_group
                i = m[4 * j + 1] * ns + m[4 * j];
                mx[i] = x[j];           // read host inputs for work_group
                my[i] = y[j];           // read device outputs for work_group
            }

            for (j = 0; j < nj; ++j) {  // for each subgroup
                ii = j * ns;
                n = ii + ns > nw ? nw - ii : ns;

                // Compute target
                if (Which == 0) {       // function any
                    taa = 0;
                    for (i = 0; i < NON_UNIFORM; ++i)
                        taa |= mx[ii + i] != 0; // return non zero if value non zero at least for one
                }
                else if (Which == 1) {  // function all
                    taa = 1;
                    for (i = 0; i < NON_UNIFORM; ++i)
                        taa &= mx[ii + i] != 0; // return non zero if value non zero for all
                } else {                // function all equal
                    taa = 1;
                    for (i = 0; i < NON_UNIFORM; ++i) {
                        taa &= mx[ii] == mx[ii + i]; // return non zero if all the same
                    }
                }

                // Check result
                for (i = 0; i < n && i < NON_UNIFORM; ++i) {
                    raa = my[ii + i] != 0;
                    if (raa != taa) {
                        log_error("ERROR: sub_group_non_uniform_%s mismatch for local id %d in sub group %d in group %d, obtained %d, expected %d\n",
                            Which == 0 ? "any" : (Which == 1 ? "all" : "all_equal"), i, j, k, raa, taa);
                        return -1;
                    }
                }
            }
            x += nw;
            y += nw;
            m += 4 * nw;
        }
        return 0;
    }
};

// Any/All test functions
template <int Which>
struct AA {
    static void gen(cl_int *x, cl_int *t, cl_int *m, int ns, int nw, int ng)
    {
        int i, ii, j, k, n;
        int nj = (nw + ns - 1)/ns;
        int e;

        ii = 0;
        for (k=0; k<ng; ++k) {
            for (j=0; j<nj; ++j) {
                ii = j*ns;
                n = ii + ns > nw ? nw - ii : ns;
                e = (int)(genrand_int32(gMTdata) % 3);

                // Initialize data matrix indexed by local id and sub group id
                switch (e) {
                case 0:
                    memset(&t[ii], 0, n*sizeof(cl_int));
                    break;
                case 1:
                    memset(&t[ii], 0, n*sizeof(cl_int));
                    i = (int)(genrand_int32(gMTdata) % (cl_uint)n);
                    t[ii + i] = 41;
                    break;
                case 2:
                    memset(&t[ii], 0xff, n*sizeof(cl_int));
                    break;
                }
            }

            // Now map into work group using map from device
            for (j=0;j<nw;++j) {
                i = m[4*j+1]*ns + m[4*j];
                x[j] = t[i];
            }

            x += nw;
        m += 4*nw;
        }
    }

    static int chk(cl_int *x, cl_int *y, cl_int *mx, cl_int *my, cl_int *m, int ns, int nw, int ng)
    {
        int ii, i, j, k, n;
        int nj = (nw + ns - 1)/ns;
        cl_int taa, raa;

        log_info("  sub_group_%s...\n", Which == 0 ? "any" : "all");

        for (k=0; k<ng; ++k) {
            // Map to array indexed to array indexed by local ID and sub group
            for (j=0; j<nw; ++j) {
                i = m[4*j+1]*ns + m[4*j];
                mx[i] = x[j];
                my[i] = y[j];
            }

            for (j=0; j<nj; ++j) {
                ii = j*ns;
                n = ii + ns > nw ? nw - ii : ns;

                // Compute target
                if (Which == 0) {
                    taa = 0;
                    for (i=0; i<n; ++i)
                        taa |=  mx[ii + i] != 0;
                } else {
                    taa = 1;
                    for (i=0; i<n; ++i)
                        taa &=  mx[ii + i] != 0;
                }

                // Check result
                for (i=0; i<n; ++i) {
                    raa = my[ii+i] != 0;
                    if (raa != taa) {
                        log_error("ERROR: sub_group_%s mismatch for local id %d in sub group %d in group %d\n",
                                   Which == 0 ? "any" : "all", i, j, k);
                        return -1;
                    }
                }
            }

            x += nw;
            y += nw;
            m += 4*nw;
        }

        return 0;
    }
};

template <typename Ty, int Which>
struct OPERATION;

template <typename Ty> struct OPERATION<Ty, 0> { static Ty calculate(Ty a, Ty b) { return a + b; } };
template <typename Ty> struct OPERATION<Ty, 1> { static Ty calculate(Ty a, Ty b) { return a > b ? a : b; } };
template <typename Ty> struct OPERATION<Ty, 2> { static Ty calculate(Ty a, Ty b) { return a < b ? a : b; } };
template <typename Ty> struct OPERATION<Ty, 3> { static Ty calculate(Ty a, Ty b) { return a * b; } };
template <typename Ty> struct OPERATION<Ty, 4> { static Ty calculate(Ty a, Ty b) { return a & b; } };
template <typename Ty> struct OPERATION<Ty, 5> { static Ty calculate(Ty a, Ty b) { return a | b; } };
template <typename Ty> struct OPERATION<Ty, 6> { static Ty calculate(Ty a, Ty b) { return a ^ b; } };
template <typename Ty> struct OPERATION<Ty, 7> { static Ty calculate(Ty a, Ty b) { return a && b; } };
template <typename Ty> struct OPERATION<Ty, 8> { static Ty calculate(Ty a, Ty b) { return a || b; } };
template <typename Ty> struct OPERATION<Ty, 9> { static Ty calculate(Ty a, Ty b) { return !a ^ !b; } };

static const char * const operation_names[] = { "add", "max", "min", "mul", "and", "or", "xor", "logical_and", "logical_or", "logical_xor" };

template <typename Ty>
bool is_floating_point()
{
    return std::is_floating_point<Ty>::value || std::is_same<Ty, subgroups::cl_half>::value;
}

template <typename Ty, int Which>
void genrand(Ty *x, Ty *t, cl_int *m, int ns, int nw, int ng)
{
    int nj = (nw + ns - 1) / ns;

    for (int k = 0; k < ng; ++k) {
        for (int j = 0; j < nj; ++j) {
            int ii = j * ns;
            int n = ii + ns > nw ? nw - ii : ns;

            for (int i = 0; i < n; ++i) {
                cl_ulong x;
                if ((Which == 0 || Which == 3) && is_floating_point<Ty>()) {
                    // work around different results depending on operation order
                    // by having input with little precision
                    x = genrand_int32(gMTdata) % 64;
                }
                else {
                    x = genrand_int64(gMTdata);
                    if (Which >= 7 && Which <= 9 && ((x >> 32) & 1) == 0)
                        x = 0; // increase probability of false
                }
                t[ii + i] = static_cast<Ty>(x);
            }
        }

        // Now map into work group using map from device
        for (int j = 0; j < nw; ++j) {
            int i = m[4 * j + 1] * ns + m[4 * j];
            x[j] = t[i];
        }

        x += nw;
        m += 4 * nw;
    }
}

// Reduce functions
template <typename Ty, int Which>
struct RED {
    static void gen(Ty *x, Ty *t, cl_int *m, int ns, int nw, int ng)
    {
        genrand<Ty, Which>(x, t, m, ns, nw, ng);
    }

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m, int ns, int nw, int ng)
    {
        int ii, i, j, k, n;
        int nj = (nw + ns - 1)/ns;
        Ty tr, rr;

        log_info("  sub_group_reduce_%s(%s)...\n", operation_names[Which], TypeName<Ty>::val());

        for (k=0; k<ng; ++k) {
            // Map to array indexed to array indexed by local ID and sub group
            for (j=0; j<nw; ++j) {
                i = m[4*j+1]*ns + m[4*j];
                mx[i] = x[j];
                my[i] = y[j];
            }

            for (j=0; j<nj; ++j) {
                ii = j*ns;
                n = ii + ns > nw ? nw - ii : ns;

                // Compute target
                tr = mx[ii];
                for (i=1; i<n; ++i)
                    tr = OPERATION<Ty, Which>::calculate(tr, mx[ii + i]);

                // Check result
                for (i=0; i<n; ++i) {
                    rr = my[ii+i];
                    if (rr != tr) {
                        log_error("ERROR: sub_group_reduce_%s(%s) mismatch for local id %d in sub group %d in group %d\n",
                                   operation_names[Which], TypeName<Ty>::val(), i, j, k);
                        return -1;
                    }
                }
            }

            x += nw;
            y += nw;
            m += 4*nw;
        }

        return 0;
    }
};

// DESCRIPTION:
// Test for reduce non uniform functions
// Which: 0 - add, 1 - max, 2 - min, 3 - mul, 4 - and, 5 - or, 6 - xor, 7 - logical and, 8 - logical or, 9 - logical xor
template <typename Ty, int Which>
struct RED_NU {
    static void gen(Ty *x, Ty *t, cl_int *m, int ns, int nw, int ng)
    {
        genrand<Ty, Which>(x, t, m, ns, nw, ng);
    }

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m, int ns, int nw, int ng)
    {
        int ii, i, j, k, n;
        int nj = (nw + ns - 1) / ns;
        Ty tr, rr;

        log_info("  sub_group_non_uniform_reduce_%s(%s)...\n", operation_names[Which], TypeName<Ty>::val());

        for (k = 0; k < ng; ++k) {
            // Map to array indexed to array indexed by local ID and sub group
            for (j = 0; j < nw; ++j) {
                i = m[4 * j + 1] * ns + m[4 * j];
                mx[i] = x[j];
                my[i] = y[j];
            }

            for (j = 0; j < nj; ++j) {
                ii = j * ns;
                n = ii + ns > nw ? nw - ii : ns;
                if (n > NON_UNIFORM)
                    n = NON_UNIFORM;

                // Compute target
                tr = mx[ii];
                for (i = 1; i < n; ++i) {
                    tr = OPERATION<Ty, Which>::calculate(tr, mx[ii + i]);
                }
                // Check result
                for (i = 0; i < n; ++i) {
                    rr = my[ii + i];
                    if (rr != tr) {
                        log_error("ERROR: sub_group_non_uniform_reduce_%s(%s) mismatch for local id %d in sub group %d in group %d obtained %.17g, expected %.17g\n",
                            operation_names[Which], TypeName<Ty>::val(), i, j, k, static_cast<double>(rr), static_cast<double>(tr));
                        return -1;
                    }
                }
            }

            x += nw;
            y += nw;
            m += 4 * nw;
        }

        return 0;
    }
};

// DESCRIPTION:
// Test for reduce cluster functions
// Which: 0 - add, 1 - max, 2 - min, 3 - mul, 4 - and, 5 - or, 6 - xor, 7 - logical and, 8 - logical or, 9 - logical xor
template <typename Ty, int Which>
struct RED_CLU {
    static void gen(Ty *x, Ty *t, cl_int *m, int ns, int nw, int ng)
    {
        genrand<Ty, Which>(x, t, m, ns, nw, ng);
    }

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m, int ns, int nw, int ng)
    {
        int nj = (nw + ns - 1) / ns;

        log_info("  sub_group_clustered_reduce_%s(%s)...\n", operation_names[Which], TypeName<Ty>::val());

        for (int k = 0; k < ng; ++k) {
            // Map to array indexed to array indexed by local ID and sub group
            for (int j = 0; j < nw; ++j) {
                int i = m[4 * j + 1] * ns + m[4 * j];
                mx[i] = x[j];
                my[i] = y[j];
            }

            for (int j = 0; j < nj; ++j) {
                int ii = j * ns;
                int n = ii + ns > nw ? nw - ii : ns;
                int midx = 4 * ii + 2;
                std::vector<Ty> clusters_results;
                int clusters_counter = ns / CLUSTER_SIZE;
                clusters_results.resize(clusters_counter);

                // Compute target
                Ty tr = mx[ii];
                for (int i = 0; i < n; ++i) {
                    //log_info("i=%d mx=%d my=%d cluster_size=%d\n", i, mx[ii + i], my[ii + i], cluster_size);
                    if (i % CLUSTER_SIZE == 0)
                        tr = mx[ii + i];
                    else
                        tr = OPERATION<Ty, Which>::calculate(tr, mx[ii + i]);
                    clusters_results[i / CLUSTER_SIZE] = tr;
                }

                // Check result
                for (int i = 0; i < n; ++i) {
                    Ty rr = my[ii + i];
                    tr = clusters_results[i / CLUSTER_SIZE];
                    if (rr != tr) {
                        log_error("ERROR: sub_group_clustered_reduce_%s(%s) mismatch for local id %d in sub group %d in group %d obtained %.17g, expected %.17g\n",
                            operation_names[Which], TypeName<Ty>::val(), i, j, k, static_cast<double>(rr), static_cast<double>(tr));
                        return -1;
                    }
                }
            }

            x += nw;
            y += nw;
            m += 4 * nw;
        }

        return 0;
    }
};

// DESCRIPTION:
// Test for scan inclusive non uniform functions
// Which: 0 - add, 1 - max, 2 - min, 3 - mul, 4 - and, 5 - or, 6 - xor, 7 - logical and, 8 - logical or, 9 - logical xor
template <typename Ty, int Which>
struct SCIN_NU {
    static void gen(Ty *x, Ty *t, cl_int *m, int ns, int nw, int ng)
    {
        genrand<Ty, Which>(x, t, m, ns, nw, ng);
    }

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m, int ns, int nw, int ng)
    {
        int ii, i, j, k, n;
        int nj = (nw + ns - 1) / ns;
        Ty tr, rr;

        log_info("  sub_group_non_uniform_scan_inclusive_%s(%s)...\n", operation_names[Which], TypeName<Ty>::val());

        for (k = 0; k < ng; ++k) {      // for each work_group
            // Map to array indexed to array indexed by local ID and sub group
            for (j = 0; j < nw; ++j) {  // inside the work_group
                i = m[4 * j + 1] * ns + m[4 * j];
                mx[i] = x[j];           // read host inputs for work_group
                my[i] = y[j];           // read device outputs for work_group
            }

            for (j = 0; j < nj; ++j) {  // for each subgroup
                ii = j * ns;
                n = ii + ns > nw ? nw - ii : ns;
                // Check result
                tr = TypeIdentity<Ty, Which>::val();
                for (i = 0; i < n && i < NON_UNIFORM; ++i) {   // inside the subgroup
                    tr = OPERATION<Ty, Which>::calculate(tr, mx[ii + i]);
                    rr = my[ii + i];
                    if (rr != tr) {
                        log_error("ERROR: sub_group_non_uniform_scan_inclusive_%s(%s) mismatch for local id %d in sub group %d in group %d obtained %.17g, expected %.17g\n",
                            operation_names[Which], TypeName<Ty>::val(), i, j, k, static_cast<double>(rr), static_cast<double>(tr));
                        return -1;
                    }
                }
            }
            x += nw;
            y += nw;
            m += 4 * nw;
        }

        return 0;
    }
};

// Scan Inclusive functions
template <typename Ty, int Which>
struct SCIN {
    static void gen(Ty *x, Ty *t, cl_int *m, int ns, int nw, int ng)
    {
        genrand<Ty, Which>(x, t, m, ns, nw, ng);
    }

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m, int ns, int nw, int ng)
    {
        int ii, i, j, k, n;
        int nj = (nw + ns - 1)/ns;
        Ty tr, rr;

        log_info("  sub_group_scan_inclusive_%s(%s)...\n",  operation_names[Which], TypeName<Ty>::val());

        for (k=0; k<ng; ++k) {
            // Map to array indexed to array indexed by local ID and sub group
            for (j=0; j<nw; ++j) {
                i = m[4*j+1]*ns + m[4*j];
                mx[i] = x[j];
                my[i] = y[j];
            }

            for (j=0; j<nj; ++j) {
                ii = j*ns;
                n = ii + ns > nw ? nw - ii : ns;

                // Check result
                for (i=0; i<n; ++i) {
                    tr = i == 0 ? mx[ii] : OPERATION<Ty, Which>::calculate(tr, mx[ii + i]);

                    rr = my[ii+i];
                    if (rr != tr) {
                        log_error("ERROR: sub_group_scan_inclusive_%s(%s) mismatch for local id %d in sub group %d in group %d\n",
                                   operation_names[Which], TypeName<Ty>::val(), i, j, k);
                        return -1;
                    }
                }
            }

            x += nw;
            y += nw;
            m += 4*nw;
        }

        return 0;
    }
};

// Scan Exclusive functions
template <typename Ty, int Which>
struct SCEX {
    static void gen(Ty *x, Ty *t, cl_int *m, int ns, int nw, int ng)
    {
        genrand<Ty, Which>(x, t, m, ns, nw, ng);
    }

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m, int ns, int nw, int ng)
    {
        int ii, i, j, k, n;
        int nj = (nw + ns - 1)/ns;
        Ty tr, trt, rr;

        log_info("  sub_group_scan_exclusive_%s(%s)...\n", operation_names[Which], TypeName<Ty>::val());

        for (k=0; k<ng; ++k) {
            // Map to array indexed to array indexed by local ID and sub group
            for (j=0; j<nw; ++j) {
                i = m[4*j+1]*ns + m[4*j];
                mx[i] = x[j];
                my[i] = y[j];
            }

            for (j=0; j<nj; ++j) {
                ii = j*ns;
                n = ii + ns > nw ? nw - ii : ns;

                // Check result
                for (i=0; i<n; ++i) {
                    if (Which == 0) {
                        tr = i == 0 ? TypeIdentity<Ty,Which>::val() : tr + trt;
                    } else if (Which == 1) {
                        tr = i == 0 ? TypeIdentity<Ty,Which>::val() : (trt > tr ? trt : tr);
                    } else {
                        tr = i == 0 ? TypeIdentity<Ty,Which>::val() : (trt > tr ? tr : trt);
                    }
                    trt = mx[ii+i];
                    rr = my[ii+i];

                    if (rr != tr) {
                        log_error("ERROR: sub_group_scan_exclusive_%s(%s) mismatch for local id %d in sub group %d in group %d\n",
                                   operation_names[Which], TypeName<Ty>::val(), i, j, k);
                        return -1;
                    }
                }
            }

            x += nw;
            y += nw;
            m += 4*nw;
        }

        return 0;
    }
};

// Scan Exclusive non uniform functions
// Which: 0 - add, 1 - max, 2 - min, 3 - mul, 4 - and, 5 - or, 6 - xor, 7 - logical and, 8 - logical or, 9 - logical xor
template <typename Ty, int Which>
struct SCEX_NU {
    static void gen(Ty *x, Ty *t, cl_int *m, int ns, int nw, int ng)
    {
        genrand<Ty, Which>(x, t, m, ns, nw, ng);
    }

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m, int ns, int nw, int ng)
    {
        int ii, i, j, k, n;
        int nj = (nw + ns - 1) / ns;
        Ty tr, rr;

        log_info("  sub_group_non_uniform_scan_exclusive_%s(%s)...\n", operation_names[Which], TypeName<Ty>::val());

        for (k = 0; k < ng; ++k) {      // for each work_group
            // Map to array indexed to array indexed by local ID and sub group
            for (j = 0; j < nw; ++j) {  // inside the work_group
                i = m[4 * j + 1] * ns + m[4 * j];
                mx[i] = x[j];           // read host inputs for work_group
                my[i] = y[j];           // read device outputs for work_group
            }
            for (j = 0; j < nj; ++j) {      // for each subgroup
                ii = j * ns;
                n = ii + ns > nw ? nw - ii : ns;
                // Check result
                tr = TypeIdentity<Ty, Which>::val();
                for (i = 0; i < n && i < NON_UNIFORM; ++i) {   // inside the subgroup
                    rr = my[ii + i];

                    if (rr != tr) {
                        log_error("ERROR: sub_group_non_uniform_scan_exclusive_%s(%s) mismatch for local id %d in sub group %d in group %d obtained %.17g, expected %.17g\n",
                            operation_names[Which], TypeName<Ty>::val(), i, j, k, static_cast<double>(rr), static_cast<double>(tr));
                        return -1;
                    }

                    tr = OPERATION<Ty, Which>::calculate(tr, mx[ii + i]);
                }
            }
            x += nw;
            y += nw;
            m += 4 * nw;
        }

        return 0;
    }
};


// DESCRIPTION :
//Which = 0 -  sub_group_broadcast - each work_item registers it's own value. All work_items in subgroup takes one value from only one (any) work_item
//Which = 1 -  sub_group_broadcast_first - same as type 0. All work_items in subgroup takes only one value from only one chosen (the smallest subgroup ID) work_item
//Which = 2 -  sub_group_non_uniform_broadcast - same as type 0 but only 4 work_items from subgroup enter the code (are active)

template <typename Ty, int Which = 0>
struct BC {
    static void gen(Ty *x, Ty *t, cl_int *m, int ns, int nw, int ng)
    {
        int i, ii, j, k, l, n;
        int nj = (nw + ns - 1) / ns;
        int d = ns > 100 ? 100 : ns;

        ii = 0;
        for (k = 0; k < ng; ++k) {      // for each work_group
            for (j = 0; j < nj; ++j) {  // for each subgroup
                ii = j * ns;
                n = ii + ns > nw ? nw - ii : ns;
                //l - calculate subgroup local id from which value will be broadcasted (one the same value for whole subgroup)
                l = (int)(genrand_int32(gMTdata) & 0x7fffffff) % (d > n ? n : d);
                if (Which == 2) {
                    // only 4 work_items in subgroup will be active
                    l = l % 4;
                }

                for (i = 0; i < n; ++i) {
                    int midx = 4 * ii + 4 * i + 2; // index of the third element int the vector.
                    m[midx] = (cl_int)l;           // storing information about broadcasting index - earlier calculated
                    int number;
                    number = (int)(genrand_int32(gMTdata) & 0x7fffffff); // caclute value for broadcasting
                    set_value(t[ii + i], number);
                    //log_info("wg = %d ,sg = %d, inside sg = %d, number == %d, l = %d, midx = %d\n", k, j, i, number, l, midx);
                }
            }

            // Now map into work group using map from device
            for (j = 0; j < nw; ++j) {              // for each element in work_group
                i = m[4 * j + 1] * ns + m[4 * j];   // calculate index as number of subgroup plus subgroup local id
                x[j] = t[i];
            }
            x += nw;
            m += 4 * nw;
        }
    }

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m, int ns, int nw, int ng)
    {
        int ii, i, j, k, l, n;
        int nj = (nw + ns - 1) / ns;
        Ty tr, rr;

        if (Which == 0) {
            log_info("  sub_group_broadcast(%s)...\n", TypeName<Ty>::val());
        } else if (Which == 1) {
            log_info("  sub_group_broadcast_first(%s)...\n", TypeName<Ty>::val());
        } else if (Which == 2) {
            log_info("  sub_group_non_uniform_broadcast(%s)...\n", TypeName<Ty>::val());
        } else {
            log_error("ERROR: Unknown function name...\n");
            return -1;
        }

        for (k = 0; k < ng; ++k) {      // for each work_group
            for (j = 0; j < nw; ++j) {  // inside the work_group
                i = m[4 * j + 1] * ns + m[4 * j];
                mx[i] = x[j];           // read host inputs for work_group
                my[i] = y[j];           // read device outputs for work_group
            }

            for (j = 0; j < nj; ++j) {  // for each subgroup
                ii = j * ns;
                n = ii + ns > nw ? nw - ii : ns;
                int midx = 4 * ii + 2;  // take index of array where info which work_item will be broadcast its value is stored
                l = (int)m[midx];       // take subgroup local id of this work_item
                tr = mx[ii + l];        // take value generated on host for this work_item

                // Check result
                if (Which == 1) {
                    int lowest_active_id = -1;
                    for (i = 0; i < n; ++i) {
                        tr = mx[ii + i];
                        rr = my[ii + i];
                        if (compare(rr, tr)) {  // find work_item id in subgroup which value could be broadcasted
                            lowest_active_id = i;
                            break;
                        }
                    }
                    if (lowest_active_id == -1) {
                        log_error("ERROR: sub_group_broadcast_first(%s) do not found any matching values in sub group %d in group %d\n",
                            TypeName<Ty>::val(), j, k);
                        return -1;
                    }
                    for (i = 0; i < n; ++i) {
                        tr = mx[ii + lowest_active_id]; //  findout if broadcasted value is the same
                        rr = my[ii + i];                //  findout if broadcasted to all
                        if (!compare(rr, tr)) {
                            log_error("ERROR: sub_group_broadcast_first(%s) mismatch for local id %d in sub group %d in group %d\n",
                                TypeName<Ty>::val(), i, j, k);
                        }
                    }
                }
                else {
                    for (i = 0; i < n; ++i) {
                        if (Which == 2 && i >= NON_UNIFORM) {
                            break;   // non uniform case - only first 4 workitems should broadcast. Others are undefined.
                        }
                        rr = my[ii + i];        // read device outputs for work_item in the subgroup
                        if (!compare(rr, tr)) {
                            if (Which == 0) {
                                log_error("ERROR: sub_group_broadcast(%s) mismatch for local id %d in sub group %d in group %d\n",
                                    TypeName<Ty>::val(), i, j, k);
                            }
                            if (Which == 2) {
                                log_error("ERROR: sub_group_non_uniform_broadcast(%s) mismatch for local id %d in sub group %d in group %d\n",
                                    TypeName<Ty>::val(), i, j, k);
                            }
                            return -1;
                        }
                    }
                }
            }
            x += nw;
            y += nw;
            m += 4 * nw;
        }
        return 0;
    }
};

// DESCRIPTION: discover only one elected work item in subgroup - with the lowest subgroup local id
struct ELECT {
    static void gen(cl_int *x, cl_int *t, cl_int *m, int ns, int nw, int ng)
    {
        // no work here needed.
    }

    static int chk(cl_int *x, cl_int *y, cl_int *mx, cl_int *my, cl_int *m, int ns, int nw, int ng)
    {
        int ii, i, j, k, n;
        int nj = (nw + ns - 1)/ns;
        cl_int tr, rr;
        log_info("  sub_group_elect...\n");

        for (k=0; k<ng; ++k) {      // for each work_group
            for (j=0; j<nw; ++j) {  // inside the work_group
                i = m[4*j+1]*ns + m[4*j];
                my[i] = y[j];       // read device outputs for work_group
            }

            for (j=0; j<nj; ++j) {  // for each subgroup
                ii = j*ns;
                n = ii + ns > nw ? nw - ii : ns;
                rr = 0;
                for (i = 0; i < n; ++i) {   // for each work_item in subgroup
                    my[ii + i] > 0 ? rr += 1 : rr += 0; // sum of output values should be 1
                }
                tr = 1; // expectation is that only one elected returned true
                if (rr != tr) {
                    log_error("ERROR: sub_group_elect() mismatch for sub group %d in work group %d. Expected: %d Obtained: %d  \n", j, k, tr, rr);
                    return -1;
                }
            }

            x += nw;
            y += nw;
            m += 4*nw;
        }
        return 0;
    }
};

// Independent forward progress stuff
// Note:
//   Output needs num_groups * NUM_LOC elements
//   local_size must be > NUM_LOC
//   Input needs num_groups * num_sub_groups * (NUM_LOC+1) elements

static inline int
inst(int op, int loc, int val)
{
    return (val << INST_VAL_SHIFT) | (loc << INST_LOC_SHIFT) | (op << INST_OP_SHIFT);
}

void gen_insts(cl_int *x, cl_int *p, int n)
{
    int i, j0, j1;
    int val;
    int ii[NUM_LOC];

    // Create a random permutation of 0...NUM_LOC-1
    ii[0] = 0;
    for (i=1; i<NUM_LOC;++i) {
        j0 = random_in_range(0, i, gMTdata);
        if (j0 != i)
            ii[i] = ii[j0];
        ii[j0] = i;
    }

    // Initialize "instruction pointers"
    memset(p, 0, n*4);

    for (i=0; i<NUM_LOC; ++i) {
        // Randomly choose 2 different sub groups
        // One does a random amount of work, and the other waits for it
        j0 = random_in_range(0, n-1, gMTdata);

        do
            j1 = random_in_range(0, n-1, gMTdata);
        while (j1 == j0);

        // Randomly choose a wait value and assign "instructions"
        val = random_in_range(100, 200 + 10*NUM_LOC, gMTdata);
        x[j0*(NUM_LOC+1) + p[j0]] = inst(INST_COUNT, ii[i], val);
        x[j1*(NUM_LOC+1) + p[j1]] = inst(INST_WAIT,  ii[i], val);
        ++p[j0];
        ++p[j1];
    }

    // Last "inst" for each sub group is END
    for (i=0; i<n; ++i)
        x[i*(NUM_LOC+1) + p[i]] = inst(INST_END, 0, 0);
}

// Execute one group's "instructions"
void run_insts(cl_int *x, cl_int *p, int n)
{
    int i, nend;
    bool scont;
    cl_int loc[NUM_LOC];

    // Initialize result and "instruction pointers"
    memset(loc, 0, sizeof(loc));
    memset(p, 0, 4*n);

    // Repetitively loop over subgroups with each executing "instructions" until blocked
    // The loop terminates when all subgroups have hit the "END instruction"
    do {
        nend = 0;
        for (i=0; i<n; ++i) {
            do {
                cl_int inst = x[i*(NUM_LOC+1) + p[i]];
                cl_int iop = (inst >> INST_OP_SHIFT) & INST_OP_MASK;
                cl_int iloc = (inst >> INST_LOC_SHIFT) & INST_LOC_MASK;
                cl_int ival = (inst >> INST_VAL_SHIFT) & INST_VAL_MASK;
                scont = false;

                switch (iop) {
                case INST_STORE:
                    loc[iloc] = ival;
                    ++p[i];
                    scont = true;
                    break;
                case INST_WAIT:
                    if (loc[iloc] == ival) {
                        ++p[i];
                        scont = true;
                    }
                    break;
                case INST_COUNT:
                    loc[iloc] += ival;
                    ++p[i];
                    scont = true;
                    break;
                case INST_END:
                    ++nend;
                    break;
                }
            } while (scont);
        }
    } while (nend < n);

    // Return result, reusing "p"
    memcpy(p, loc, sizeof(loc));
}


struct IFP {
    static void gen(cl_int *x, cl_int *t, cl_int *, int ns, int nw, int ng)
    {
        int k;
        int nj = (nw + ns - 1) / ns;

        // We need at least 2 sub groups per group for this test
        if (nj == 1)
            return;

        for (k=0; k<ng; ++k) {
            gen_insts(x, t, nj);
            x += nj * (NUM_LOC+1);
        }
    }

    static int chk(cl_int *x, cl_int *y, cl_int *t, cl_int *, cl_int *, int ns, int nw, int ng)
    {
        int i, k;
        int nj = (nw + ns - 1) / ns;

        // We need at least 2 sub groups per group for this tes
        if (nj == 1)
            return 0;

        log_info("  independent forward progress...\n");

        for (k=0; k<ng; ++k) {
            run_insts(x, t, nj);
            for (i=0; i<NUM_LOC; ++i) {
                if (t[i] != y[i]) {
                    log_error("ERROR: mismatch at element %d in work group %d\n", i, k);
                    return -1;
                }
            }
            x += nj * (NUM_LOC+1);
            y += NUM_LOC;
        }

        return 0;
    }
};

template <typename Ty>
struct BALLOT {
    static void gen(Ty *x, Ty *t, cl_int *m, int ns, int nw, int ng)
    {
        int i, ii, j, k, n;
        int nj = (nw + ns - 1) / ns;
        int e;
        ii = 0;
        for (k = 0; k < ng; ++k) {          // for each work_group
            for (j = 0; j < nj; ++j) {      // for each subgroup
                ii = j * ns;
                n = ii + ns > nw ? nw - ii : ns;
                e = (int)(genrand_int32(gMTdata) % 3);

                // Initialize data matrix indexed by local id and sub group id
                switch (e) {
                case 0:
                    memset(&t[ii], 0, n * sizeof(Ty));
                    break;
                case 1:
                    memset(&t[ii], 0, n * sizeof(Ty));
                    i = (int)(genrand_int32(gMTdata) % (cl_uint)n);
                    set_value(t[ii + i], 41);
                    break;
                case 2:
                    memset(&t[ii], 0xff, n * sizeof(Ty));
                    break;
                }
            }

            // Now map into work group using map from device
            for (j = 0; j < nw; ++j) {
                i = m[4 * j + 1] * ns + m[4 * j];
                x[j] = t[i];
            }

            x += nw;
            m += 4 * nw;
        }
    }

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m, int ns, int nw, int ng)
    {
        int ii, i, j, k, n;
        int nj = (nw + ns - 1) / ns;
        cl_uint4 tr, rr;

        log_info("  sub_group_ballot...\n");

        for (k = 0; k < ng; ++k) {          // for each work_group
            // Map to array indexed to array indexed by local ID and sub group
            for (j = 0; j < nw; ++j) {      // inside the work_group
                i = m[4 * j + 1] * ns + m[4 * j];
                mx[i] = x[j];               // read host inputs for work_group
                my[i] = y[j];               // read host inputs for work_group
            }

            for (j = 0; j < nj; ++j) {      // for each subgroup
                ii = j * ns;
                n = ii + ns > nw ? nw - ii : ns;

                // Check result
                tr = { 0, 0, 0, 0 };
                for (i = 0; i < n; ++i) {   // for each subgroup
                    int bit_value = 0;
                    (mx[ii + i].s0 != 0) ? bit_value = 1 : bit_value = 0;
                    if (i < 32)
                        tr.s0 = set_bit(bit_value, tr.s0, i);
                    if (i >= 32 && i < 64)
                        tr.s1 = set_bit(bit_value, tr.s1, i);
                    if (i >= 64 && i < 96)
                        tr.s2 = set_bit(bit_value, tr.s2, i);
                    if (i >= 96 && i < 128)
                        tr.s3 = set_bit(bit_value, tr.s3, i);
                }
                rr = my[ii];
                if (!compare(rr, tr)) {
                    log_error("ERROR: sub_group_ballot mismatch for local id %d in sub group %d in group %d obtained {%d, %d, %d, %d}, expected {%d, %d, %d, %d}\n", i, j, k, rr.s0, rr.s1, rr.s2, rr.s3, tr.s0, tr.s1, tr.s2, tr.s3);
                    return -1;
                }
            }
            x += nw;
            y += nw;
            m += 4 * nw;
        }

        return 0;
    }
};

// DESCRIPTION:
// Test for inverse/bit extract ballot functions
// Which : 0 - inverse , 1 - bit extract
template <typename Ty, int Which>
struct BALLOT2 {
    static void gen(Ty *x, Ty *t, cl_int *m, int ns, int nw, int ng)
    {
        int i, ii, j, k, n, l;
        int nj = (nw + ns - 1) / ns;
        int e;
        ii = 0;
        int d = ns > 100 ? 100 : ns;
        for (k = 0; k < ng; ++k) {          // for each work_group
            for (j = 0; j < nj; ++j) {      // for each subgroup
                ii = j * ns;
                n = ii + ns > nw ? nw - ii : ns;
                l = (int)(genrand_int32(gMTdata) & 0x7fffffff) % (d > n ? n : d); // rand index to bit extract
                e = (int)(genrand_int32(gMTdata) % 3);
                for (i = 0; i < n; ++i) {
                    int midx = 4 * ii + 4 * i + 2; // index of the third element int the vector.
                    m[midx] = (cl_int)l;           // storing information about index to bit extract
                }
                // Initialize data matrix indexed by local id and sub group id
                switch (e) {
                case 0:
                    memset(&t[ii], 0, n * sizeof(Ty));
                    break;
                case 1:
                    // inverse ballot requires that value must be the same for all active invocations
                    if (Which == 0)
                        memset(&t[ii], (int)(genrand_int32(gMTdata)) & 0xff, n * sizeof(Ty));
                    else {
                        memset(&t[ii], 0, n * sizeof(Ty));
                        i = (int)(genrand_int32(gMTdata) % (cl_uint)n);
                        set_value(t[ii + i], 41);
                    }
                    break;
                case 2:
                    memset(&t[ii], 0xff, n * sizeof(Ty));
                    break;
                }
            }

            // Now map into work group using map from device
            for (j = 0; j < nw; ++j) {
                i = m[4 * j + 1] * ns + m[4 * j];
                x[j] = t[i];
            }

            x += nw;
            m += 4 * nw;
        }
    }

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m, int ns, int nw, int ng)
    {
        int ii, i, j, k, n, l;
        int nj = (nw + ns - 1) / ns;
        cl_uint4 tr, rr;

        log_info("  sub_group_%s(%s)...\n", Which == 0 ? "inverse_ballot" : (Which == 1 ? "ballot_bit_extract" : Which == 2 ? "" : ""), TypeName<Ty>::val());

        for (k = 0; k < ng; ++k) {          // for each work_group
            // Map to array indexed to array indexed by local ID and sub group
            for (j = 0; j < nw; ++j) {      // inside the work_group
                i = m[4 * j + 1] * ns + m[4 * j];
                mx[i] = x[j];               // read host inputs for work_group
                my[i] = y[j];               // read host inputs for work_group
            }

            for (j = 0; j < nj; ++j) {      // for each subgroup
                ii = j * ns;
                n = ii + ns > nw ? nw - ii : ns;
                int midx = 4 * ii + 2;  // take index of array where info which work_item will be broadcast its value is stored
                l = (int)m[midx];       // take subgroup local id of this work_item
                // Check result

                for (i = 0; i < n; ++i) {   // for each subgroup
                    int bit_value = 0;
                    int bit_mask = 1 << ((Which == 0) ? i : l % 32); //from which value of bitfield bit verification will be done

                    if (i < 32)
                        (mx[ii + i].s0 & bit_mask) > 0 ? bit_value = 1 : bit_value = 0;
                    if (i >= 32 && i < 64)
                        (mx[ii + i].s1 & bit_mask) > 0 ? bit_value = 1 : bit_value = 0;
                    if (i >= 64 && i < 96)
                        (mx[ii + i].s2 & bit_mask) > 0 ? bit_value = 1 : bit_value = 0;
                    if (i >= 96 && i < 128)
                        (mx[ii + i].s3 & bit_mask) > 0 ? bit_value = 1 : bit_value = 0;

                    bit_value == 1 ? tr = { 1, 0, 0, 0 }: tr = { 0, 0 , 0, 0 };

                    rr = my[ii + i];
                    if (!compare(rr, tr)) {
                        log_error("ERROR: sub_group_%s mismatch for local id %d in sub group %d in group %d obtained {%d, %d, %d, %d}, expected {%d, %d, %d, %d}\n", Which == 0 ? "inverse_ballot" : (Which == 1 ? "ballot_bit_extract" : (Which == 2 ? "" : "")), i, j, k, rr.s0, rr.s1, rr.s2, rr.s3, tr.s0, tr.s1, tr.s2, tr.s3);
                        return -1;
                    }
                }
            }
            x += nw;
            y += nw;
            m += 4 * nw;
        }

        return 0;
    }
};


// DESCRIPTION:
// Test for bit count/inclusive and exclusive scan/ find lsb msb ballot function
// Which : 0 - bit count , 1 - inclusive scan, 2 - exclusive scan , 3 - find lsb, 4 - find msb
template <typename Ty, int Which>
struct BALLOT3 {
    static void gen(Ty *x, Ty *t, cl_int *m, int ns, int nw, int ng)
    {
        int i, ii, j, k, n;
        int nj = (nw + ns - 1) / ns;
        int e;
        ii = 0;
        for (k = 0; k < ng; ++k) {          // for each work_group
            for (j = 0; j < nj; ++j) {      // for each subgroup
                ii = j * ns;
                n = ii + ns > nw ? nw - ii : ns;
                if (Which >= 0 && Which <= 2) {
                    // Initialize data matrix indexed by local id and sub group id
                    e = (int)(genrand_int32(gMTdata) % 3);
                    switch (e) {
                    case 0:
                        memset(&t[ii], 0, n * sizeof(Ty));
                        break;
                    case 1:
                        memset(&t[ii], 0, n * sizeof(Ty));
                        i = (int)(genrand_int32(gMTdata) % (cl_uint)n);
                        set_value(t[ii + i], 41);
                        break;
                    case 2:
                        memset(&t[ii], 0xff, n * sizeof(Ty));
                        break;
                    }
                }
                else {
                    // Regarding to the spec, find lsb and find msb result is undefined behavior
                    // if input value is zero, so generate only non-zero values.
                    e = (int)(genrand_int32(gMTdata) % 2);
                    switch (e) {
                    case 0:
                        memset(&t[ii], 0xff, n * sizeof(Ty));
                        break;
                    case 1:
                        char x = (genrand_int32(gMTdata)) & 0xff;
                        memset(&t[ii], x ? x : 1, n * sizeof(Ty));
                        break;
                    }
                }
            }

            // Now map into work group using map from device
            for (j = 0; j < nw; ++j) {
                i = m[4 * j + 1] * ns + m[4 * j];
                x[j] = t[i];
            }

            x += nw;
            m += 4 * nw;
        }
    }

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m, int ns, int nw, int ng)
    {
        int ii, i, j, k, n;
        int nj = (nw + ns - 1) / ns;
        cl_uint4 tr, rr;

        log_info("  sub_group_ballot_%s(%s)...\n", Which == 0 ? "bit_count" : (Which == 1 ? "inclusive_scan" : (Which == 2 ? "exclusive_scan" : (Which == 3 ? "find_lsb" : "find_msb"))), TypeName<Ty>::val());

        for (k = 0; k < ng; ++k) {          // for each work_group
            // Map to array indexed to array indexed by local ID and sub group
            for (j = 0; j < nw; ++j) {      // inside the work_group
                i = m[4 * j + 1] * ns + m[4 * j];
                mx[i] = x[j];               // read host inputs for work_group
                my[i] = y[j];               // read host inputs for work_group
            }

            for (j = 0; j < nj; ++j) {      // for each subgroup
                ii = j * ns;
                n = ii + ns > nw ? nw - ii : ns;
                // Check result
                tr = {0, 0, 0, 0};
                int min_id = -1;
                int max_id = -1;
                for (i = 0; i < n; ++i) {   // for each subgroup
                    int bit_value = 0;
                    int bit_mask = 1 << (i % 32); //from which value of bitfield bit verification will be done
                    bool inc = !(Which == 2 && i == 0);

                    if (i < 32)
                        (mx[ii + i].s0 & bit_mask) > 0 ? bit_value = 1 : bit_value = 0;
                    if (i >= 32 && i < 64)
                        (mx[ii + i].s1 & bit_mask) > 0 ? bit_value = 1 : bit_value = 0;
                    if (i >= 64 && i < 96)
                        (mx[ii + i].s2 & bit_mask) > 0 ? bit_value = 1 : bit_value = 0;
                    if (i >= 96 && i < 128)
                        (mx[ii + i].s3 & bit_mask) > 0 ? bit_value = 1 : bit_value = 0;
                    if (bit_value == 1) {
                        if (min_id == -1) {
                            min_id = i;
                        }
                        if (max_id == -1 || max_id < i) {
                            max_id = i;
                        }
                    }
                    (inc && bit_value == 1) ? tr = { tr.s0 + 1, 0, 0, 0 } : tr = { tr.s0, 0 , 0, 0 };

                    rr = my[ii + i];
                    if (Which == 0 && i == n - 1) {
                        if (!compare(rr, tr)) {
                            log_error("ERROR: sub_group_ballot_bit_count mismatch for local id %d in sub group %d in group %d obtained {%d, %d, %d, %d}, expected {%d, %d, %d, %d}\n", i, j, k, rr.s0, rr.s1, rr.s2, rr.s3, tr.s0, tr.s1, tr.s2, tr.s3);
                            return -1;
                        }
                    }
                    else if (Which == 1) {
                        if (!compare(rr, tr)) {
                            log_error("ERROR: sub_group_ballot_inclusive_scan mismatch for local id %d in sub group %d in group %d obtained {%d, %d, %d, %d}, expected {%d, %d, %d, %d}\n", i, j, k, rr.s0, rr.s1, rr.s2, rr.s3, tr.s0, tr.s1, tr.s2, tr.s3);
                            return -1;
                        }
                    }
                    else if (Which == 2) {
                        if (!compare(rr, tr)) {
                            log_error("ERROR: sub_group_ballot_exclusive_scan mismatch for local id %d in sub group %d in group %d obtained {%d, %d, %d, %d}, expected {%d, %d, %d, %d}\n", i, j, k, rr.s0, rr.s1, rr.s2, rr.s3, tr.s0, tr.s1, tr.s2, tr.s3);
                            return -1;
                        }
                    }
                    if ((Which == 3 || Which == 4) && i == n - 1) {
                        if (min_id == -1) {
                            min_id = 0;
                        }
                        if (max_id == -1) {
                            max_id = 0;
                        }
                        Which == 3 ? tr = { (cl_uint)min_id, 0, 0, 0 } : tr = { (cl_uint)max_id, 0, 0, 0 };
                        if (!compare(rr, tr)) {
                            log_error("ERROR: sub_group_ballot_%s mismatch for local id %d in sub group %d in group %d obtained {%d, %d, %d, %d}, expected {%d, %d, %d, %d}\n", Which == 3 ? "find_lsb" : "find_msb", i, j, k, rr.s0, rr.s1, rr.s2, rr.s3, tr.s0, tr.s1, tr.s2, tr.s3);
                            return -1;
                        }
                    }

                }
            }
            x += nw;
            y += nw;
            m += 4 * nw;
        }

        return 0;
    }
};
// Entry point from main
int
test_work_group_functions(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    int error = false;

    // Adjust these individually below if desired/needed
#define G 2000
#define L 200
    std::vector<std::string> required_extensions;
    error |= test<int, AA<0>, G, L>::run(device, context, queue, num_elements, "test_any", any_source);
    error |= test<int, AA<1>, G, L>::run(device, context, queue, num_elements, "test_all", all_source);

    // error |= test<cl_half, BC<cl_half>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source);
    error |= test<cl_uint, BC<cl_uint>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source);
    error |= test<cl_int, BC<cl_int>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source);
    error |= test<cl_ulong, BC<cl_ulong>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source);
    error |= test<cl_long, BC<cl_long>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source);
    error |= test<float, BC<float>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source);
    error |= test<double, BC<double>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source);

    // error |= test<cl_half, RED<cl_half,0>, G, L>::run(device, context, queue, num_elements, "test_redadd", redadd_source);
    error |= test<cl_uint, RED<cl_uint,0>, G, L>::run(device, context, queue, num_elements, "test_redadd", redadd_source);
    error |= test<cl_int, RED<cl_int,0>, G, L>::run(device, context, queue, num_elements, "test_redadd", redadd_source);
    error |= test<cl_ulong, RED<cl_ulong,0>, G, L>::run(device, context, queue, num_elements, "test_redadd", redadd_source);
    error |= test<cl_long, RED<cl_long,0>, G, L>::run(device, context, queue, num_elements, "test_redadd", redadd_source);
    error |= test<float, RED<float,0>, G, L>::run(device, context, queue, num_elements, "test_redadd", redadd_source);
    error |= test<double, RED<double,0>, G, L>::run(device, context, queue, num_elements, "test_redadd", redadd_source);

    // error |= test<cl_half, RED<cl_half,1>, G, L>::run(device, context, queue, num_elements, "test_redmax", redmax_source);
    error |= test<cl_uint, RED<cl_uint,1>, G, L>::run(device, context, queue, num_elements, "test_redmax", redmax_source);
    error |= test<cl_int, RED<cl_int,1>, G, L>::run(device, context, queue, num_elements, "test_redmax", redmax_source);
    error |= test<cl_ulong, RED<cl_ulong,1>, G, L>::run(device, context, queue, num_elements, "test_redmax", redmax_source);
    error |= test<cl_long, RED<cl_long,1>, G, L>::run(device, context, queue, num_elements, "test_redmax", redmax_source);
    error |= test<float, RED<float,1>, G, L>::run(device, context, queue, num_elements, "test_redmax", redmax_source);
    error |= test<double, RED<double,1>, G, L>::run(device, context, queue, num_elements, "test_redmax", redmax_source);

    // error |= test<cl_half, RED<cl_half,2>, G, L>::run(device, context, queue, num_elements, "test_redmin", redmin_source);
    error |= test<cl_uint, RED<cl_uint,2>, G, L>::run(device, context, queue, num_elements, "test_redmin", redmin_source);
    error |= test<cl_int, RED<cl_int,2>, G, L>::run(device, context, queue, num_elements, "test_redmin", redmin_source);
    error |= test<cl_ulong, RED<cl_ulong,2>, G, L>::run(device, context, queue, num_elements, "test_redmin", redmin_source);
    error |= test<cl_long, RED<cl_long,2>, G, L>::run(device, context, queue, num_elements, "test_redmin", redmin_source);
    error |= test<float, RED<float,2>, G, L>::run(device, context, queue, num_elements, "test_redmin", redmin_source);
    error |= test<double, RED<double,2>, G, L>::run(device, context, queue, num_elements, "test_redmin", redmin_source);

    // error |= test<cl_half, SCIN<cl_half,0>, G, L>::run(device, context, queue, num_elements, "test_scinadd", scinadd_source);
    error |= test<cl_uint, SCIN<cl_uint,0>, G, L>::run(device, context, queue, num_elements, "test_scinadd", scinadd_source);
    error |= test<cl_int, SCIN<cl_int,0>, G, L>::run(device, context, queue, num_elements, "test_scinadd", scinadd_source);
    error |= test<cl_ulong, SCIN<cl_ulong,0>, G, L>::run(device, context, queue, num_elements, "test_scinadd", scinadd_source);
    error |= test<cl_long, SCIN<cl_long,0>, G, L>::run(device, context, queue, num_elements, "test_scinadd", scinadd_source);
    error |= test<float, SCIN<float,0>, G, L>::run(device, context, queue, num_elements, "test_scinadd", scinadd_source);
    error |= test<double, SCIN<double,0>, G, L>::run(device, context, queue, num_elements, "test_scinadd", scinadd_source);

    // error |= test<cl_half, SCIN<cl_half,1>, G, L>::run(device, context, queue, num_elements, "test_scinmax", scinmax_source);
    error |= test<cl_uint, SCIN<cl_uint,1>, G, L>::run(device, context, queue, num_elements, "test_scinmax", scinmax_source);
    error |= test<cl_int, SCIN<cl_int,1>, G, L>::run(device, context, queue, num_elements, "test_scinmax", scinmax_source);
    error |= test<cl_ulong, SCIN<cl_ulong,1>, G, L>::run(device, context, queue, num_elements, "test_scinmax", scinmax_source);
    error |= test<cl_long, SCIN<cl_long,1>, G, L>::run(device, context, queue, num_elements, "test_scinmax", scinmax_source);
    error |= test<float, SCIN<float,1>, G, L>::run(device, context, queue, num_elements, "test_scinmax", scinmax_source);
    error |= test<double, SCIN<double,1>, G, L>::run(device, context, queue, num_elements, "test_scinmax", scinmax_source);

    // error |= test<cl_half, SCIN<cl_half,2>, G, L>::run(device, context, queue, num_elements, "test_scinmin", scinmin_source);
    error |= test<cl_uint, SCIN<cl_uint,2>, G, L>::run(device, context, queue, num_elements, "test_scinmin", scinmin_source);
    error |= test<cl_int, SCIN<cl_int,2>, G, L>::run(device, context, queue, num_elements, "test_scinmin", scinmin_source);
    error |= test<cl_ulong, SCIN<cl_ulong,2>, G, L>::run(device, context, queue, num_elements, "test_scinmin", scinmin_source);
    error |= test<cl_long, SCIN<cl_long,2>, G, L>::run(device, context, queue, num_elements, "test_scinmin", scinmin_source);
    error |= test<float, SCIN<float,2>, G, L>::run(device, context, queue, num_elements, "test_scinmin", scinmin_source);
    error |= test<double, SCIN<double,2>, G, L>::run(device, context, queue, num_elements, "test_scinmin", scinmin_source);

    // error |= test<cl_half, SCEX<cl_half,0>, G, L>::run(device, context, queue, num_elements, "test_scexadd", scexadd_source);
    error |= test<cl_uint, SCEX<cl_uint,0>, G, L>::run(device, context, queue, num_elements, "test_scexadd", scexadd_source);
    error |= test<cl_int, SCEX<cl_int,0>, G, L>::run(device, context, queue, num_elements, "test_scexadd", scexadd_source);
    error |= test<cl_ulong, SCEX<cl_ulong,0>, G, L>::run(device, context, queue, num_elements, "test_scexadd", scexadd_source);
    error |= test<cl_long, SCEX<cl_long,0>, G, L>::run(device, context, queue, num_elements, "test_scexadd", scexadd_source);
    error |= test<float, SCEX<float,0>, G, L>::run(device, context, queue, num_elements, "test_scexadd", scexadd_source);
    error |= test<double, SCEX<double,0>, G, L>::run(device, context, queue, num_elements, "test_scexadd", scexadd_source);

    // error |= test<cl_half, SCEX<cl_half,1>, G, L>::run(device, context, queue, num_elements, "test_scexmax", scexmax_source);
    error |= test<cl_uint, SCEX<cl_uint,1>, G, L>::run(device, context, queue, num_elements, "test_scexmax", scexmax_source);
    error |= test<cl_int, SCEX<cl_int,1>, G, L>::run(device, context, queue, num_elements, "test_scexmax", scexmax_source);
    error |= test<cl_ulong, SCEX<cl_ulong,1>, G, L>::run(device, context, queue, num_elements, "test_scexmax", scexmax_source);
    error |= test<cl_long, SCEX<cl_long,1>, G, L>::run(device, context, queue, num_elements, "test_scexmax", scexmax_source);
    error |= test<float, SCEX<float,1>, G, L>::run(device, context, queue, num_elements, "test_scexmax", scexmax_source);
    error |= test<double, SCEX<double,1>, G, L>::run(device, context, queue, num_elements, "test_scexmax", scexmax_source);

    // error |= test<cl_half, SCEX<cl_half,2>, G, L>::run(device, context, queue, num_elements, "test_scexmin", scexmin_source);
    error |= test<cl_uint, SCEX<cl_uint,2>, G, L>::run(device, context, queue, num_elements, "test_scexmin", scexmin_source);
    error |= test<cl_int, SCEX<cl_int,2>, G, L>::run(device, context, queue, num_elements, "test_scexmin", scexmin_source);
    error |= test<cl_ulong, SCEX<cl_ulong,2>, G, L>::run(device, context, queue, num_elements, "test_scexmin", scexmin_source);
    error |= test<cl_long, SCEX<cl_long,2>, G, L>::run(device, context, queue, num_elements, "test_scexmin", scexmin_source);
    error |= test<float, SCEX<float,2>, G, L>::run(device, context, queue, num_elements, "test_scexmin", scexmin_source);
    error |= test<double, SCEX<double,2>, G, L>::run(device, context, queue, num_elements, "test_scexmin", scexmin_source);

    error |= test<cl_int, IFP, G, L>::run(device, context, queue, num_elements, "test_ifp", ifp_source, NUM_LOC + 1);
    error |= test<subgroups::cl_half, BC<subgroups::cl_half>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source, 0, required_extensions);

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

    //*************************************************************************

    return error;
}

