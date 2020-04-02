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
#ifndef WORKGROUPKERNELSOURCES_H
#define WORKGROUPKERNELSOURCES_H
#include "subhelpers.h"

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


#endif
