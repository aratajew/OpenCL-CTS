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

                    bit_value == 1 ? tr = { 1, 0, 0, 0 } : tr = { 0, 0 , 0, 0 };

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

    static bs128 getImportantBits(int sub_group_local_id, int sub_group_size) {
        bs128 mask;
        if (Which == 0 || Which == 3 || Which == 4) {
            for (cl_uint i = 0; i < sub_group_size; ++i)
                mask.set(i);
        }
        else if (Which == 1 || Which == 2) {
            for (cl_uint i = 0; i <= sub_group_local_id; ++i)
                mask.set(i);
            if (Which == 2)
                mask.reset(sub_group_local_id);
        }
        return mask;
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
                for (i = 0; i < n; ++i) {   // for each subgroup
                    bs128 bs;
                    // convert cl_uint4 input into std::bitset<128>
                    bs |= bs128(mx[ii + i].s0) | (bs128(mx[ii + i].s1) << 32) | (bs128(mx[ii + i].s2) << 64) | (bs128(mx[ii + i].s3) << 96);
                    bs &= getImportantBits(i, n);

                    rr = my[ii + i];
                    if (Which == 0) {
                        tr.s0 = bs.count();
                        if (!compare(rr, tr)) {
                            log_error("ERROR: sub_group_ballot_bit_count mismatch for local id %d in sub group %d in group %d obtained {%d, %d, %d, %d}, expected {%d, %d, %d, %d}\n", i, j, k, rr.s0, rr.s1, rr.s2, rr.s3, tr.s0, tr.s1, tr.s2, tr.s3);
                            return -1;
                        }
                    }
                    else if (Which == 1) {
                        tr.s0 = bs.count();
                        if (!compare(rr, tr)) {
                            log_error("ERROR: sub_group_ballot_inclusive_scan mismatch for local id %d in sub group %d in group %d obtained {%d, %d, %d, %d}, expected {%d, %d, %d, %d}\n", i, j, k, rr.s0, rr.s1, rr.s2, rr.s3, tr.s0, tr.s1, tr.s2, tr.s3);
                            return -1;
                        }
                    }
                    else if (Which == 2) {
                        tr.s0 = bs.count();
                        if (!compare(rr, tr)) {
                            log_error("ERROR: sub_group_ballot_exclusive_scan mismatch for local id %d in sub group %d in group %d obtained {%d, %d, %d, %d}, expected {%d, %d, %d, %d}\n", i, j, k, rr.s0, rr.s1, rr.s2, rr.s3, tr.s0, tr.s1, tr.s2, tr.s3);
                            return -1;
                        }
                    }
                    else if (Which == 3) {
                        for (int id = 0; id < n; ++id) {
                            if (bs.test(id)) {
                                tr.s0 = id;
                                break;
                            }
                        }
                        if (!compare(rr, tr)) {
                            log_error("ERROR: sub_group_ballot_find_lsb mismatch for local id %d in sub group %d in group %d obtained {%d, %d, %d, %d}, expected {%d, %d, %d, %d}\n", i, j, k, rr.s0, rr.s1, rr.s2, rr.s3, tr.s0, tr.s1, tr.s2, tr.s3);
                            return -1;
                        }
                    }
                    if (Which == 4) {
                        for (int id = n - 1; id >= 0; --id) {
                            if (bs.test(id)) {
                                tr.s0 = id;
                                break;
                            }
                        }
                        if (!compare(rr, tr)) {
                            log_error("ERROR: sub_group_ballot_find_msb mismatch for local id %d in sub group %d in group %d obtained {%d, %d, %d, %d}, expected {%d, %d, %d, %d}\n", i, j, k, rr.s0, rr.s1, rr.s2, rr.s3, tr.s0, tr.s1, tr.s2, tr.s3);
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
                    cl_uint max_sub_group_size = m[midx + 2];
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

