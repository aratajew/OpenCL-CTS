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
#include "workgroup_common_kernels.h"
#include "harness/typeWrappers.h"

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

// DESCRIPTION: discover only one elected work item in subgroup - with the lowest subgroup local id
struct ELECT {
    static void gen(cl_int *x, cl_int *t, cl_int *m, int ns, int nw, int ng)
    {
        // no work here needed.
    }

    static int chk(cl_int *x, cl_int *y, cl_int *mx, cl_int *my, cl_int *m, int ns, int nw, int ng)
    {
        int ii, i, j, k, n;
        int nj = (nw + ns - 1) / ns;
        cl_int tr, rr;
        log_info("  sub_group_elect...\n");

        for (k = 0; k < ng; ++k) {      // for each work_group
            for (j = 0; j < nw; ++j) {  // inside the work_group
                i = m[4 * j + 1] * ns + m[4 * j];
                my[i] = y[j];       // read device outputs for work_group
            }

            for (j = 0; j < nj; ++j) {  // for each subgroup
                ii = j * ns;
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
            m += 4 * nw;
        }
        return 0;
    }
};

static const char * elect_source =
"__kernel void test_elect(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    Type x = in[gid];\n"
"    int am_i_elected = sub_group_elect();\n"
"    out[gid] = am_i_elected;\n" //one in subgroup true others false.
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

