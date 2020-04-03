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

#define CLUSTER_SIZE 4
#define CLUSTER_SIZE_STR "4"

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

int
test_work_group_functions_clustered_reduce(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    int error = false;
    std::vector<std::string> required_extensions;
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

    return error;
}

