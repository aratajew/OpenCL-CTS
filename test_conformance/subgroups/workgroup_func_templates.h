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
#ifndef WORKGROUPFUNCTEMPLATES_H
#define WORKGROUPFUNCTEMPLATES_H
#include "typeWrappers.h"
#include <bitset>

#define CLUSTER_SIZE 4
#define CLUSTER_SIZE_STR "4"
#define NON_UNIFORM 4

// Adjust these individually below if desired/needed
#define G 2000
#define L 200

inline cl_uint set_bit(cl_uint bit_value, cl_uint number, cl_uint position) {
    number ^= (-(bit_value) ^ number) & (1UL << position);
    return number;
}
inline cl_uint4 generate_bit_mask(cl_uint subgroup_local_id, std::string mask_type, cl_uint max_sub_group_size) {
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



#endif
