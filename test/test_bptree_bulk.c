/**
 * @file test_bptree_bulk.c
 * @brief Unit tests for bulk B+ tree construction (bptree_bulk.h).
 *
 * Covers ordered-array construction, empty input, rejection of duplicate or
 * descending keys, equivalence with repeated bptree_put() calls, invalid node
 * capacity, and custom comparators. String key mode is supported through the
 * same BPTREE_KEY_TYPE_STRING switch used by test_bptree.c.
 *
 * @version 0.5.1
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Allocation failure injection. The wrappers are defined before the library
 * header so that the implementation's malloc/calloc/aligned_alloc calls pass
 * through them. A value of -1 disables injection. */
static int fail_after = -1;
static int allocation_calls = 0;

static void* test_malloc(size_t size) {
    if (fail_after == 0) return NULL;
    if (fail_after > 0) fail_after--;
    allocation_calls++;
    return malloc(size);
}

static void* test_calloc(size_t count, size_t size) {
    if (fail_after == 0) return NULL;
    if (fail_after > 0) fail_after--;
    allocation_calls++;
    return calloc(count, size);
}

static void* test_aligned_alloc(size_t alignment, size_t size) {
    if (fail_after == 0) return NULL;
    if (fail_after > 0) fail_after--;
    allocation_calls++;
    return aligned_alloc(alignment, size);
}

#define malloc test_malloc
#define calloc test_calloc
#define aligned_alloc test_aligned_alloc
#define BPTREE_IMPLEMENTATION
#include "bptree_bulk.h"
#undef malloc
#undef calloc
#undef aligned_alloc

#define ASSERT(cond, ...)                                         \
    do {                                                          \
        if (!(cond)) {                                            \
            fprintf(stderr, "FAIL(%s:%d): ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                         \
            fprintf(stderr, "\n");                                \
            tests_failed++;                                       \
        }                                                         \
    } while (0)

#define RUN_TEST(test)                             \
    do {                                           \
        fprintf(stderr, "== Running %s\n", #test); \
        test();                                    \
        tests_run++;                               \
    } while (0)

static int tests_run = 0;
static int tests_failed = 0;

/* ---------------------------------------------------------------------------
 * Key and value helpers
 * ------------------------------------------------------------------------- */

#ifdef BPTREE_KEY_TYPE_STRING
static bptree_key_t make_key(const int n) {
    bptree_key_t key;
    memset(&key, 0, sizeof(key));
    snprintf(key.data, BPTREE_KEY_SIZE, "%05d", n);
    return key;
}

static int string_key_compare(const bptree_key_t* a, const bptree_key_t* b) {
    return strcmp(a->data, b->data);
}

#define TEST_COMPARE string_key_compare
#else
static bptree_key_t make_key(const int n) { return (bptree_key_t)n; }

#define TEST_COMPARE NULL
#endif

/** @brief Builds an ascending key array of @p count entries with a stride of 2. */
static void fill_ascending_keys(bptree_key_t* keys, const int count) {
    for (int i = 0; i < count; ++i) {
        keys[i] = make_key(i * 2);
    }
}

/** @brief Compares two keys with the same semantics as the tree under test. */
static int compare_keys(const bptree_key_t* a, const bptree_key_t* b) {
#ifdef BPTREE_KEY_TYPE_STRING
    return strcmp(a->data, b->data);
#else
    return (*a < *b) ? -1 : ((*a > *b) ? 1 : 0);
#endif
}

/** @brief Fills a value array so each value encodes its key index. */
static void fill_values(bptree_value_t* values, const int count) {
    for (int i = 0; i < count; ++i) {
        values[i] = (bptree_value_t)(intptr_t)(i * 3 + 1);
    }
}

/** @brief Checks that @p value matches the encoded value for key index @p i. */
static bool value_matches(const bptree_value_t value, const int i) {
    return value == (bptree_value_t)(intptr_t)(i * 3 + 1);
}

/** @brief Verifies count, height, per-key lookup, and iteration order. */
static void verify_tree_matches_input(const bptree* tree, const bptree_key_t* keys,
                                      const bptree_value_t* values, const int count) {
    ASSERT(bptree_count(tree) == count, "count %d != %d", bptree_count(tree), count);
    ASSERT(bptree_height(tree) >= 1, "height %d < 1", bptree_height(tree));
    ASSERT(bptree_check_invariants(tree), "invariants failed for count %d", count);

    for (int i = 0; i < count; ++i) {
        bptree_value_t out;
        ASSERT(bptree_get(tree, &keys[i], &out) == BPTREE_OK, "get failed for index %d", i);
        ASSERT(out == values[i], "value mismatch at index %d", i);
    }

    bptree_iter it = bptree_iter_begin(tree);
    int seen = 0;
    while (bptree_iter_valid(&it)) {
        ASSERT(seen < count, "iterator ran past %d entries", count);
        if (seen >= count) break;
        const bptree_key_t key = bptree_iter_key(&it);
        ASSERT(compare_keys(&key, &keys[seen]) == 0, "iteration order differs at index %d", seen);
        ASSERT(bptree_iter_value(&it) == values[seen], "iteration value mismatch at %d", seen);
        ++seen;
        bptree_iter_next(&it);
    }
    ASSERT(seen == count, "iterator visited %d of %d entries", seen, count);
}

/* ---------------------------------------------------------------------------
 * Ordered input and empty input
 * ------------------------------------------------------------------------- */

/** @brief Tree orders exercised by the bulk tests. */
static const int test_max_keys_values[] = {3, 4, 7, 12, 32};

/** @brief Sizes chosen around leaf and internal split boundaries. */
static const int test_counts[] = {1, 2, 3, 4, 5, 7, 8, 11, 31, 32, 33, 64, 100, 257, 1000};

/** @brief Entry count for the shared test buffers. */
#define BULK_TEST_CAPACITY 1000

static bptree_key_t bulk_keys[BULK_TEST_CAPACITY];
static bptree_value_t bulk_values[BULK_TEST_CAPACITY];

/** @brief Computes the smallest possible height for @p count entries. */
static int minimum_height(const int count, const int max_keys) {
    if (count == 0) return 1;
    int leaves = count / max_keys + (count % max_keys != 0 ? 1 : 0);
    int height = 1;
    while (leaves > 1) {
        leaves = leaves / (max_keys + 1) + (leaves % (max_keys + 1) != 0 ? 1 : 0);
        height++;
    }
    return height;
}

static void test_bulk_build_sorted(void) {
    const int num_orders = (int)(sizeof(test_max_keys_values) / sizeof(test_max_keys_values[0]));
    const int num_counts = (int)(sizeof(test_counts) / sizeof(test_counts[0]));
    for (int o = 0; o < num_orders; ++o) {
        for (int c = 0; c < num_counts; ++c) {
            const int count = test_counts[c];
            fill_ascending_keys(bulk_keys, count);
            fill_values(bulk_values, count);
            bptree* tree = bptree_bulk_build(bulk_keys, bulk_values, count, test_max_keys_values[o],
                                             TEST_COMPARE);
            ASSERT(tree != NULL, "build returned NULL (max_keys=%d, count=%d)",
                   test_max_keys_values[o], count);
            if (tree) {
                verify_tree_matches_input(tree, bulk_keys, bulk_values, count);
                ASSERT(bptree_max_keys(tree) == test_max_keys_values[o],
                       "max_keys mismatch: %d != %d", bptree_max_keys(tree),
                       test_max_keys_values[o]);
                ASSERT(bptree_height(tree) == minimum_height(count, test_max_keys_values[o]),
                       "height %d != minimum %d (max_keys=%d, count=%d)", bptree_height(tree),
                       minimum_height(count, test_max_keys_values[o]), test_max_keys_values[o],
                       count);
                bptree_free(tree);
            }
        }
    }
}

static void test_bulk_build_empty(void) {
    bptree* tree = bptree_bulk_build(NULL, NULL, 0, 5, TEST_COMPARE);
    ASSERT(tree != NULL, "empty build returned NULL");
    if (!tree) return;
    ASSERT(bptree_count(tree) == 0, "empty tree count %d != 0", bptree_count(tree));
    ASSERT(bptree_height(tree) == 1, "empty tree height %d != 1", bptree_height(tree));
    ASSERT(bptree_check_invariants(tree), "empty tree invariants failed");

    /* The empty tree must remain usable for further insertions. */
    const bptree_key_t key = make_key(42);
    const bptree_value_t value = (bptree_value_t)(intptr_t)7;
    ASSERT(bptree_put(tree, &key, value) == BPTREE_OK, "put on empty bulk tree failed");
    ASSERT(bptree_count(tree) == 1, "count after put %d != 1", bptree_count(tree));
    bptree_value_t out;
    ASSERT(bptree_get(tree, &key, &out) == BPTREE_OK, "get after put failed");
    ASSERT(out == value, "value after put mismatch");
    ASSERT(bptree_check_invariants(tree), "invariants after put failed");
    bptree_free(tree);
}

/* ---------------------------------------------------------------------------
 * Input validation and rejection
 * ------------------------------------------------------------------------- */

static void test_bulk_rejects_invalid_input(void) {
    bptree_key_t keys[8];
    bptree_value_t values[8];
    fill_ascending_keys(keys, 8);
    fill_values(values, 8);

    /* Duplicate key in the middle. */
    bptree_key_t duplicate[8];
    memcpy(duplicate, keys, sizeof(duplicate));
    duplicate[4] = duplicate[3];
    ASSERT(bptree_bulk_build(duplicate, values, 8, 4, TEST_COMPARE) == NULL,
           "duplicate keys were not rejected");

    /* Descending pair. */
    bptree_key_t descending[8];
    memcpy(descending, keys, sizeof(descending));
    const bptree_key_t swap = descending[2];
    descending[2] = descending[5];
    descending[5] = swap;
    ASSERT(bptree_bulk_build(descending, values, 8, 4, TEST_COMPARE) == NULL,
           "descending keys were not rejected");

    /* Negative count. */
    ASSERT(bptree_bulk_build(keys, values, -1, 4, TEST_COMPARE) == NULL,
           "negative count was not rejected");

    /* Missing arrays for a non-empty build. */
    ASSERT(bptree_bulk_build(NULL, values, 4, 4, TEST_COMPARE) == NULL,
           "NULL keys with count > 0 was not rejected");
    ASSERT(bptree_bulk_build(keys, NULL, 4, 4, TEST_COMPARE) == NULL,
           "NULL values with count > 0 was not rejected");
}

static void test_bulk_rejects_invalid_max_keys(void) {
    bptree_key_t keys[4];
    bptree_value_t values[4];
    fill_ascending_keys(keys, 4);
    fill_values(values, 4);
    const int invalid[] = {-1, 0, 1, 2, 4097};
    const int num_invalid = (int)(sizeof(invalid) / sizeof(invalid[0]));
    for (int i = 0; i < num_invalid; ++i) {
        ASSERT(bptree_bulk_build(keys, values, 4, invalid[i], TEST_COMPARE) == NULL,
               "max_keys %d was not rejected", invalid[i]);
    }
}

/* ---------------------------------------------------------------------------
 * Structural equivalence with repeated insertion
 * ------------------------------------------------------------------------- */

static bptree* build_by_put(const bptree_key_t* keys, const bptree_value_t* values, const int count,
                            const int max_keys) {
    bptree* tree = bptree_create(max_keys, TEST_COMPARE, false);
    if (!tree) return NULL;
    for (int i = 0; i < count; ++i) {
        if (bptree_put(tree, &keys[i], values[i]) != BPTREE_OK) {
            bptree_free(tree);
            return NULL;
        }
    }
    return tree;
}

/** @brief Collects the values produced by a full-range query. */
static bool range_values_match(const bptree* tree, const bptree_key_t* keys,
                               const bptree_value_t* values, const int count) {
    bptree_value_t* results = NULL;
    int n_results = 0;
    if (bptree_get_range(tree, &keys[0], &keys[count - 1], &results, &n_results) != BPTREE_OK) {
        return false;
    }
    bool ok = (n_results == count);
    for (int i = 0; ok && i < count; ++i) {
        ok = (results[i] == values[i]);
    }
    bptree_free_range_results(results);
    return ok;
}

static void test_bulk_matches_put(void) {
    const int num_orders = (int)(sizeof(test_max_keys_values) / sizeof(test_max_keys_values[0]));
    const int num_counts = (int)(sizeof(test_counts) / sizeof(test_counts[0]));
    for (int o = 0; o < num_orders; ++o) {
        for (int c = 0; c < num_counts; ++c) {
            const int count = test_counts[c];
            const int max_keys = test_max_keys_values[o];
            fill_ascending_keys(bulk_keys, count);
            fill_values(bulk_values, count);
            bptree* bulk = bptree_bulk_build(bulk_keys, bulk_values, count, max_keys, TEST_COMPARE);
            bptree* put = build_by_put(bulk_keys, bulk_values, count, max_keys);
            ASSERT(bulk != NULL && put != NULL, "build failed (max_keys=%d, count=%d)", max_keys,
                   count);
            if (bulk && put) {
                ASSERT(bptree_count(bulk) == bptree_count(put), "count differs from put tree");
                ASSERT(bptree_height(bulk) <= bptree_height(put),
                       "bulk height %d exceeds put height %d (max_keys=%d, count=%d)",
                       bptree_height(bulk), bptree_height(put), max_keys, count);
                ASSERT(bptree_check_invariants(bulk), "bulk invariants failed");
                ASSERT(bptree_check_invariants(put), "put invariants failed");

                ASSERT(range_values_match(bulk, bulk_keys, bulk_values, count),
                       "bulk full-range query mismatch (max_keys=%d, count=%d)", max_keys, count);
                ASSERT(range_values_match(put, bulk_keys, bulk_values, count),
                       "put full-range query mismatch (max_keys=%d, count=%d)", max_keys, count);

                /* A sub-range crossing leaf boundaries must agree as well. */
                if (count >= 20) {
                    const int start = count / 7;
                    const int end = count - count / 9 - 1;
                    bptree_value_t* bulk_results = NULL;
                    bptree_value_t* put_results = NULL;
                    int bulk_n = 0;
                    int put_n = 0;
                    ASSERT(bptree_get_range(bulk, &bulk_keys[start], &bulk_keys[end], &bulk_results,
                                            &bulk_n) == BPTREE_OK,
                           "bulk sub-range query failed");
                    ASSERT(bptree_get_range(put, &bulk_keys[start], &bulk_keys[end], &put_results,
                                            &put_n) == BPTREE_OK,
                           "put sub-range query failed");
                    ASSERT(bulk_n == put_n, "sub-range sizes differ: %d != %d", bulk_n, put_n);
                    bool same = (bulk_n == put_n);
                    for (int i = 0; same && i < bulk_n; ++i) {
                        same = bulk_results[i] == put_results[i];
                    }
                    ASSERT(same, "sub-range contents differ (max_keys=%d, count=%d)", max_keys,
                           count);
                    bptree_free_range_results(bulk_results);
                    bptree_free_range_results(put_results);
                }
                bptree_free(bulk);
                bptree_free(put);
            }
        }
    }
}

/** @brief Counts leaves and verifies the leaf-level next chain. */
static bool leaf_chain_matches(const bptree* tree, const bptree_key_t* keys, const int count) {
    const bptree_node* leaf = tree->root;
    while (leaf && !leaf->is_leaf) {
        leaf = bptree_node_children((bptree_node*)leaf, tree->max_keys)[0];
    }
    int seen = 0;
    for (const bptree_node* node = leaf; node != NULL; node = node->next) {
        if (node->num_keys <= 0) return false;
        const bptree_key_t* node_keys = bptree_node_keys(node);
        for (int i = 0; i < node->num_keys; ++i) {
            if (seen >= count || compare_keys(&node_keys[i], &keys[seen]) != 0) return false;
            ++seen;
        }
    }
    return seen == count;
}

static void test_bulk_leaf_chain(void) {
    const int num_orders = (int)(sizeof(test_max_keys_values) / sizeof(test_max_keys_values[0]));
    const int num_counts = (int)(sizeof(test_counts) / sizeof(test_counts[0]));
    for (int o = 0; o < num_orders; ++o) {
        for (int c = 0; c < num_counts; ++c) {
            const int count = test_counts[c];
            fill_ascending_keys(bulk_keys, count);
            fill_values(bulk_values, count);
            bptree* tree = bptree_bulk_build(bulk_keys, bulk_values, count, test_max_keys_values[o],
                                             TEST_COMPARE);
            ASSERT(tree != NULL, "build failed");
            if (tree) {
                ASSERT(leaf_chain_matches(tree, bulk_keys, count),
                       "leaf chain mismatch (max_keys=%d, count=%d)", test_max_keys_values[o],
                       count);
                bptree_free(tree);
            }
        }
    }
}

/* ---------------------------------------------------------------------------
 * Custom comparator
 * ------------------------------------------------------------------------- */

#ifndef BPTREE_KEY_TYPE_STRING
/** @brief Reverses the numeric order so keys must be supplied in descending order. */
static int reverse_compare(const bptree_key_t* a, const bptree_key_t* b) {
    return (*a > *b) ? -1 : ((*a < *b) ? 1 : 0);
}

static void test_bulk_custom_compare(void) {
    const int count = 500;
    bptree_key_t* keys = malloc((size_t)count * sizeof(bptree_key_t));
    bptree_value_t* values = malloc((size_t)count * sizeof(bptree_value_t));
    ASSERT(keys != NULL && values != NULL, "allocation failed");
    if (!keys || !values) {
        free(keys);
        free(values);
        return;
    }
    /* Ascending under reverse_compare means numerically descending. */
    for (int i = 0; i < count; ++i) keys[i] = (bptree_key_t)(count - i);
    fill_values(values, count);

    bptree* tree = bptree_bulk_build(keys, values, count, 7, reverse_compare);
    ASSERT(tree != NULL, "custom comparator build returned NULL");
    if (tree) {
        ASSERT(bptree_count(tree) == count, "custom comparator count mismatch");
        ASSERT(bptree_check_invariants(tree), "custom comparator invariants failed");
        for (int i = 0; i < count; ++i) {
            bptree_value_t out;
            ASSERT(bptree_get(tree, &keys[i], &out) == BPTREE_OK,
                   "custom comparator get failed at %d", i);
            ASSERT(value_matches(out, i), "custom comparator value mismatch at %d", i);
        }
        bptree_iter it = bptree_iter_begin(tree);
        int seen = 0;
        while (bptree_iter_valid(&it)) {
            ASSERT(seen < count, "custom comparator iterator overrun");
            if (seen >= count) break;
            ASSERT(bptree_iter_key(&it) == keys[seen], "custom comparator order differs at %d",
                   seen);
            ++seen;
            bptree_iter_next(&it);
        }
        ASSERT(seen == count, "custom comparator iterator visited %d of %d", seen, count);
        bptree_free(tree);
    }
    free(keys);
    free(values);
}
#endif

/* ---------------------------------------------------------------------------
 * Input ownership and large inputs
 * ------------------------------------------------------------------------- */

static void test_bulk_does_not_retain_input_arrays(void) {
    const int count = 200;
    bptree_key_t* keys = malloc((size_t)count * sizeof(bptree_key_t));
    bptree_value_t* values = malloc((size_t)count * sizeof(bptree_value_t));
    ASSERT(keys != NULL && values != NULL, "allocation failed");
    if (!keys || !values) {
        free(keys);
        free(values);
        return;
    }
    fill_ascending_keys(keys, count);
    fill_values(values, count);
    bptree* tree = bptree_bulk_build(keys, values, count, 4, TEST_COMPARE);
    ASSERT(tree != NULL, "build returned NULL");
    /* Release the caller arrays; the tree must remain fully usable. */
    free(keys);
    free(values);
    if (tree) {
        ASSERT(bptree_count(tree) == count, "count changed after input release");
        ASSERT(bptree_check_invariants(tree), "invariants failed after input release");
        for (int i = 0; i < count; i += 17) {
            const bptree_key_t key = make_key(i * 2);
            bptree_value_t out;
            ASSERT(bptree_get(tree, &key, &out) == BPTREE_OK,
                   "get failed after input release at %d", i);
            ASSERT(value_matches(out, i), "value changed after input release at %d", i);
        }
        bptree_free(tree);
    }
}

static void test_bulk_large_input(void) {
    const int count = 20000;
    bptree_key_t* keys = malloc((size_t)count * sizeof(bptree_key_t));
    bptree_value_t* values = malloc((size_t)count * sizeof(bptree_value_t));
    ASSERT(keys != NULL && values != NULL, "allocation failed");
    if (!keys || !values) {
        free(keys);
        free(values);
        return;
    }
    fill_ascending_keys(keys, count);
    fill_values(values, count);
    const int orders[] = {3, 32};
    for (int o = 0; o < 2; ++o) {
        bptree* tree = bptree_bulk_build(keys, values, count, orders[o], TEST_COMPARE);
        ASSERT(tree != NULL, "large build returned NULL (max_keys=%d)", orders[o]);
        if (tree) {
            ASSERT(bptree_count(tree) == count, "large count mismatch");
            ASSERT(bptree_check_invariants(tree), "large invariants failed (max_keys=%d)",
                   orders[o]);
            ASSERT(bptree_get_stats(tree).node_count > 0, "large node count is zero");
            bptree_value_t out;
            ASSERT(bptree_get(tree, &keys[count / 2], &out) == BPTREE_OK, "large midpoint get");
            ASSERT(out == values[count / 2], "large midpoint value mismatch");
            bptree_free(tree);
        }
    }
    free(keys);
    free(values);
}

/* ---------------------------------------------------------------------------
 * Allocation failure
 * ------------------------------------------------------------------------- */

static void test_bulk_allocation_failure(void) {
    const int count = 300;
    bptree_key_t* keys = malloc((size_t)count * sizeof(bptree_key_t));
    bptree_value_t* values = malloc((size_t)count * sizeof(bptree_value_t));
    ASSERT(keys != NULL && values != NULL, "allocation failed");
    if (!keys || !values) {
        free(keys);
        free(values);
        return;
    }
    fill_ascending_keys(keys, count);
    fill_values(values, count);

    /* Count how many allocations a successful build performs. */
    allocation_calls = 0;
    bptree* tree = bptree_bulk_build(keys, values, count, 4, TEST_COMPARE);
    const int total_allocations = allocation_calls;
    ASSERT(tree != NULL, "baseline build failed");
    if (tree) bptree_free(tree);
    ASSERT(total_allocations > 2, "unexpected allocation count %d", total_allocations);

    /* Fail each allocation in turn; every failure must return NULL cleanly. */
    for (int fail_at = 0; fail_at < total_allocations; ++fail_at) {
        fail_after = fail_at;
        tree = bptree_bulk_build(keys, values, count, 4, TEST_COMPARE);
        fail_after = -1;
        ASSERT(tree == NULL, "allocation failure at %d did not return NULL", fail_at);
        if (tree) bptree_free(tree);
    }

    /* The same build must still succeed once injection is disabled. */
    tree = bptree_bulk_build(keys, values, count, 4, TEST_COMPARE);
    ASSERT(tree != NULL, "build failed after allocation injection was disabled");
    if (tree) {
        ASSERT(bptree_check_invariants(tree), "invariants failed after injection test");
        ASSERT(bptree_count(tree) == count, "count mismatch after injection test");
        bptree_free(tree);
    }
    free(keys);
    free(values);
}

int main(void) {
    RUN_TEST(test_bulk_build_sorted);
    RUN_TEST(test_bulk_build_empty);
    RUN_TEST(test_bulk_rejects_invalid_input);
    RUN_TEST(test_bulk_rejects_invalid_max_keys);
    RUN_TEST(test_bulk_matches_put);
    RUN_TEST(test_bulk_leaf_chain);
    RUN_TEST(test_bulk_does_not_retain_input_arrays);
    RUN_TEST(test_bulk_large_input);
    RUN_TEST(test_bulk_allocation_failure);
#ifndef BPTREE_KEY_TYPE_STRING
    RUN_TEST(test_bulk_custom_compare);
#endif

    fprintf(stderr, "----------------------------------------\n");
    fprintf(stderr, "Test Summary: %d tests run, %d failures.\n", tests_run, tests_failed);
    fprintf(stderr, "----------------------------------------\n");
    return tests_failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
