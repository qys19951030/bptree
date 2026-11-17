/**
 * @file bptree_bulk.h
 * @brief Bulk construction of a B+ tree from an ordered key/value array.
 *
 * This header complements "bptree.h". It builds a B+ tree directly from a
 * caller-provided array of keys sorted in strict ascending order, laying out
 * nodes from the leaf level upward instead of inserting keys one by one.
 *
 * Usage follows the single-header convention of "bptree.h": include this
 * header for the declaration, and define BPTREE_IMPLEMENTATION in exactly one
 * translation unit before including it to generate the implementation.
 *
 * Example (in one .c file):
 * #define BPTREE_IMPLEMENTATION
 * #include "bptree_bulk.h"
 *
 * @version 0.5.1
 */

#ifndef BPTREE_BULK_H
#define BPTREE_BULK_H

#include "bptree.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Builds a B+ tree from a sorted array of keys and values.
 *
 * The keys must be in strict ascending order according to @p compare (or the
 * default comparison when @p compare is NULL). Keys stored in the tree are
 * copies, while values are stored as given. The caller keeps ownership of both
 * input arrays and of any data referenced by the values.
 *
 * A zero @p count produces an empty tree equivalent to the one returned by
 * bptree_create(), which can be used with bptree_put() afterwards.
 *
 * @param keys     Array of @p count keys in strict ascending order.
 * @param values   Array of @p count values matching @p keys by index.
 * @param count    Number of key/value pairs. Must not be negative.
 * @param max_keys Maximum number of keys that each node can contain.
 *                 Must satisfy the same bounds as bptree_create().
 * @param compare  Key comparison function, or NULL for the default comparison.
 * @return A newly built tree, or NULL on invalid input or allocation failure.
 *         A failed call leaves no tree behind.
 */
BPTREE_API bptree* bptree_bulk_build(const bptree_key_t* keys, const bptree_value_t* values,
                                     int count, int max_keys,
                                     int (*compare)(const bptree_key_t*, const bptree_key_t*));

#ifdef BPTREE_IMPLEMENTATION

/**
 * @brief Distribute @p total items across @p groups as evenly as possible.
 *
 * Earlier groups receive one extra item when the division is not exact.
 *
 * @param group_index Index of the group in [0, groups).
 * @param total Number of items to distribute.
 * @param groups Number of groups. Must be positive.
 * @return Number of items assigned to @p group_index.
 */
static int bptree_bulk_group_size(const int group_index, const int total, const int groups) {
    const int base = total / groups;
    const int remainder = total % groups;
    return base + (group_index < remainder ? 1 : 0);
}

/**
 * @brief Build the leaf level of the tree.
 *
 * Allocates and fills @p leaf_count leaves, copying keys and values from the
 * caller arrays, and links the leaves through their `next` pointers.
 *
 * @param tree Tree providing node layout parameters.
 * @param keys Sorted key array.
 * @param values Matching value array.
 * @param count Number of entries. Must be positive.
 * @param out_leaves Receives the allocated leaf array.
 * @param out_leaf_count Receives the number of allocated leaves.
 * @return BPTREE_OK on success, or BPTREE_ALLOCATION_FAILURE.
 */
static bptree_status bptree_bulk_build_leaves(const bptree* tree, const bptree_key_t* keys,
                                              const bptree_value_t* values, const int count,
                                              bptree_node*** out_leaves, int* out_leaf_count) {
    const int leaf_count = count / tree->max_keys + (count % tree->max_keys != 0 ? 1 : 0);
    bptree_node** leaves = calloc((size_t)leaf_count, sizeof(bptree_node*));
    if (!leaves) return BPTREE_ALLOCATION_FAILURE;
    int offset = 0;
    for (int i = 0; i < leaf_count; ++i) {
        bptree_node* leaf = bptree_node_alloc(tree, true);
        if (!leaf) {
            for (int j = 0; j < i; ++j) free(leaves[j]);
            free(leaves);
            return BPTREE_ALLOCATION_FAILURE;
        }
        const int leaf_size = bptree_bulk_group_size(i, count, leaf_count);
        leaf->num_keys = leaf_size;
        leaf->next = NULL;
        memcpy(bptree_node_keys(leaf), &keys[offset], (size_t)leaf_size * sizeof(bptree_key_t));
        memcpy(bptree_node_values(leaf, tree->max_keys), &values[offset],
               (size_t)leaf_size * sizeof(bptree_value_t));
        leaves[i] = leaf;
        offset += leaf_size;
    }
    for (int i = 0; i + 1 < leaf_count; ++i) {
        leaves[i]->next = leaves[i + 1];
    }
    *out_leaves = leaves;
    *out_leaf_count = leaf_count;
    return BPTREE_OK;
}

/**
 * @brief Build the internal levels above the leaf level.
 *
 * Groups the nodes of each level into parents and repeats until a single root
 * remains. Each parent key is the smallest key of its matching child subtree,
 * which keeps the ordering invariants of "bptree.h" intact.
 *
 * @param tree Tree providing node layout parameters.
 * @param leaves Leaf nodes sorted by key.
 * @param leaf_count Number of leaves. Must be positive.
 * @return BPTREE_OK on success, or BPTREE_ALLOCATION_FAILURE.
 */
static bptree_status bptree_bulk_build_levels(bptree* tree, bptree_node** leaves,
                                              const int leaf_count) {
    const int max_keys = tree->max_keys;
    bptree_node** level = leaves;
    bptree_key_t* first_keys = malloc((size_t)leaf_count * sizeof(bptree_key_t));
    if (!first_keys) {
        for (int i = 0; i < leaf_count; ++i) bptree_free_node(leaves[i], tree);
        free(leaves);
        return BPTREE_ALLOCATION_FAILURE;
    }
    for (int i = 0; i < leaf_count; ++i) {
        first_keys[i] = bptree_node_keys(level[i])[0];
    }
    int level_count = leaf_count;
    while (level_count > 1) {
        const int group_count = (level_count + max_keys) / (max_keys + 1);
        bptree_node** parents = calloc((size_t)group_count, sizeof(bptree_node*));
        bptree_key_t* parent_first_keys = malloc((size_t)group_count * sizeof(bptree_key_t));
        if (!parents || !parent_first_keys) {
            free(parents);
            free(parent_first_keys);
            for (int i = 0; i < level_count; ++i) bptree_free_node(level[i], tree);
            free(level);
            free(first_keys);
            return BPTREE_ALLOCATION_FAILURE;
        }
        int built = 0;
        int child = 0;
        for (int i = 0; i < group_count; ++i) {
            const int child_count = bptree_bulk_group_size(i, level_count, group_count);
            bptree_node* parent = bptree_node_alloc(tree, false);
            if (!parent) {
                for (int j = 0; j < built; ++j) free(parents[j]);
                free(parents);
                free(parent_first_keys);
                for (int j = 0; j < level_count; ++j) bptree_free_node(level[j], tree);
                free(level);
                free(first_keys);
                return BPTREE_ALLOCATION_FAILURE;
            }
            parent->num_keys = child_count - 1;
            parent->next = NULL;
            bptree_node** children = bptree_node_children(parent, max_keys);
            bptree_key_t* parent_keys = bptree_node_keys(parent);
            parent_first_keys[i] = first_keys[child];
            for (int j = 0; j < child_count; ++j) {
                children[j] = level[child];
                if (j > 0) parent_keys[j - 1] = first_keys[child];
                ++child;
            }
            parents[i] = parent;
            ++built;
        }
        free(level);
        free(first_keys);
        level = parents;
        first_keys = parent_first_keys;
        level_count = group_count;
        ++tree->height;
    }
    tree->root = level[0];
    free(level);
    free(first_keys);
    return BPTREE_OK;
}

BPTREE_API bptree* bptree_bulk_build(const bptree_key_t* keys, const bptree_value_t* values,
                                     const int count, const int max_keys,
                                     int (*compare)(const bptree_key_t*, const bptree_key_t*)) {
    if (count < 0) return NULL;
    if (count > 0 && (!keys || !values)) return NULL;

    bptree* tree = bptree_create(max_keys, compare, false);
    if (!tree) return NULL;

    for (int i = 1; i < count; ++i) {
        if (tree->compare(&keys[i - 1], &keys[i]) >= 0) {
            bptree_free(tree);
            return NULL;
        }
    }

    if (count == 0) {
        return tree;
    }

    bptree_node* old_root = tree->root;
    bptree_node** leaves = NULL;
    int leaf_count = 0;
    if (bptree_bulk_build_leaves(tree, keys, values, count, &leaves, &leaf_count) != BPTREE_OK) {
        bptree_free(tree);
        return NULL;
    }

    free(old_root);
    tree->root = leaves[0];
    tree->count = count;
    tree->height = 1;
    if (bptree_bulk_build_levels(tree, leaves, leaf_count) != BPTREE_OK) {
        free(tree);
        return NULL;
    }
    return tree;
}

#endif

#ifdef __cplusplus
}
#endif

#endif
