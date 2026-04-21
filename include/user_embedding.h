/**
 * @file    user_embedding.h
 * @brief   User embedding computation from interaction history.
 *
 * A user is represented as a weighted average of the item embeddings
 * they have interacted with, weighted by their rating normalised to [0, 1].
 *
 *   user_vec = Σ (rating_i / 5.0) * item_vec_i
 *              ─────────────────────────────────
 *                   Σ (rating_i / 5.0)
 *
 * This follows the two-tower architecture described in:
 *   Yi et al. (2019) — Sampling-Bias-Corrected Neural Modeling
 *
 * Properties:
 *   - Output is L2-normalised — consistent with item embeddings, so
 *     dot product == cosine similarity during KD-tree retrieval.
 *   - Items not found in the embedding index are skipped with a warning.
 *     This can happen for the 16 items dropped during preprocessing.
 *   - A user with no valid embedded items returns a zero vector. The
 *     caller should check for this before querying the KD-tree.
 *
 * @author  anshulbadhani
 * @date    2026
 */

#pragma once
#include "data_loader.h"
#include <array>
#include <string>
#include <vector>
#include <unordered_map>

using std::array,
    std::string,
    std::vector,
    std::unordered_map;

/**
 * @brief Computes a normalised user embedding from their interaction history.
 *
 * @param history       Ordered list of items the user has rated (from train).
 * @param embeddings    Item embedding matrix (row → 384 floats).
 * @param asin_to_idx   Lookup map from ASIN to embedding row.
 *
 * @return  L2-normalised user embedding vector of size DIM.
 *          Returns a zero vector if no items in history have embeddings.
 */
embedding_t compute_user_embedding(
    const vector<Interaction>& history,
    const vector<embedding_t>& embeddings,
    const unordered_map<string, int>& asin_to_idx
);

/**
 * @brief Computes user embeddings for all users in the training set.
 *
 * Convenience wrapper around compute_user_embedding(). Skips users
 * whose embedding is all zeros (no valid items in history).
 *
 * @param user_history  All user interaction histories from train.csv.
 * @param embeddings    Item embedding matrix.
 * @param asin_to_idx   ASIN → row index lookup.
 *
 * @return  Map of user_id → normalised embedding vector.
 */
unordered_map<string, embedding_t> compute_all_user_embeddings(
    const unordered_map<string, vector<Interaction>>& user_history,
    const vector<embedding_t>& embeddings, // item embeddings
    const unordered_map<string, int>& asin_to_idx
);