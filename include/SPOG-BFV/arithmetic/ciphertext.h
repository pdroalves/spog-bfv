// SPOG
// Copyright (C) 2017-2021 SPOG Authors
// 
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
// 
#ifndef CIPHERTEXT_H
#define CIPHERTEXT_H

#include <cuPoly/arithmetic/polynomial.h>


/**
 * Represents a BFV or CKKS ciphertext
 */
typedef struct{
    poly_t c[2];
    int level = 0; /// The ciphertext level (CKKS only!)
} cipher_t;

/**
 * Initializes a cipher_t
 * @param ctx  The context
 * @param c    A pointer
 * @param base The base that must be supported by the cipher_t
 */
__host__ void cipher_init(Context *ctx, cipher_t *c, int base = QBase);

/**
 * Clear the memory without freeing it
 * @param ctx The context
 * @param ct  A pointer
 */
__host__ void cipher_clear(Context *ctx, cipher_t *ct);

/**
 * Releases the internal pointers
 * @param ctx The context
 * @param ct  A pointer
 */
__host__ void cipher_free(Context *ctx, cipher_t *c);

/**
 * Copy cipher_t's
 * @param ctx The context
 * @param[out] b   The outcome
 * @param[in] a   The input
 */
__host__ void cipher_copy(Context *ctx, cipher_t *b, cipher_t *a);

__host__ std::string cipher_to_string(Context *ctx, cipher_t *c);

/**
 * @brief      Serializes a ciphertext
 *
 * @param      ctx    The context
 * @param      ct     The ciphertext to be serialized
 *
 * @return     { description_of_the_return_value }
 */
std::vector<std::string> cipher_export(Context *ctx, cipher_t *ct);

/**
 * @brief      Imports a serialized ciphertext
 *
 * @param      ctx    The context
 * @param[in]  v     The ciphertext to be imported
 *
 * @return     { description_of_the_return_value }
 */
cipher_t* cipher_import(Context *ctx, std::vector<std::string> v);

#endif