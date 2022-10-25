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

#ifndef BFV_H
#define BFV_H

#include <NTL/ZZ.h>
#include <SPOG-BFV/bfvcontext.h>
#include <SPOG-BFV/arithmetic/ciphertext.h>
#include <cuPoly/arithmetic/polynomial.h>
#include <cuPoly/cuda/sampler.h>
#include <cuPoly/tool/log.h>
#include <cuPoly/arithmetic/context.h>
#include <omp.h>

/**
 * @brief      Generate a set of pk and evk keys
 *
 * @param ctx The context
 * @param[in] sk The secret key
 * 
 * @return     A struct with a tuple of keys
 */
Keys* bfv_keygen(BFVContext *ctx, SecretKey *sk);

/**
 * @brief      BFV encryption
 *
 * @param  ctx The context
 * @param[in]  m The input message
 *
 * @return     The encryption of m
 */
cipher_t* bfv_encrypt(BFVContext *ctx, poly_t *m);

/**
 * @brief      BFV encryption
 *
 * @param ctx The context
 * @param[out] ct    The encryption of m
 * @param[in]  m     The input message
 *
 * @return     The encryption of m
 */
cipher_t* bfv_encrypt(BFVContext *ctx, cipher_t *ct, poly_t *m);

/**
 * @brief      BFV decryption
 *
 * @param ctx The context
 * @param[in] c The encryption of m
 * @param sk The secret key
 *
 * @return     Message m
 */
poly_t* bfv_decrypt(BFVContext *ctx, cipher_t *c, SecretKey *sk);

/**
 * @brief      BFV decryption
 *
 * @param ctx The context
 * @param[out]      m     Message m
 * @param[in]       c     The encryption of m
 * @param sk The secret key
 *
 * @return     Message m
 */
poly_t* bfv_decrypt(BFVContext *ctx, poly_t *m, cipher_t *c, SecretKey *sk);

/**
 * @brief      BFV's Homomorphic addition
 *
 * @param ctx The context
 * @param[in]  ct1   First operand
 * @param[in]  ct2   Second operand
 *
 * @return     Homomorphic addition of ct1 and ct2
 */
cipher_t* bfv_add(BFVContext *ctx, cipher_t *ct1, cipher_t *ct2);

/**
 * @brief      BFV's Homomorphic addition
 *
 * @param ctx The context
 * @param[out]  c3   Homomorphic addition of c1 and c2
 * @param[in]  c1   First operand
 * @param[in]  c2   Second operand
 */
void bfv_add(BFVContext *ctx, cipher_t *c3, cipher_t *c1, cipher_t *c2);

/**
 * @brief      BFV's Homomorphic multiplication
 *
 * @param ctx The context
 * @param[in]  ct1   First operand
 * @param[in]  ct2   Second operand
 * 
 * @return   \f$ct1 \times ct2\f$
 */
cipher_t* bfv_mul(BFVContext *ctx, cipher_t *ct1, cipher_t *ct2);

/**
 * @brief      BFV's Homomorphic multiplication
 *
 * @param ctx The context
 * @param[out]  ct3   Homomorphic multiplication of ct1 and ct2
 * @param[in]  ct1   First operand
 * @param[in]  ct2   Second operand
 */
void bfv_mul(BFVContext *ctx, cipher_t *ct3, cipher_t *ct1, cipher_t *ct2);

/**
 * @brief       Executes the relinearization procedure for BFV's homomorphic multiplication 
 * @param ctx   The context
 * @param[out] c     Relinearized outcome
 * @param[in] cmult  The input three-part ciphertext
 * @param evk        A evk key.
 */
void bfv_relin(
    BFVContext *ctx,
    cipher_t *c,
    poly_t *cmult,
    EvaluationKey *evk);

/**
 * @brief      Multiply a ciphertext by a plaintext
 *
 * @param ctx The context
 * @param[in]  ct    The ciphertext
 * @param[in]  pt    The plaintext
 *
 * @return     \f$ct * pt\f$
 */
cipher_t* bfv_plainmul(BFVContext *ctx, cipher_t *ct, poly_t *pt);

/**
 * @brief      Multiply a ciphertext by a plaintext
 *
 * @param ctx The context
 * @param[out]  result	\f$ct * pt\f$
 * @param[in]  ct    	The ciphertext
 * @param[in]  pt    	The plaintext
 */
void bfv_plainmul(BFVContext *ctx, cipher_t *result, cipher_t *ct, poly_t *pt);

/**
 * @brief      Multiply a ciphertext by a plaintext followed by the 
 *             addition of another ciphertext
 *             
 *   Compute \f$ctm * pt + cta\f$
 *
 * @param ctx The context
 * @param[in]  ctm   The ciphertext to be multiplied 
 * @param[in]  cta   The ciphertext to be added
 * @param[in]  pt    The plaintext
 *
 * @return     \f$ctm * pt + cta\f$
 */
cipher_t* bfv_plainmuladd(BFVContext *ctx, cipher_t *ctm, cipher_t *cta, poly_t *pt);

/**
 * @brief      Multiply a ciphertext by a plaintext followed by the 
 *             addition of another ciphertext
 *             
 *   Compute \f$ctm * pt + cta\f$
 *
 * @param ctx The context
 * @param[out] result  \f$ctm * pt + cta\f$
 * @param[in]   ctm   	The ciphertext to be multiplied 
 * @param[in]   cta   	The ciphertext to be added
 * @param[in]   pt    	The plaintext
 */
void bfv_plainmuladd(BFVContext *ctx, cipher_t *result, cipher_t *ctm, cipher_t *cta, poly_t *pt);


	

#endif