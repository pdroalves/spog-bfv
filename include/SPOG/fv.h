/**
 * SPOG
 * Copyright (C) 2017-2019 SPOG Authors
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef FV_H
#define FV_H

#include <NTL/ZZ.h>
#include <SPOG/fvcontext.h>
#include <SPOG/arithmetic/ciphertext.h>
#include <cuPoly/arithmetic/polynomial.h>
#include <cuPoly/cuda/sampler.h>
#include <cuPoly/tool/log.h>
#include <cuPoly/arithmetic/context.h>
#include <omp.h>

/**
 * [xi_rns description]
 * @param b [output]
 * @param a [input]
 */
__host__ void xi_rns(poly_t *b, poly_t *a, int radix);

/**
 * [rho_rns description]
 * @param b [output]
 * @param a [input]
 */
__host__ void rho_rns(poly_t *b, poly_t *a);

__host__ void fv_relin(
    FVContext *ctx,
    cipher_t *c,
    poly_t *cmult,
    EvaluationKey *evk);

/////////////////////
// FV cryptosystem //
/////////////////////

/**
 * @brief      Generate a set of keys
 *
 * @return     A struct with a tuple of keys
 */
Keys* fv_keygen(FVContext *ctx, SecretKey *sk);

/**
 * @brief      FV encryption
 *
 * @param[in]      m     The input message
 *
 * @return     The encryption of m
 */
cipher_t* fv_encrypt(FVContext *ctx, poly_t *m);

/**
 * @brief      FV encryption
 *
 * @param[out]      ct    The encryption of m
 * @param[in]       m     The input message
 *
 * @return     The encryption of m
 */
cipher_t* fv_encrypt(FVContext *ctx, cipher_t *ct, poly_t *m);

/**
 * @brief      FV decryption
 *
 * @param[in]      c     The encryption of m
 *
 * @return     Message m
 */
poly_t* fv_decrypt(FVContext *ctx, cipher_t *c, SecretKey *sk);

/**
 * @brief      FV decryption
 *
 * @param[out]      m     Message m
 * @param[in]       c     The encryption of m
 *
 * @return     Message m
 */
poly_t* fv_decrypt(FVContext *ctx, poly_t *m, cipher_t *c, SecretKey *sk);

/**
 * @brief      FV's Homomorphic addition
 *
 * @param[in]  ct1   First operand
 * @param[in]  ct2   Second operand
 *
 * @return     Homomorphic addition of ct1 and ct2
 */
cipher_t* fv_add(FVContext *ctx, cipher_t *ct1, cipher_t *ct2);

/**
 * @brief      FV's Homomorphic addition
 *
 * @param[in]  ct3   Homomorphic addition of ct1 and ct2
 * @param[in]  ct1   First operand
 * @param[in]  ct2   Second operand
 */
void fv_add(FVContext *ctx, cipher_t *c3, cipher_t *c1, cipher_t *c2);

/**
 * @brief      FV's Homomorphic multiplication
 *
 * @param[in]  ct1   First operand
 * @param[in]  ct2   Second operand
 * 
 * @return   \f$ct1 \times ct2\f$
 */
cipher_t* fv_mul(FVContext *ctx, cipher_t *ct1, cipher_t *ct2);

/**
 * @brief      FV's Homomorphic multiplication
 *
 * @param[in]  ct3   Homomorphic multiplication of ct1 and ct2
 * @param[in]  ct1   First operand
 * @param[in]  ct2   Second operand
 */
void fv_mul(FVContext *ctx, cipher_t *ct3, cipher_t *ct1, cipher_t *ct2);
/**
 * @brief      Multiply a ciphertext by a plaintext
 *
 * @param[in]  ct    The ciphertext
 * @param[in]  pt    The plaintext
 *
 * @return     \f$ct * pt\f$
 */
cipher_t* fv_plainmul(FVContext *ctx, cipher_t *ct, poly_t *pt);

/**
 * @brief      Multiply a ciphertext by a plaintext
 *
 * @param[in]  result	\f$ct * pt\f$
 * @param[in]  ct    	The ciphertext
 * @param[in]  pt    	The plaintext
 */
void fv_plainmul(FVContext *ctx, cipher_t *result, cipher_t *ct, poly_t *pt);

/**
 * @brief      Multiply a ciphertext by a plaintext followed by the 
 *             addition of another ciphertext
 *             
 *   Compute \f$ctm * pt + cta\f$
 *
 * @param[in]  ctm   The ciphertext to be multiplied 
 * @param[in]  cta   The ciphertext to be added
 * @param[in]  pt    The plaintext
 *
 * @return     \f$ctm * pt + cta\f$
 */
cipher_t* fv_plainmuladd(FVContext *ctx, cipher_t *ctm, cipher_t *cta, poly_t *pt);

/**
 * @brief      Multiply a ciphertext by a plaintext followed by the 
 *             addition of another ciphertext
 *             
 *   Compute \f$ctm * pt + cta\f$
 *
 * @return[out] result  \f$ctm * pt + cta\f$
 * @param[in]   ctm   	The ciphertext to be multiplied 
 * @param[in]   cta   	The ciphertext to be added
 * @param[in]   pt    	The plaintext
 */
void fv_plainmuladd(FVContext *ctx, cipher_t *result, cipher_t *ctm, cipher_t *cta, poly_t *pt);
	

#endif