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

#include <SPOG-BFV/bfv.h>

__host__ Keys* bfv_keygen(BFVContext *ctx, SecretKey *sk){
	Keys *keys;
	assert(sk->s.init);

	//////////////////////////////////
	// Alloc memory for each key //
	//////////////////////////////////
	keys = new Keys;
	keys_init(ctx, keys);

	const int k = CUDAEngine::RNSPrimes.size();

	////////////////
	// Public key //
	////////////////
	poly_t e;
	Sampler::sample(ctx, &keys->pk.a, UNIFORM);
	Sampler::sample(ctx, &e, DISCRETE_GAUSSIAN);

	//
	// b = [-(a*s + e)]_q
	//
	poly_mul(ctx, &keys->pk.b, &keys->pk.a, &sk->s);
	poly_add(ctx, &keys->pk.b, &keys->pk.b, &e);
	poly_negate(ctx, &keys->pk.b);

	poly_copy_to_device(ctx, &keys->pk.a);
	poly_copy_to_device(ctx, &keys->pk.b);

	////////////////////
	// Evaluation key //
	////////////////////
	poly_t s2;
	poly_mul(ctx, &s2, &sk->s, &sk->s);

	// Compute rho //
	poly_rho_bfv(ctx, keys->evk.b, &s2);

	poly_t a;
	poly_init(ctx, &a);
	for(int i = 0; i < k; i++){
		poly_clear(ctx, &a);
		poly_clear(ctx, &e);

		Sampler::sample(ctx, &keys->evk.a[i], UNIFORM);
		Sampler::sample(ctx, &e, DISCRETE_GAUSSIAN);

		// Rho_rns(s^2) - a*s + e
		poly_mul(ctx, &a, &keys->evk.a[i], &sk->s);
		poly_sub(ctx, &keys->evk.b[i], &keys->evk.b[i], &a);
		poly_add(ctx, &keys->evk.b[i], &keys->evk.b[i], &e);

		//
		poly_copy_to_device(ctx, &keys->evk.a[i]);
		poly_copy_to_device(ctx, &keys->evk.b[i]);
	}


	/////////////////
	// Release memory //
	/////////////////
	poly_free(ctx, &e);
	poly_free(ctx, &s2);
	poly_free(ctx, &a);

	////////////
	// Export //
	////////////
	ctx->keys = keys;
	return keys;
}

__host__ cipher_t* bfv_encrypt(BFVContext *ctx, poly_t *m){
    	/////////////////////
	// Allocate memory //
	/////////////////////
	cipher_t *ct = new cipher_t;
	cipher_init(ctx, ct);
        
    return bfv_encrypt(ctx, ct, m);
}

__host__ cipher_t* bfv_encrypt(BFVContext *ctx, cipher_t *ct, poly_t *m){
	assert(CUDAEngine::is_init);
	assert(m->init);

	////////////
	// Sample //
	////////////
	Sampler::sample(ctx, ctx->u, NARROW);
	Sampler::sample(ctx, ctx->e1, DISCRETE_GAUSSIAN);

	/////////
	// Enc //
	/////////

	// c0 = delta*m + u*p0 + e0
	poly_mul(	 ctx, &ct->c[0], m, &ctx->delta);
	poly_mul_add(ctx, &ct->c[0], ctx->u, &ctx->keys->pk.b, &ct->c[0]);
	poly_add(    ctx, &ct->c[0], &ct->c[0], ctx->e1);

	// c1 = u*p1 + e1
	Sampler::sample(ctx, ctx->e1, DISCRETE_GAUSSIAN);
	poly_mul_add(ctx, &ct->c[1], ctx->u, &ctx->keys->pk.a, ctx->e1);

	//
	poly_clear(ctx, ctx->u);
	poly_clear(ctx, ctx->e1);
	return ct;
}

__host__ poly_t* bfv_decrypt(BFVContext *ctx, cipher_t *c, SecretKey *sk){
	poly_t *m = new poly_t;
    return bfv_decrypt(ctx, m, c, sk);
}

__host__ poly_t* bfv_decrypt(BFVContext *ctx, poly_t *m, cipher_t *c, SecretKey *sk){
	assert(CUDAEngine::is_init);

	//////////////////////////////////////////
	// Compute x = |(c0 + c1*s)|_q //
	//////////////////////////////////////////
	poly_mul_add(ctx, ctx->m_aux, &c->c[1], &sk->s, &c->c[0]);

	//////////////////////////////////////////////
	// Simple scaling procedure from Section 2.3//
	//////////////////////////////////////////////
	poly_simple_scaling_tDivQ(ctx, m, ctx->m_aux);
	return m;
}

cipher_t* bfv_add(BFVContext *ctx, cipher_t *c1, cipher_t *c2){
	assert(CUDAEngine::is_init);
	cipher_t *c3 = new cipher_t;

	bfv_add(ctx, c3, c1, c2);

	return c3;
}

__host__  void bfv_add(BFVContext *ctx, cipher_t *c3, cipher_t *c1, cipher_t *c2){
	assert(CUDAEngine::is_init);

	poly_double_add(
		ctx,
		&c3->c[0],
		&c1->c[0],
		&c2->c[0],
		&c3->c[1],
		&c1->c[1],
		&c2->c[1]);

	return;
}

__host__ cipher_t* bfv_plainmul(BFVContext *ctx, cipher_t *ct, poly_t *pt)
{
    assert(CUDAEngine::is_init);
    cipher_t *c3 = new cipher_t;
    
    bfv_plainmul(ctx, c3, ct, pt);
    return c3;
}

__host__ void bfv_plainmul(BFVContext *ctx, cipher_t *result, cipher_t *ct, poly_t *pt)
{
    assert(CUDAEngine::is_init);
    
    poly_mul(ctx, &result->c[0], &ct->c[0], pt);
    poly_mul(ctx, &result->c[1], &ct->c[1], pt);
}

//d = c + a*b
//void poly_mul_add(poly_t *d, poly_t *a, poly_t *b, poly_t *c)
__host__ cipher_t* bfv_plainmuladd(BFVContext *ctx, cipher_t *ctm, cipher_t *cta, poly_t *pt)
{
    assert(CUDAEngine::is_init);
    
    cipher_t *result = new cipher_t;
    
    bfv_plainmuladd(ctx, result, ctm, cta, pt);
    
    return result;
}

__host__ void bfv_plainmuladd(BFVContext *ctx, cipher_t *result, cipher_t *ctm, cipher_t *cta, poly_t *pt)
{
    assert(CUDAEngine::is_init);
    
    poly_mul_add(ctx, &result->c[0], &ctm->c[0], pt, &cta->c[0]);
    poly_mul_add(ctx, &result->c[1], &ctm->c[1], pt, &cta->c[1]);
}

__host__ cipher_t* bfv_mul(BFVContext *ctx, cipher_t *c1, cipher_t *c2){
	assert(CUDAEngine::is_init);

	cipher_t *c3 = new cipher_t;

	bfv_mul(ctx, c3, c1, c2);

	return c3;
}


__host__ void bfv_mul(BFVContext *ctx, cipher_t *c3, cipher_t *c1, cipher_t *c2){
	////////////////////////////////////////////
	// Execute the homomorphic multiplication //
	////////////////////////////////////////////
	///
    cudaStreamSynchronize(ctx->get_stream());
    cudaCheckError();

	////////////////////////////
	// Basis extension to Q*B //
	////////////////////////////
	poly_basis_extension_Q_to_B(ctx, &ctx->c1_B->c[0], &c1->c[0]);
	poly_basis_extension_Q_to_B(ctx->get_alt_ctx(1), &ctx->c2_B->c[0], &c2->c[0]);
	poly_basis_extension_Q_to_B(ctx->get_alt_ctx(2), &ctx->c1_B->c[1], &c1->c[1]);
	poly_basis_extension_Q_to_B(ctx->get_alt_ctx(3), &ctx->c2_B->c[1], &c2->c[1]);

	////////////////////////
	// Compute DR2(c1,c2) //
	////////////////////////
	poly_dr2(
		ctx,
		&ctx->cstar_Q[0],
		&ctx->cstar_Q[1],
		&ctx->cstar_Q[2],
		&c1->c[0],
		&c1->c[1],
		&c2->c[0],
		&c2->c[1]);

	// Sync
	ctx->sync_related();

	// We need to finish the basis extension before being able to compute DR2
	// at base B
	poly_dr2(
		ctx,
		&ctx->cstar_B[0],
		&ctx->cstar_B[1],
		&ctx->cstar_B[2],
		&ctx->c1_B->c[0],
		&ctx->c1_B->c[1],
		&ctx->c2_B->c[0],
		&ctx->c2_B->c[1]);

	ctx->sync();

	//////////////////////////////////////
	// Scaling and basis extension to Q //
	//////////////////////////////////////
	for (int i = 0; i < 3; i++)
		poly_complex_scaling_tDivQ(
			ctx->get_alt_ctx(i),
			&ctx->cmult[i], 
			&ctx->cstar_Q[i], 
			&ctx->cstar_B[i]);


	ctx->sync_related();

	/////////////////////
	// Relinearization //
	/////////////////////
	bfv_relin(
		ctx,
		c3,
		ctx->cmult,
		&ctx->keys->evk);

	return;
}


__host__ void bfv_relin(
	BFVContext *ctx,
	cipher_t *c,
	poly_t *cmult,
	EvaluationKey *evk){

	const int k = CUDAEngine::RNSPrimes.size();

	///////////////////////////////
	// Compute the decomposition //
	///////////////////////////////
	poly_xi_bfv(ctx, ctx->d, &cmult[2]);

	///////////////////
	// Inner product //
	///////////////////
	////////
	// C0 //
	////////
	poly_dot(ctx, &c->c[0], ctx->d, evk->a, k);
	poly_add(ctx, &c->c[0], &c->c[0], &cmult[0]);

	////////
	// C1 //
	////////
	poly_dot(ctx, &c->c[1], ctx->d, evk->b, k);
	poly_add(ctx, &c->c[1], &c->c[1], &cmult[1]);
}


