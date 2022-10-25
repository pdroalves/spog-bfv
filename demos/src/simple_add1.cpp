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

#include <cuPoly/settings.h>
#include <cuPoly/arithmetic/polynomial.h>
#include <SPOG/fv.h>
#include <stdlib.h>
#include <NTL/ZZ.h>
#include <cuda_profiler_api.h>

NTL_CLIENT

#include <sstream>
#include <string>

#define NRUNS 100

int main() {
	cudaProfilerStop();

	FVContext *cipher;
	ZZ q;

	srand(0);
	NTL::SetSeed(to_ZZ(0));

	// Params
    Params p;
    p.nphi = 4096;
    p.t = 256;
	int k = 1;

	// Init
	CUDAEngine::init(k, k + 1, p.nphi, p.t);// Init CUDA
	p.q = CUDAEngine::RNSProduct;

	// FV setup
	cipher = new FVContext(p);
	Sampler::init(cipher);
	SecretKey *sk = fv_new_sk(cipher);
	fv_keygen(cipher, sk);

	/////////////
	// Message //
	/////////////
	poly_t m1, m2;
	poly_init(cipher, &m1);
	poly_init(cipher, &m2);
    Sampler::sample(cipher, &m1, UNIFORM);
    Sampler::sample(cipher, &m2, UNIFORM);

	// Copy to GPU's global memory
	poly_copy_to_device(cipher, &m1);
	poly_copy_to_device(cipher, &m2);

	/////////////
	// Encrypt //
	/////////////
	cipher_t* ct1 = fv_encrypt(cipher, &m1);
	cipher_t* ct2 = fv_encrypt(cipher, &m2);
	cipher_t* ct3 = new cipher_t;
	cipher_init(cipher, ct3);

	//////////
	// Add //
	////////
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
  	float latency = 0;

	cudaDeviceSynchronize();
	cudaCheckError();
  	cudaEventRecord(start, cipher->get_stream());
  	for(int i = 0; i < NRUNS; i++)
		// Encrypted add
		fv_add(cipher, ct3, ct1, ct2);
	cudaEventRecord(stop, cipher->get_stream());
	cudaCheckError();
	cudaEventSynchronize(stop);
	cudaCheckError();
	cudaEventElapsedTime(&latency, start, stop);
	cudaProfilerStop();
	std::cout << "cudaEvent_T got " << (latency/NRUNS) << " ms" << std::endl;
	
	poly_free(cipher, &m1);
	poly_free(cipher, &m2);
	cipher_free(cipher, ct1);
	cipher_free(cipher, ct2);
	cipher_free(cipher, ct3);
	delete ct1;
	delete ct2;
	delete ct3;
	delete cipher;
	CUDAEngine::destroy();
	cudaDeviceReset();
	cudaCheckError();
	return 0;
}