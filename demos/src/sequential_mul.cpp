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
#include <SPOG/fvcontext.h>
#include <stdlib.h>
#include <NTL/ZZ.h>
#include <cuda_profiler_api.h>

/**
	sequential_mul:

	The intention of this program is to demonstrate the steps required for a
	simple workload of generating a message, encrypt, and execute a lot of
	sequential multiplications. At the end of each iteration it	shall validate
	if the multiplication was successful or if it failed.
 */
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
	int k = 4;

	// Set logger
	Logger::getInstance()->set_mode(INFO);

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
	poly_t m, mR;
	poly_set_coeff(cipher, &m, 1, to_ZZ(1));	

	/////////////
	// Encrypt //
	/////////////
	Logger::getInstance()->log_info("==========================");
	Logger::getInstance()->log_info("Will encrypt");
	cipher_t* ct = fv_encrypt(cipher, &m);

	//////////
	// Mul //
	////////
	Logger::getInstance()->log_info("==========================");
	Logger::getInstance()->log_info("Will mul");

	cipher_t *ctR = new cipher_t();
	poly_t *m_decrypted = new poly_t();
	cipher_init(cipher, ctR, QBase);
	poly_init(cipher, m_decrypted);
	poly_init(cipher, &mR);

	cipher_copy(cipher, ctR, ct);
	poly_copy(cipher, &mR, &m);
	int it = 0;
	do{
		///////////
		// Clean //
		///////////
		poly_free(cipher, m_decrypted);

		/////////
		// Mul //
		/////////
		std::cout << "Iteration " << it << ": ";
		ctR = fv_mul(cipher, ct, ctR);
		poly_mul(cipher, &mR, &m, &mR);

		//////////////
		// Validate //
		/////////////
		m_decrypted = fv_decrypt(cipher, ctR, sk);
		if(poly_are_equal(cipher, m_decrypted, &mR))
			std::cout << "We are good!"  << std::endl;
		else
			std::cout << "Failure /o\\" << std::endl;

		it++;
	} while(poly_are_equal(cipher, m_decrypted, &mR));

	/////////////
	// Release //
	/////////////
	delete cipher;
	CUDAEngine::destroy();
	poly_free(cipher, &m);
	poly_free(cipher, m_decrypted);
	cipher_free(cipher, ct);
	cipher_free(cipher, ctR);

	cudaDeviceSynchronize();
	cudaCheckError();
	
	cudaDeviceReset();
	cudaCheckError();
	return 0;
}