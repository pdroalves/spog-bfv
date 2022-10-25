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
#include <spog/fv.h>
#include <stdlib.h>
#include <NTL/ZZ.h>
#include <cuda_profiler_api.h>
#include <sstream>
#include <string>

std::string zToString(const ZZ &z) {
    std::stringstream buffer;
    buffer << z;
    return buffer.str();
}

/**
	encrypt_decrypt:

	The intention of this program is to demonstrate the steps required for a
	simple workload of generating a message, encrypt, and decrypt.
 */
int main(int argc, char **argv) {
	cudaProfilerStop();

	FV *cipher;
	ZZ q;

	srand(0);
	NTL::SetSeed(to_ZZ(0));

	// Params
	int nphi = 8192;
	int k = 4;
	int t = 256;

	// Set logger
	Logger::getInstance()->set_mode(INFO);

	// Init
	CUDAEngine::init(k, nphi, t);// Init CUDA

	// FV setup
	cipher = new FV(CUDAEngine::RNSProduct, nphi);
	Sampler::init(cipher->ctx);
	cipher->keygen();

	/////////////
	// Message //
	/////////////
	poly_t m;

	//////////////
	// Sampling //
	//////////////
	poly_t *m_decrypted;
	bool result;
	poly_set_coeff(&m, 0, to_ZZ(42));
	do{
		// Sampler::sample(cipher->ctx, &m, UNIFORM);
		// poly_mod_by_ZZ(&m, &m, to_ZZ(t));

		/////////////
		// Encrypt //
		/////////////
		cipher_t* ct = cipher->encrypt(m);

		/////////////
		// Decrypt //
		/////////////
		m_decrypted = cipher->decrypt(*ct);
		result = poly_are_equal(m_decrypted, &m);

		if(result)
			poly_free(m_decrypted);
		cipher_free(ct);
	} while(result);
	std::cout << "Found it!" << std::endl;
	std::cout << poly_to_string(&m) << std::endl;
	std::cout << poly_to_string(m_decrypted) << "(degree: " << poly_get_deg(m_decrypted) << ")" << std::endl;

	cudaDeviceSynchronize();
	cudaCheckError();
	
	poly_free(&m);
	CUDAEngine::destroy();
	cudaDeviceReset();
	cudaCheckError();
	return 0;
}