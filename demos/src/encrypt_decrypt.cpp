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
int main() {
	cudaProfilerStop();

	FVContext *cipher;
	ZZ q;

	srand(0);
	NTL::SetSeed(to_ZZ(0));

	// Params
    Params p;
    p.nphi = 8192;
    p.t = 256;
	int k = 3;

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
	poly_t m;

	poly_set_coeff(cipher, &m, 0, to_ZZ(68));	
	poly_set_coeff(cipher, &m, 1, to_ZZ(102));
	poly_set_coeff(cipher, &m, 2, to_ZZ(88));
	poly_set_coeff(cipher, &m, 3, to_ZZ(113));
	poly_set_coeff(cipher, &m, 4, to_ZZ(74));
	poly_set_coeff(cipher, &m, 5, to_ZZ(101));
	poly_set_coeff(cipher, &m, 6, to_ZZ(22));
	poly_set_coeff(cipher, &m, 7, to_ZZ(127));

	/////////////
	// Encrypt //
	/////////////
	Logger::getInstance()->log_info("==========================");
	Logger::getInstance()->log_info("Will encrypt");
	cipher_t* ct = fv_encrypt(cipher, &m);

	/////////////
	// Decrypt //
	/////////////
	poly_t *m_decrypted = fv_decrypt(cipher, ct, sk);
	Logger::getInstance()->log_info(( "m: " + poly_to_string(cipher, &m)).c_str());

	//////////////
	// Validate //
	/////////////
	Logger::getInstance()->log_info(( "m_expected: " + poly_to_string(cipher, &m)).c_str());
	Logger::getInstance()->log_info(( "m_decrypted: " + poly_to_string(cipher, m_decrypted)).c_str());
	Logger::getInstance()->log_info(( poly_are_equal(cipher, m_decrypted, &m) == true? "Success!" : "Failure =("));

	cudaDeviceSynchronize();
	cudaCheckError();
	
	poly_free(cipher, &m);
	poly_free(cipher, m_decrypted);
	cipher_free(cipher, ct);
	delete cipher;
	CUDAEngine::destroy();
	cudaDeviceReset();
	cudaCheckError();
	return 0;
}