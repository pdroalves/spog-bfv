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

std::string zToString(const ZZ &z) {
    std::stringstream buffer;
    buffer << z;
    return buffer.str();
}

std::string stringifyJson(json d){
	StringBuffer buffer;
	buffer.Clear();

	Writer<StringBuffer> writer(buffer);
	d.Accept(writer);

	return buffer.GetString();
}
/**
	add_mul:

	The intention of this program is to demonstrate the steps required for a
	simple workload of generating messages m1, m2, and m3; encrypt; execute 
	m1 * m2 + m3; and verify the result.
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
	int k = 3;

	// Set logger
	Logger::getInstance()->set_mode(INFO);

	Logger::getInstance()->log_info("==========================");
	Logger::getInstance()->log_info( ("nphi: " + std::to_string(p.nphi)).c_str());
	Logger::getInstance()->log_info( ("k: " + std::to_string(k)).c_str());
	Logger::getInstance()->log_info( ("t: " + std::to_string(p.t)).c_str());

	// Init
	CUDAEngine::init(k, k + 1, p.nphi, p.t);// Init CUDA
	p.q = CUDAEngine::RNSProduct;
	Logger::getInstance()->log_info( ("q: " + zToString(p.q) + " (" + std::to_string(NTL::NumBits(p.q)) + " bits) ").c_str());
	Logger::getInstance()->log_info("==========================");

	// FV setup
	cipher = new FVContext(p);
	Sampler::init(cipher);
	SecretKey *sk = fv_new_sk(cipher);
	fv_keygen(cipher, sk);
	Logger::getInstance()->log_debug(("Keys:" + stringifyJson(cipher->export_keys())).c_str());

	/////////////
	// Message //
	/////////////
	poly_t m1, m2, m3;
	poly_init(cipher, &m1);
	poly_set_coeff(cipher, &m1, 0, to_ZZ(1));	
	poly_set_coeff(cipher, &m1, 1, to_ZZ(4));

	poly_init(cipher, &m2);
	poly_set_coeff(cipher, &m2, 0, to_ZZ(68));	
	poly_set_coeff(cipher, &m2, 1, to_ZZ(102));
	poly_set_coeff(cipher, &m2, 2, to_ZZ(88));
	poly_set_coeff(cipher, &m2, 3, to_ZZ(113));
	poly_set_coeff(cipher, &m2, 4, to_ZZ(74));
	poly_set_coeff(cipher, &m2, 5, to_ZZ(101));
	poly_set_coeff(cipher, &m2, 6, to_ZZ(22));
	poly_set_coeff(cipher, &m2, 7, to_ZZ(127));
	
	poly_init(cipher, &m3);
	poly_set_coeff(cipher, &m3, 0, to_ZZ(42));

	// Print
	Logger::getInstance()->log_info("==========================");
	Logger::getInstance()->log_info(( "m1: " + poly_to_string(cipher, &m1)).c_str());
	Logger::getInstance()->log_info(( "m2: " + poly_to_string(cipher, &m2)).c_str());
	Logger::getInstance()->log_info(( "m3: " + poly_to_string(cipher, &m3)).c_str());

	// Copy to GPU's global memory
	poly_copy_to_device(cipher, &m1);
	poly_copy_to_device(cipher, &m2);
	poly_copy_to_device(cipher, &m3);

	/////////////
	// Encrypt //
	/////////////
	Logger::getInstance()->log_info("==========================");
	Logger::getInstance()->log_info("Will encrypt");
	cipher_t* ct1 = fv_encrypt(cipher, &m1);
	Logger::getInstance()->log_debug( ("ct1: \n" + cipher_to_string(cipher, ct1)).c_str());
	cipher_t* ct2 = fv_encrypt(cipher, &m2);
	Logger::getInstance()->log_debug( ("ct2: \n" + cipher_to_string(cipher, ct2)).c_str());
	cipher_t* ct3 = fv_encrypt(cipher, &m3);
	
	//////////
	// Mul //
	////////
	Logger::getInstance()->log_info("==========================");
	Logger::getInstance()->log_info("Will mul: m1 * m2");
	
	// Encrypted mul
	cipher_t *ctR1 = fv_mul(cipher, ct1, ct2);


	// Plaintext mul
	poly_t mR1;
	poly_init(cipher, &mR1);
	poly_mul(cipher, &mR1, &m1, &m2);

	Logger::getInstance()->log_debug( ("ctR1: \n" + cipher_to_string(cipher, ctR1)).c_str());

	//////////
	// Add //
	////////
	Logger::getInstance()->log_info("==========================");
	Logger::getInstance()->log_info("Will add: m1 * m2 + m3");
	
	// Encrypted add
	cipher_t* ctR2 = fv_add(cipher, ctR1, ct3);
	
	// Plaintext add
	poly_t mR2;
	poly_init(cipher, &mR2);
	poly_add(cipher, &mR2, &mR1, &m3);

	Logger::getInstance()->log_debug( ("ctR2: \n" + cipher_to_string(cipher, ctR2)).c_str());

	/////////////
	// Decrypt //
	/////////////
	Logger::getInstance()->log_info("==========================");
	Logger::getInstance()->log_info("Will decrypt");
	poly_t *m_decrypted = fv_decrypt(cipher, ctR2, sk);


	//////////////
	// Validate //
	/////////////
	poly_mod_by_ZZ(cipher, &mR2, &mR2, to_ZZ(cipher->get_params().t));
	Logger::getInstance()->log_info(( "m_expected: " + poly_to_string(cipher, &mR2)).c_str());
	Logger::getInstance()->log_info(( "m_decrypted: " + poly_to_string(cipher, m_decrypted)).c_str());
	Logger::getInstance()->log_info(( poly_are_equal(cipher, m_decrypted, &mR2) == true? "Success!" : "Failure =("));

	Logger::getInstance()->log_info("\n\nDone!");

	cudaDeviceSynchronize();
	cudaCheckError();
	
	poly_free(cipher, &m1);
	poly_free(cipher, &m2);
	poly_free(cipher, m_decrypted);
	cipher_free(cipher, ct1);
	cipher_free(cipher, ct2);
	cipher_free(cipher, ct3);
	cipher_free(cipher, ctR1);
	cipher_free(cipher, ctR2);
	delete cipher;
	CUDAEngine::destroy();
	cudaDeviceReset();
	cudaCheckError();
	return 0;
}