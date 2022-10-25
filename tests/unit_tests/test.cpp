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

#include <stdlib.h>
#include <gtest/gtest.h>
#include <cuPoly/settings.h>
#include <cuPoly/arithmetic/polynomial.h>
#include <cuPoly/tool/version.h>
#include <SPOG-BFV/tool/version.h>
#include <SPOG-BFV/bfvcontext.h>
#include <SPOG-BFV/bfv.h>
#include <NTL/ZZ.h>
#include <NTL/ZZ_p.h>
#include <NTL/ZZ_pEX.h>

typedef struct{
	int logq2;
	uint64_t t;
	int nphi;
} TestParams;

const int LOGLEVEL = QUIET;
const int NTESTS = 100;

class TestBFV : public ::testing::TestWithParam<TestParams> {
	protected:
    BFVContext* cipher;
	ZZ_pX NTL_Phi;
	Keys* keys;
	SecretKey *sk;   

	public:
	__host__ void SetUp(){
	    BFVParams p;
		srand(0);
		NTL::SetSeed(to_ZZ(0));
		Logger::getInstance()->set_mode(LOGLEVEL);

		// TestParams
		int k = ceil((float)GetParam().logq2 / 63);
		int nphi = GetParam().nphi;
		uint64_t t = GetParam().t;

		// Init
		CUDAEngine::init(k, k + 1, nphi, t);// Init ClUDA

		p.q = CUDAEngine::RNSProduct;		
        p.t = t;
		p.nphi = nphi;
		ZZ_p::init(p.q);
		NTL::SetCoeff(NTL_Phi,0,conv<ZZ_p>(1));
		NTL::SetCoeff(NTL_Phi, nphi, conv<ZZ_p>(1));
		ZZ_pE::init(NTL_Phi);

		// FV setup
        cipher = new BFVContext(p);
        Sampler::init(cipher);

		sk = bfv_new_sk(cipher);
        keys = bfv_keygen(cipher, sk);

		cudaStreamSynchronize(cipher->get_stream());
		cudaCheckError();

	}

	__host__ void TearDown(){
        keys_free(cipher, keys);
        delete cipher;
        delete keys;

		cudaDeviceSynchronize();
		cudaCheckError();

		Sampler::destroy();
		CUDAEngine::destroy();

		cudaDeviceReset();
		cudaCheckError();
	}
};


TEST_P(TestBFV, EncryptDecrypt)
{
	for(int N = 0; N < NTESTS; N++){
		/////////////
		// Message //
		/////////////
		poly_t m;
		poly_init(cipher, &m);

		//////////////
		// Sampling //
		//////////////
		for(int i = 0; i < 2*CUDAEngine::N; i++)
			poly_set_coeff(cipher, &m, i, to_ZZ(rand() % cipher->get_params().t));
		
		/////////////
		// Encrypt //
		/////////////
		cipher_t* ct = bfv_encrypt(cipher, &m);
		poly_t *m_decrypted = bfv_decrypt(cipher, ct, sk);

		///////////
		// Check //
		///////////

		for(int i = 0; i < CUDAEngine::N; i++){
			ZZ e = poly_get_coeff(cipher, &m, i) % to_ZZ(cipher->get_params().t);
			ZZ r = poly_get_coeff(cipher, m_decrypted, i) % to_ZZ(cipher->get_params().t);

			ASSERT_EQ(r, e)  << "Fail for the " << i << "-th coefficient (diff: " << (r - e) <<	", t: " << cipher->get_params().t << ")";
		}

		ASSERT_EQ(
			poly_get_deg(cipher, m_decrypted),
			poly_get_deg(cipher, &m));

		poly_free(cipher, &m);
		poly_free(cipher, m_decrypted);
		cipher_free(cipher, ct);
	}
}


TEST_P(TestBFV, Add)
{        
	for(int N = 0; N < NTESTS; N++){
		/////////////
		// Message //
		/////////////
		poly_t m1, m2;
		poly_init(cipher, &m1);
		poly_init(cipher, &m2);

		//////////////
		// Sampling //
		//////////////
		for(int i = 0; i < 2*CUDAEngine::N; i++){
			poly_set_coeff(cipher, &m1, i, to_ZZ(rand() % cipher->get_params().t));
			poly_set_coeff(cipher, &m2, i, to_ZZ(rand() % cipher->get_params().t));
		}

		/////////////
		// Encrypt //
		/////////////
		cipher_t *ct1 = bfv_encrypt(cipher, &m1);
		cipher_t *ct2 = bfv_encrypt(cipher, &m2);

		/////////
		// Add //
		/////////
		cipher_t *ct3 = bfv_add(cipher, ct1, ct2);

		//////////////
		// Decrypt  //
		//////////////
		poly_t *m3 = bfv_decrypt(cipher, ct3, sk);

		///////////
		// Check //
		///////////
		for(int i = 0; i < CUDAEngine::N; i++){
			ZZ e = (poly_get_coeff(cipher, &m1, i) + poly_get_coeff(cipher, &m2, i)) % to_ZZ(cipher->get_params().t);
			ZZ r = poly_get_coeff(cipher, m3, i);

			ASSERT_EQ(r, e)  << "Fail for the " << i << "-th coefficient (diff: " << (r - e) <<	")";
		}

		poly_free(cipher, &m1);
		poly_free(cipher, &m2);
		poly_free(cipher, m3);
		cipher_free(cipher, ct1);
		cipher_free(cipher, ct2);
		cipher_free(cipher, ct3);
	}
}


TEST_P(TestBFV, Mul)
{
	ZZ_p::init(to_ZZ(cipher->get_params().t));
	for(int N = 0; N < NTESTS; N++){
		/////////////
		// Message //
		/////////////
		poly_t m1, m2;
		poly_init(cipher, &m1);
		poly_init(cipher, &m2);

		//////////////
		// Sampling //
		//////////////
		for(int i = 0; i < 2 * CUDAEngine::N; i++){
			poly_set_coeff(cipher, &m1, i, to_ZZ(rand() % cipher->get_params().t));
			poly_set_coeff(cipher, &m2, i, to_ZZ(rand() % cipher->get_params().t));
		}

		//////////////
		// Init NTL //
		//////////////
		ZZ_pX ntl_m1, ntl_m2, ntl_m3;
		
		for(int i = 0; i <= poly_get_deg(cipher, &m1); i++)
			SetCoeff( ntl_m1, i, conv<ZZ_p>(poly_get_coeff(cipher, &m1, i)));
		for(int i = 0; i <= poly_get_deg(cipher, &m2); i++)
			SetCoeff( ntl_m2, i, conv<ZZ_p>(poly_get_coeff(cipher, &m2, i)));

		/////////////
		// Encrypt //
		/////////////
		cipher_t *ct1 = bfv_encrypt(cipher, &m1);
		cipher_t *ct2 = bfv_encrypt(cipher, &m2);

		cudaDeviceSynchronize();
		cudaCheckError();

		/////////
		// Mul //
		/////////
		cipher_t *ct3 = bfv_mul(cipher, ct1, ct2);
		ntl_m3 = ntl_m1 * ntl_m2 % NTL_Phi;

		//////////////
		// Decrypt  //
		//////////////
		poly_t *m3 = bfv_decrypt(cipher, ct3, sk);

		///////////
		// Check //
		///////////
		//std::cout << "a = polynomial(" << poly_to_string(cipher, &m1) << std::endl;
		//std::cout << "b = polynomial(" << poly_to_string(cipher, &m2) << std::endl;
		//std::cout << "c = polynomial(" << poly_to_string(cipher, m3) << std::endl;
		ZZ c = to_ZZ(0);
		for(int i = 0; i < 2*CUDAEngine::N; i++){
			ZZ e = conv<ZZ>(coeff(ntl_m3, i)) % to_ZZ(cipher->get_params().t);
			ZZ r = poly_get_coeff(cipher, m3, i);

			ASSERT_EQ(r, e) << N << ") Fail for the " << i << "-th coefficient (diff: " << (r - e) << ")";
			c += r;
		}
		ASSERT_GT(c, to_ZZ(0));

		poly_free(cipher, &m1);
		poly_free(cipher, &m2);
		poly_free(cipher, m3);
		cipher_free(cipher, ct1);
		cipher_free(cipher, ct2);
		cipher_free(cipher, ct3);
	}
	ZZ_p::init(CUDAEngine::RNSProduct);
}

TEST_P(TestBFV, PlainMul)
{
	ZZ_p::init(to_ZZ(cipher->get_params().t));
	for(int N = 0; N < NTESTS; N++){
		/////////////
		// Message //
		/////////////
		poly_t m1, m2;
		poly_init(cipher, &m1);
		poly_init(cipher, &m2);

		//////////////
		// Sampling //
		//////////////
		for(int i = 0; i < 2 * CUDAEngine::N; i++){
			poly_set_coeff(cipher, &m1, i, to_ZZ(rand() % cipher->get_params().t));
			poly_set_coeff(cipher, &m2, i, to_ZZ(rand() % cipher->get_params().t));
		}

		//////////////
		// Init NTL //
		//////////////
		ZZ_pX ntl_m1, ntl_m2, ntl_m3;

		for(int i = 0; i <= poly_get_deg(cipher,&m1); i++)
			SetCoeff(ntl_m1, i, conv<ZZ_p>(poly_get_coeff(cipher,&m1, i)));
		for(int i = 0; i <= poly_get_deg(cipher,&m2); i++)
			SetCoeff(ntl_m2, i, conv<ZZ_p>(poly_get_coeff(cipher,&m2, i)));

		/////////////
		// Encrypt //
		/////////////
		cipher_t *ct1 = bfv_encrypt(cipher, &m1);

		/////////
		// Mul //
		/////////
        cipher_t *ct3 = bfv_plainmul(cipher, ct1, &m2);
		ntl_m3 = ntl_m1 * ntl_m2 % NTL_Phi;

		//////////////
		// Decrypt  //
		//////////////
		poly_t *m3 = bfv_decrypt(cipher, ct3, sk);

		///////////
		// Check //
		///////////
		// ASSERT_EQ(poly_get_deg(m3), deg(ntl_m3));
		ZZ c = to_ZZ(0);
		for(int i = 0; i < 2 * CUDAEngine::N; i++){
			ZZ e = conv<ZZ>(coeff(ntl_m3, i)) % to_ZZ(cipher->get_params().t);
			ZZ r = poly_get_coeff(cipher, m3, i);

			ASSERT_EQ(r, e)  << "Fail for the " << i << "-th coefficient (diff: " << (r - e) <<	")";
			c += r;
		}
		ASSERT_GT(c, to_ZZ(0));

		poly_free(cipher, &m1);
		poly_free(cipher, &m2);
		poly_free(cipher, m3);
		cipher_free(cipher, ct1);
		cipher_free(cipher, ct3);
	}
	ZZ_p::init(CUDAEngine::RNSProduct);
}

// TEST_P(TestSPOG, ImportExportKeys)
// {
		
// 	for(int N = 0; N < NTESTS; N++){
// 		// Export
// 		json jkeys = cipher->export_keys();

// 		// Encrypt something
// 		poly_t m;
// 		Sampler::sample(cipher, &m, UNIFORM);
// 		poly_mod_by_ZZ(cipher, &m, &m, to_ZZ(cipher->get_params().t));
// 		// std::cout << poly_to_string(&m) << std::endl;

// 		cipher_t* ct = bfv_encrypt(cipher, m);

// 		// Clear keys
// 		cipher->clear_keys();

// 		// Import
// 		cipher->load_keys(jkeys);

// 		// Verifies if we are still able to decrypt
// 		// std::cout << poly_to_string(bfv_decrypt(cipher, *ct, sk)) << std::endl;
// 		ASSERT_TRUE(poly_are_equal(&m, bfv_decrypt(cipher, *ct, sk)));
// 	}
// }

//
//Defines for which parameters set cuPoly will be tested.
//It executes each test for all pairs on phis X qs (Cartesian product)
::testing::internal::ParamGenerator<TestParams> params = ::testing::Values(
	//      {   logq2, 	     t, nphi},
	// (TestParams){20, 256, 32},
	(TestParams){130, 256, 32},
	(TestParams){130, 256, 64},
	(TestParams){130, 256, 2048},
	(TestParams){186, 256, 4096},
	(TestParams){372, 256, 8192}
	//(TestParams){340, 256, 16384} //mul test fails for 340, 512, 16384 (so can't do the multi_t tests!)
	);
std::string printParamName(::testing::TestParamInfo<TestParams> p){
	TestParams params = p.param;

	return std::to_string(params.nphi) +
	"_q" + std::to_string(params.logq2) +
	"_t" + std::to_string(params.t);
}

INSTANTIATE_TEST_CASE_P(SPOGBFVInstantiation,
	TestBFV,
	params,
	printParamName
);


int main(int argc, char **argv) {
  //////////////////////////
  ////////// Google tests //
  //////////////////////////
  std::cout << "Testing cuPoly " << GET_CUPOLY_VERSION() << std::endl;
  std::cout << "Testing SPOG " << GET_SPOGBFV_VERSION() << std::endl;
  std::cout << "Running " << NTESTS << std::endl << std::endl;
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
