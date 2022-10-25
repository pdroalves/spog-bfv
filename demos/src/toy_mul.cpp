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

    
#include <fstream>
#include <iterator>
#include <iomanip>
#include <cuda_runtime_api.h>
#include <NTL/ZZ.h>
#include <NTL/ZZ_pX.h>
#include <NTL/ZZ_pE.h>
#include <time.h>
#include <unistd.h>
#include <iomanip>
#include <cuPoly/settings.h>
#include <cuPoly/arithmetic/polynomial.h>
#include <cuPoly/cuda/sampler.h>
#include <cuPoly/tool/version.h>
#include <SPOG/fv.h>
#include <SPOG/tool/version.h>
#include <cuda_profiler_api.h>

/**
  toy_mul:
  
  This demo reproduces the procedure on FV::mul adding measurements of CPU
  cycles for each step
 */
 int main(){
  std::cout << "cuPoly " << GET_CUPOLY_VERSION() << std::endl;
  std::cout << "SPOG " << GET_SPOG_VERSION() << std::endl << std::endl;

  cudaProfilerStop();

  Logger::getInstance()->set_mode(VERBOSE);

  srand(0);
  NTL::SetSeed(to_ZZ(0));

  // Params
  Params p;
  int k = 1;
	p.nphi = 64;
	p.t = 256;

	// Start the engine
  CUDAEngine::init(k, k + 1, p.nphi, p.t);// Init CUDA
	p.q = CUDAEngine::RNSProduct;

  // FV setup
  FVContext *cipher = new FVContext(p);
	Sampler::init(cipher);
  SecretKey *sk = fv_new_sk(cipher);
  fv_keygen(cipher, sk);

  ///////////////
  // Variables //
  ///////////////
  Logger::getInstance()->log_info("######################");
  Logger::getInstance()->log_info("Initializing variables");

  cipher_t c1, c2, c3;
  cipher_init(cipher, &c1);
  cipher_init(cipher, &c2);
  cipher_init(cipher, &c3);

  cipher_t c1_B, c2_B;
  cipher_init(cipher, &c1_B, BBase);
  cipher_init(cipher, &c2_B, BBase);

  poly_t *cstar_Q =  new poly_t[3];
  poly_t *cstar_B =  new poly_t[3];
  poly_t *cmult =  new poly_t[3];
  for (int i = 0; i < 3; i++){
    poly_init(cipher, &cstar_Q[i]);
    poly_init(cipher, &cstar_B[i], BBase);
    poly_init(cipher, &cmult[i]);
  }

  poly_t *d = new poly_t[COPRIMES_BUCKET_SIZE]; // Relin
  for(unsigned int i = 0; i < CUDAEngine::RNSPrimes.size(); i++)
    poly_init(cipher, &d[i]);
  ////////////////////////////////////////////
  // Execute the homomorphic multiplication //
  ////////////////////////////////////////////
  uint64_t start_be, start_dr2q, start_dr2b, start_cs, start_relin;
  uint64_t end_be, end_dr2q, end_dr2b, end_cs, end_relin;

  ////////////////////////////
  // Basis extension to Q*B //
  ////////////////////////////
  cudaProfilerStart();
  Logger::getInstance()->log_info("######################");
  Logger::getInstance()->log_info("Basis extension: Q to QB");
  start_be = get_cycles();

  poly_basis_extension_Q_to_B(cipher, &c1_B.c[0], &c1.c[0]);
  poly_basis_extension_Q_to_B(cipher, &c2_B.c[0], &c2.c[0]);
  poly_basis_extension_Q_to_B(cipher, &c1_B.c[1], &c1.c[1]);
  poly_basis_extension_Q_to_B(cipher, &c2_B.c[1], &c2.c[1]);

  cudaDeviceSynchronize();
  cudaCheckError();
  cudaProfilerStop();
  end_be = get_cycles();

  ////////////////////////
  // Compute DR2(c1,c2) //
  ////////////////////////
  Logger::getInstance()->log_info("######################");
  Logger::getInstance()->log_info("DR2_Q");
  cudaProfilerStart();
  start_dr2q = get_cycles();


  poly_dr2(
    cipher,
    &cipher->cstar_Q[0],
    &cipher->cstar_Q[1],
    &cipher->cstar_Q[2],
    &c1.c[0],
    &c1.c[1],
    &c2.c[0],
    &c2.c[1]);
  
  cudaDeviceSynchronize();
  cudaCheckError();
  cudaProfilerStop();
  end_dr2q = get_cycles();

  Logger::getInstance()->log_info("######################");
  Logger::getInstance()->log_info("DR2_B");
  cudaProfilerStart();
  start_dr2b = get_cycles();


  poly_dr2(
    cipher,
    &cipher->cstar_Q[0],
    &cipher->cstar_Q[1],
    &cipher->cstar_Q[2],
    &c1.c[0],
    &c1.c[1],
    &c2.c[0],
    &c2.c[1]);

  cudaDeviceSynchronize();
  cudaCheckError();
  cudaProfilerStop();
  end_dr2b = get_cycles();

  //////////////////////////////////////
  // Scaling and basis extension to Q //
  //////////////////////////////////////
  cudaProfilerStart();
  Logger::getInstance()->log_info("######################");
  Logger::getInstance()->log_info("Complex scaling");
  start_cs = get_cycles();

  for (int i = 0; i < 3; i++)
    poly_complex_scaling_tDivQ(
      cipher,
      &cmult[i], 
      &cstar_Q[i], 
      &cstar_B[i]);

  cudaDeviceSynchronize();
  cudaCheckError();
  cudaProfilerStop();
  end_cs = get_cycles();

  /////////////////////
  // Relinearization //
  /////////////////////
  cudaProfilerStart();
  Logger::getInstance()->log_info("######################");
  Logger::getInstance()->log_info("Relinearization");
  start_relin = get_cycles();

  fv_relin(
    cipher,
    &c3,
    cmult,
    &cipher->evk);

  cudaDeviceSynchronize();
  cudaCheckError();
  cudaProfilerStop();
  end_relin = get_cycles();

  Logger::getInstance()->log_info("######################");
  Logger::getInstance()->log_info("Done");

  ////////////
  // Output //
  ////////////
  Logger::getInstance()->log_info("######################");
  float total = 
  (end_be - start_be)     +
  (end_dr2q - start_dr2q) +
  (end_dr2b - start_dr2b) +
  (end_cs - start_cs)     +
  (end_relin - start_relin);

  std::cout << (end_be - start_be) <<
  " cycles for basis extension (" << (end_be - start_be)/total << ")" << std::endl;
  std::cout << (end_dr2q - start_dr2q) <<
  " cycles for DR2_Q (" << (end_dr2q - start_dr2q)/total << ")" << std::endl;
  std::cout << (end_dr2b - start_dr2b) <<
  " cycles for DR2_B (" << (end_dr2b - start_dr2b)/total << ")" << std::endl;
  std::cout << (end_cs - start_cs) <<
  " cycles for complex scaling (" << (end_cs - start_cs)/total << ")" << std::endl;
  std::cout << (end_relin - start_relin) <<
  " cycles for relin (" << (end_relin - start_relin)/total << ")" << std::endl;

  /////////////
	// Release //
  /////////////

	cudaDeviceSynchronize();
	cudaCheckError();

  cipher_free(cipher, &c1);
  cipher_free(cipher, &c2);
  cipher_free(cipher, &c3);

  for (int i = 0; i < 3; i++){
    poly_free(cipher, &cstar_Q[i]);
    poly_free(cipher, &cstar_B[i]);
    poly_free(cipher, &cmult[i]);
  }
  for(unsigned int i = 0; i < CUDAEngine::RNSPrimes.size(); i++)
    poly_free(cipher, &d[i]);

  cipher_free(cipher, &c1_B);
  cipher_free(cipher, &c2_B);
  delete [] cstar_Q;
  delete [] cstar_B;
  delete [] cmult;

	delete cipher;

	CUDAEngine::destroy();
	Sampler::destroy();

	cudaDeviceReset();
	cudaCheckError();
}
