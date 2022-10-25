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
#include <SPOG/fv.h>
#include <cuda_profiler_api.h>

#define BILLION  1000000000L
#define MILLION  1000000L
#define NITERATIONS 1


/**
  simple_mul:

  The intention of this program is to explore the homomorphic multiplication on
  SPOG from the point of view of memory managing.
  
  Here we use a Params object to setup and instantiate a FV object.
 */
 int main(){
  Logger::getInstance()->set_mode(INFO);
  cudaProfilerStop();

  FVContext *cipher;
  ZZ q;

  srand(0);
  NTL::SetSeed(to_ZZ(0));

  // Params
  Params p;
  p.nphi = 64;
  p.t = 256;
  int k = 2;

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

  //Sampler::sample(cipher, &m1, UNIFORM);
  //Sampler::sample(cipher, &m2, UNIFORM);
  poly_set_coeff(cipher, &m1, 0, to_ZZ(42));
  poly_set_coeff(cipher, &m2, 0, to_ZZ(0));

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
  
  fv_mul(cipher, ct3, ct1, ct2);

  /////////////
  // Decrypt //
  /////////////
  
  poly_t *m3 = fv_decrypt(cipher, ct3, sk);

  std::cout << "Input A: " << poly_to_string(cipher, &m1) << std::endl;
  std::cout << "Input B: " << poly_to_string(cipher, &m2) << std::endl;
  std::cout << "Recovered: " << poly_to_string(cipher, m3) << std::endl;
  
  /////////////
  
  poly_free(cipher, &m1);
  poly_free(cipher, &m2);
  poly_free(cipher, m3);
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
