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

/**
  simple_mul:

  The intention of this program is to explore the homomorphic multiplication on
  SPOG from the point of view of memory managing.
  
  Here we use a BFVParams object to setup and instantiate a BFV object.
 */

int main() {
  BFVContext *cipher;

  // Params
  BFVParams p;
  p.nphi = 4096;
  p.t = 256;
  int k = 1;

  // Init
  CUDAEngine::init(k, k + 1, p.nphi, p.t);// Init CUDA
  p.q = CUDAEngine::RNSProduct;

  // BFV setup
  cipher = new BFVContext(p);
  Sampler::init(cipher);
  SecretKey *sk = bfv_new_sk(cipher);
  bfv_keygen(cipher, sk);

  /////////////
  // Message //
  /////////////
  poly_t m1, m2;
  Sampler::sample(cipher, &m1, UNIFORM);
  Sampler::sample(cipher, &m2, UNIFORM);

  /////////////
  // Encrypt //
  /////////////
  Logger::getInstance()->log_info("==========================");
  Logger::getInstance()->log_info("Will encrypt");
  cipher_t* ct1 = bfv_encrypt(cipher, &m1);
  cipher_t* ct2 = bfv_encrypt(cipher, &m2);
  cipher_t* ct3 = new cipher_t;
  cipher_init(cipher, ct3);

  //////////
  // Add //
  ////////
  Logger::getInstance()->log_info("Will Mul");
  bfv_mul(cipher, ct3, ct1, ct2);
  Logger::getInstance()->log_info("==========================");
  
  ///////////// 
  // Release //
  ///////////// 
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
