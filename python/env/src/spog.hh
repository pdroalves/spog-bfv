/**
 * SPOG
 * Copyright (C) 2017 SPOG Authors
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
#ifndef SPOG_HH
#define SPOG_HH

#include <NTL/ZZ.h>
#include <cuPoly/arithmetic/polynomial.h>
#include <cuPoly/cuda/sampler.h>
#include <cuPoly/tool/log.h>
#include <SPOG/fv.h>
#include <json.hpp>
typedef struct{
	int k;
	int t;
	int gamma;
	int mtil;
	int msk;
	int nphi;
} SPOGParams;

typedef struct{
	std::vector<std::string> data;
} SPOGPoly;

typedef struct{
	SPOGPoly *c0;
	SPOGPoly *c1;
} SPOGCipher;

class SPOG {
	private:
		FV *cipher;
		poly_t* to_poly_t(SPOGPoly *m);
		cipher_t* to_cipher_t(SPOGCipher *c);
	public:

		SPOG(SPOGParams p){
			Params params;
			params.t = p.t;
			params.gamma = p.gamma;
			params.mtil = p.mtil;
			params.msk = p.msk;
			params.nphi = p.nphi;


			// Init
			CUDAEngine::init(
				p.k,
				params.nphi,
				params.t,
				params.gamma,
				params.mtil,
				params.msk);// Init CUDA
			std::cout << "init: " << CUDAEngine::is_init << std::endl;
			params.q = CUDAEngine::RNSProduct;

			std::cout << "nphi: " << p.nphi << std::endl;
			std::cout << "q: " << params.q << std::endl;
			std::cout << "k: " << p.k << std::endl;
			std::cout << "t: " << p.t << std::endl;
			std::cout << "gamma: " << p.gamma << std::endl;
			std::cout << "msk: " << p.msk << std::endl;
			std::cout << "mtil: " << p.mtil << std::endl;

			cipher = new FV(params);
			cipher->keygen();
		}

		~SPOG(){
			if(cipher)
				delete cipher;			
			CUDAEngine::destroy();
			cudaDeviceReset();
			cudaCheckError();
		}

		SPOGCipher* encrypt(SPOGPoly *m);
		SPOGPoly* decrypt(SPOGCipher *ct);
		SPOGCipher* add(SPOGCipher *a, SPOGCipher *b);
		SPOGCipher* mul(SPOGCipher *a, SPOGCipher *b);
};

#endif