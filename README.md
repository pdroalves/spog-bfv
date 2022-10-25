# SPOG - Secure Processing On GPGPUs

[University of Campinas](http://www.unicamp.br), [Institute of Computing](http://www.ic.unicamp.br), Brazil.

Laboratory of Security and Cryptography - [LASCA](http://www.lasca.ic.unicamp.br),<br>
Multidisciplinary High Performance Computing Laboratory - [LMCAD](http://www.lmcad.ic.unicamp.br). <br>

Author: [Pedro G. M. R. Alves](http://www.iampedro.com), PhD. candidate @ IC-UNICAMP,<br/>

## About

SPOG is a proof of concept for our work looking for efficient techniques to implement RLWE-based HE cryptosystems on GPUs. It works together with [cuPoly](https://github.com/pdroalves/cuPoly).

## Goal
SPOG and cuPoly are ongoing projects and we hope to increase its performance and security in the course of time. Our focus is to provide:

- Exceptional performance on modern GPGPUs.
- An easy-to-use high-level API.
- Easily maintainable code. Easy to fix bugs and easy to scale.
- A model for implementations of cryptographic schemes based on RLWE.

## Installing

Note that `stable` is generally a work in progress, and you probably want to use a tagged release version.

### Dependencies
SPOG was tested in a Linux environment with the following packages:

| Package | Version |
| ------ | ------ |
| g++ | 8.4.0 |
| CUDA | 11.0 |
| cmake | 3.13.3 |
| googletest | v1.10.0 |
| [rapidjson](https://github.com/Tencent/rapidjson) | v1.1.0 | 
| [NTL](https://www.shoup.net/ntl/) | 11.3.2 |
| [gmp](https://gmplib.org/) | 6.1.2 |

### Procedure

1) Download and install [cuPoly](https://github.com/pdroalves/cuPoly). Be careful to choose a branch compatible with the one you intend to use on SPOG (stable or unstable).
2) Download and unzip the most recent commit of SPOG. The "stable" branch shall store the most recent version that was approved on all relevant tests. In the "unstable" branch you will ideas we are working on and may break something. Other branches are feature-related and may be merged some day.
2) Create spog/build.
3) Change to spog/build and run 
```
$ cmake ..
$ make
```

cmake will verify the environment to assert that all required packages are installed.

### Tests and benchmarks

SPOG contains binaries for testing and benchmarking. For testing, use spog_test. Since it is built over googletest you may apply their filters to select tests of interest. For instance,

```
./spog_test --gtest_filter=FVInstantiation/TestFV.Mul/4096*
```

runs all tests for homomorphic multiplication on a cyclotomic polynomial ring of degree 4096.

## How to use?

SPOG provides a high-level API, avoiding the need for the programmer to interact with the GPGPU. SPOG requires an initial setup to define the parameters used by BFV and nothing else. [cuPoly](https://github.com/pdroalves/cupoly). is a sister library that provides all the required arithmetic.

There is an embryonic version of a documentation made with doxygen, but it is not up-to-date or even complete. The most up-to-date source to understand how to use SPOG and cuPoly is by looking at the demos provided and the test suite.

## Parameters

The basic parameters needed are 

```c++
	Params p;
	p.nphi = N; // ring degree
	p.k = |q|; // size of the main base 
	p.kl = k + 1; // size of the auxiliar base
	p.t = |m|; // defines the plaintext domain
```

## Initialization

All CUDA kernels calls are made and handled by [CUDAEngine](https://github.com/pdroalves/cuPoly/blob/master/include/cuPoly/cuda/cudaengine.h). Thus, before anything this object must be initialized using the selected parameters.


```c++
    CUDAEngine::init(k, kl, nphi, t);// Init CUDA
```

The Homomorphic cryptosystem used by SPOG is [BFV](https://eprint.iacr.org/2012/144). Hence, we use polynomial arithmetic over a cyclotomic ring. Such operations are provided by cuPoly in the form of [poly_t struct](https://github.com/pdroalves/cuPoly/blob/master/include/cuPoly/arithmetic/polynomial.h).   
    
```c++
	/////////////
	// Message //
	/////////////
	poly_t m1, m2, m3;
	poly_set_coeff(&m1, 0, to_ZZ(1));	
	poly_set_coeff(&m1, 1, to_ZZ(1));
	poly_set_coeff(&m2, 1, to_ZZ(1));
	poly_set_coeff(&m3, 1, to_ZZ(42));
```

To work with BFV, once CUDAEngine is on we need to instantiate a BFVContext object, the Sampler singleton, and generate a key set. 

```c++
	// BFV setup
	BFVContext* cipher = new BFVContext(p);
	Sampler::init(cipher);

	SecretKey *sk = bfv_new_sk(cipher);
	Keys* keys = bfv_keygen(cipher, sk);

```
  
## BFV Primitives

Encryption, decryption, and homomorphic operations are executed by BFV's methods.

```c++
	poly_t m1, m2, m3;
	poly_init(cipher, &m1);
	poly_init(cipher, &m2);
	poly_init(cipher, &m3);

	// initializes m1, m2, and m3
	// ...
	// 
	
	/////////////
	// Encrypt //
	/////////////
	cipher_t* ct1 = bfv_encrypt(cipher, &m1);
	cipher_t* ct2 = bfv_encrypt(cipher, &m2);
	cipher_t* ct3 = bfv_encrypt(cipher, &m3);

    //////////
	// Add //
	////////
	cipher_t *ctR1 = bfv_add(cipher, ct1, ct2);

	//////////
	// Mul //
	////////	
	cipher_t* ctR2 = bfv_mul(cipher, ctR1, ct3);
  
	/////////////
	// Decrypt //
	/////////////
	/// The secret key is not stored in BFVContext, so you need to pass it as a parameter
	poly_t *m_decrypted = bfv_decrypt(cipher, ctR2, sk);
```

Once we are done, CUDAEngine must be destroyed and the objects created must be released.

```c++
	// Asserts there is anything being computed on the GPU
	cudaDeviceSynchronize();
	cudaCheckError();

	// Release polynomials and ciphers
    poly_free(&m1);
    poly_free(&m2);
    poly_free(&m3);
    poly_free(m_decrypted);

    cipher_free(ct1);
    cipher_free(ct2);
    cipher_free(ct3);
    cipher_free(ctR1);
    cipher_free(ctR2);

    // Release keys
    keys_free(cipher, keys);
    delete keys;

    // Release the BFVContext and the singletons
    delete cipher;
	Sampler::destroy();
	CUDAEngine::destroy();

```

## Citing

If you use SPOG/cuPoly, please cite using the template below:

to-do


## Disclaimer
SPOG is at most alpha-quality software. Implementations may not be correct or secure. Moreover, it was not tested with FV parameters different from those in the test file. Use at your own risk.

## Licensing

SPOG is released under GPLv3.

**Privacy Warning:** This site tracks visitor information.