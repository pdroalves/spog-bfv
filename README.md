# SPOG-BFV - Secure Processing On GPGPUs

[University of Campinas](http://www.unicamp.br), [Institute of Computing](http://www.ic.unicamp.br), Brazil.

Laboratory of Security and Cryptography - [LASCA](http://www.lasca.ic.unicamp.br),<br>
Multidisciplinary High Performance Computing Laboratory - [LMCAD](http://www.lmcad.ic.unicamp.br). <br>

Author: [Pedro G. M. R. Alves](http://www.iampedro.com), PhD. candidate @ IC-UNICAMP,<br/>

## About

SPOG-BFV is a proof of concept for our work looking for efficient techniques to implement RLWE-based HE cryptosystems on GPUs. It implements the BFV cryptosystem on top of [cuPoly](https://github.com/spog-library/cuPoly).

## Goal

SPOG-BFV is an ongoing project and we hope to increase its performance and security in the course of time. Our focus is to provide:

- Exceptional performance on modern GPGPUs.
- An easy-to-use high-level API.
- Easily maintainable code. Easy to fix bugs and easy to scale.
- A model for implementations of cryptographic schemes based on RLWE.

## Disclaimer 

This is not a work targetting use in production. Thus, non-standard cryptographic implementation decisions may have been taken to simulate a certain behavior that would be expected in a production-oriented library. For instance, we mention using a truncated Gaussian distribution, built over cuRAND's Gaussian distribution, instead of a truly discrete gaussian. The security of this decision still needs to be asserted.

SPOG-BFV is at most alpha-quality software. Implementations may not be correct or secure. Moreover, it was not tested with BFV parameters different from those in the test file. Use at your own risk.

## Installing

Note that `stable` is generally a work in progress, and you probably want to use a [tagged release version](https://github.com/spog-library/spogbfv/releases).

### Dependencies
SPOG-BFV was tested in a GNU/Linux environment with the following packages:

| Package | Version |
| ------ | ------ |
| g++ | 8.4.0 |
| CUDA | 11.0 |
| cmake | 3.13.3 |
| cuPoly | v0.3.4 |
| googletest | v1.10.0 |
| [rapidjson](https://github.com/Tencent/rapidjson) | v1.1.0 | 
| [NTL](https://www.shoup.net/ntl/) | 11.3.2 |
| [gmp](https://gmplib.org/) | 6.1.2 |

### Procedure

1) Download and install [cuPoly](https://github.com/spog-library/cuPoly). Be careful to choose a branch compatible with the one you intend to use on SPOG-BFV (stable or unstable).
2) Download and unzip the most recent commit of SPOG-BFV. The "stable" branch shall store the most recent version that was approved on all relevant tests. In the "unstable" branch you will ideas we are working on and may break something. Other branches are feature-related and may be merged some day.
2) Create spogbfv/build.
3) Change to spogbfv/build and run 

```
$ cmake ..
$ make
```

cmake will verify the environment to assert that all required packages are installed.

### Tests

SPOG-BFV contains binaries for testing and benchmarking. For testing, use spogbfv_test. Since it is built over googletest you may apply their filters to select tests of interest. For instance,

```
./spogbfv_test --gtest_filter=BFVInstantiation/TestBFV.Mul/4096*
```

runs all tests for homomorphic multiplication on a cyclotomic polynomial ring of degree 4096.

### Benchmarks

Our official benchmark tool is [SPOGBFV-Benchmark](https://github.com/spog-library//SPOGBFVBenchmark)

## How to use?

There is an embryonic version of a documentation made with doxygen. The most up-to-date source to understand how to use SPOG and cuPoly is by looking at the demos provided and the test suite.

SPOG-BFV provides a high-level API, avoiding the need for the programmer to interact with the GPGPU. SPOG-BFV requires an initial setup to define the parameters used by BFV and nothing else. [cuPoly](https://github.com/spog-library/cupoly) is a sister library that provides all the required arithmetic.

## Include

```c++

#include <cuPoly/arithmetic/polynomial.h> // Polynomial arithmetic
#include <cuPoly/cuda/sampler.h> // Sampling mechanism
#include <SPOG-BFV/bfv.h> // BFV's methods

```

## Parameters

The basic parameters needed are 

```
	nphi; // ring degree
	k = |q|; // number of residues of the main base
	kl = |b|; // number of residues of the auxiliar base
	t; // defines the plaintext domain
```

## Initialization

All CUDA kernels calls are made and handled by [CUDAEngine](https://github.com/spog-library/cuPoly/include/cuPoly/cuda/cudaengine.h). Thus, before anything this object must be initialized using the selected parameters.


```c++
    CUDAEngine::init(k, kl, nphi, t);// Init CUDA
```

The Homomorphic cryptosystem used by SPOG-BFV is [BFV](https://eprint.iacr.org/2012/144). Hence, we use polynomial arithmetic over a cyclotomic ring. Such operations are provided by cuPoly in the form of [poly_t struct](https://github.com/spog-library/cuPoly/include/cuPoly/arithmetic/polynomial.h). NTL is used to handle the multiprecision integers at the CPU-side, so inputs must be [ZZ](https://libntl.org/doc/ZZ.cpp.html) elements.
    
```c++
	/////////////
	// Message //
	/////////////
	poly_t m1, m2, m3;

	// The polynomial, index, and the value for that index (as a ZZ)
	poly_set_coeff(&m1, 0, to_ZZ(1));
	poly_set_coeff(&m1, 1, to_ZZ(1));
	poly_set_coeff(&m2, 1, to_ZZ(1));
	poly_set_coeff(&m3, 1, to_ZZ(42));
```

Once CUDAEngine is initiated, we need to instantiate a BFVContext object, the Sampler singleton, and generate a key set. 

```c++
	// BFV setup
	BFVContext* cipher = new BFVContext(p);
	Sampler::init(cipher);

	SecretKey *sk = bfv_new_sk(cipher);
	Keys* keys = bfv_keygen(cipher, sk);

```

The Keys struct has a SecretKey attribute that is not filled by ``bfv_keygen``. The idea is to keep the secret key separated from the other keys. However, ``Keys->sk`` may be used for application-specific reasons.  

## BFV Primitives

Encryption, decryption, and homomorphic operations are executed by BFV's methods.

```c++
	poly_t m1, m2, m3;

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

	// Print
  	std::cout << "The outcome: " << poly_to_string(cipher, m_decrypted) << std::endl;
  	std::cout << "The third coefficient: " << poly_get_coeff(cipher, m_decrypted, 2) << std::endl;

```

Once we are done, CUDAEngine must be destroyed and the objects created must be released.

```c++
	// Asserts there is anything being computed on the GPU
	cudaDeviceSynchronize();
	cudaCheckError();

	// Release polynomials and ciphers
    poly_free(cipher, &m1);
    poly_free(cipher, &m2);
    poly_free(cipher, &m3);
    poly_free(cipher, m_decrypted);

    cipher_free(cipher, ct1);
    cipher_free(cipher, ct2);
    cipher_free(cipher, ct3);
    cipher_free(cipher, ctR1);
    cipher_free(cipher, ctR2);

    // Release keys
    keys_free(cipher, keys);
    keys_sk_free(cipher, sk);
    delete keys;

    // Release the BFVContext and the singletons
    delete cipher;
	Sampler::destroy();
	CUDAEngine::destroy();

```

## Citing

If you use SPOG/cuPoly, please cite using the template below:

```
@misc{cryptoeprint:2020:861,
    author = {Pedro Geraldo M. R. Alves and Jheyne N. Ortiz and Diego F. Aranha},
    title = {Faster Homomorphic Encryption over GPGPUs via hierarchical DGT},
    howpublished = {Cryptology ePrint Archive, Report 2020/861},
    year = {2020},
    note = {\url{https://eprint.iacr.org/2020/861}},
}
```

## Licensing

SPOG-BFV is released under GPLv3.

**Privacy Warning:** This site tracks visitor information.