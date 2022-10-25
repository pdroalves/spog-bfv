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
// 
#ifndef BFVContext_H
#define BFVContext_H

#define ntl_safe_scaling(A, B) to_ZZ(NTL::round(to_RR(A) / to_RR(B)))

#include <cuPoly/arithmetic/context.h>
#include <cuPoly/cuda/sampler.h>
#include <SPOG-BFV/arithmetic/ciphertext.h>

#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
using namespace rapidjson;
typedef Document json;

/**
 * @brief       A pack of BFV's parameters that can be used to initialize BFVContext
 */
typedef struct{
    ZZ q;
    uint64_t t;
    int nphi;
} BFVParams;

/////////////////////
// Keys            //
/////////////////////

/**
 * \brief The secret key
 */
typedef struct{
    poly_t s;
} SecretKey;

/**
 * \brief The public key
 */
typedef struct{
    poly_t b;
    poly_t a;
} PublicKey;

/**
 * \brief Stores a evk
 */
typedef struct{
    poly_t *b = NULL;
    poly_t *a = NULL;
} EvaluationKey; 

/**
 * \brief A pack with all types of keys
 */
typedef struct{
    SecretKey sk;
    PublicKey pk;
    EvaluationKey evk;
} Keys;

__host__ void keys_init(Context *ctx, Keys *k); //!< Initialize a Keys struct
__host__ void keys_free(Context *ctx, Keys *k); //!< Releases the memory of a Keys struct
__host__ void keys_sk_free(Context *ctx, SecretKey *sk); //!< Releases the memory of a SecretKey struct

/**
 * @brief      This is a specialization of Context focused on handling BFV's related data 
 */
class BFVContext : public Context{

    private:
        BFVParams params;

        std::vector<Context*> alt_ctxs; //!< Alternative contexts at base Q
        std::vector<Context*> alt_b_ctxs; //!< Alternative contexts at base B
        
    public: 
        // Using delta as a polynomial implies in a much faster and simpler
        // multiplication during encryption.
        poly_t delta; //!< q/t used in encrypt

        poly_t *cstar_Q; //!< Stores the product of c1 by c2 in base q
        poly_t *cstar_B; //!< Stores the product of c1 by c2 in base b
        poly_t *cmult; //!< Stores the scaled and rounded product of c1 by c2 in base q
    
        poly_t *u, *e1, *e2; //!< Sampled polynomial used in encryption

        cipher_t *c1_B; //!< The representation in base B of the first operand in a hom. mul.
        cipher_t *c2_B; //!< The representation in base B of the second operand in a hom. mul.

        // BFV decrypt
        poly_t *m_aux; //!< Temporary storage of |(c0 + c1*s)|_q 

        Keys *keys;//!< The public keys used to encrypt and relinearize

        poly_t *d = new poly_t[COPRIMES_BUCKET_SIZE]; //!< Temporary array used in relinearization


        /////////////////////////////////////////////
        // BFV operations are done on a context //
        /////////////////////////////////////////////
        BFVContext( ZZ q,
            uint64_t t,
            int nphi){
            ////////////////
            // Parameters //
            ////////////////
            params.q = q;
            params.t = t;
            params.nphi = nphi;

            //////////////
            // Pre-comp //
            //////////////
            // Auxiliar contexts
            for(unsigned int i = 0; i < 3; i++)
                alt_ctxs.push_back(new Context());

            poly_set_coeff(this, &delta, 0, params.q / params.t);
            poly_copy_to_device(this, &delta);
            cstar_Q =  new poly_t[3];
            cstar_B =  new poly_t[3];
            cmult =  new poly_t[3];

            u = new poly_t;
            e1 = new poly_t;
            e2 = new poly_t;

            c1_B = new cipher_t;
            c2_B = new cipher_t;

            ///////////
            // Init  //
            ///////////
            cipher_init(this, c1_B, BBase);
            cipher_init(this, c2_B, BBase);

            for (int i = 0; i < 3; i++){
                poly_init(this  , &cstar_Q[i]);
                poly_init(this, &cstar_B[i], BBase);
                poly_init(this  , &cmult[i]);
            }

            for(unsigned int i = 0; i < CUDAEngine::RNSPrimes.size(); i++)
                poly_init(this, &d[i]);


            // FV decrypt
            m_aux = new poly_t();
            poly_init(this, m_aux);

        };

        BFVContext( BFVParams p ) : BFVContext(p.q, p.t, p.nphi){

        }

        /**
         * @brief      export all keys to a json structure
         *
         * @return     A json structure containing the sk, pk, and evk
         */
        json export_keys();

        /**
         * @brief      load keys from a json structure
         *
         * @param[in]  k     A json structure containing the sk, pk, and evk
         */
        void load_keys(const json & k);

        /**
         * @brief      Clear all the keys stored
         */
        void clear_keys();

        /**
         * @brief      Return alternative Contexts
         * 
         * This class carries alternative contexts that can be selected by an ID. 
         * If no Context exists with a particular ID, it shall be created.
         *
         * @param[in]  id    The identifier
         *
         * @return     The alternate context.
         */
        Context* get_alt_ctx(unsigned int id){
            while(id >= alt_ctxs.size())
                alt_ctxs.push_back(new Context());
            return alt_ctxs[id];
        }

       /**
         * @brief      Synchronizes all cudaStreams
         * 
         * Calls cudaStreamSynchronize() for all related streams
         *
         *
         * @return     .
         */ 
        void sync(){
            cudaStreamSynchronize(get_stream());
            cudaCheckError();
            sync_related();
        }

       /**
         * @brief      Synchronizes only the alternative related cudaStreams
         * 
         * Calls cudaStreamSynchronize() for all related streams
         *
         *
         * @return     .
         */ 
        void sync_related(){
            for(auto c : alt_ctxs){
                cudaStreamSynchronize(c->get_stream());
                cudaCheckError();
            }
        }

        /**
        * @brief      Return a BFVParams struct containing q, t, and nphi
        *
        * @return     The parameters.
        */
        BFVParams get_params(){
            return (BFVParams){params.q, params.t, params.nphi};
        }

        ~BFVContext(){
            poly_free(this, &delta);

            for (int i = 0; i < 3; i++){
                poly_free(this, &cstar_Q[i]);
                poly_free(this, &cstar_B[i]);
                poly_free(this, &cmult[i]);
            }

            for(unsigned int i = 0; i < CUDAEngine::RNSPrimes.size(); i++)
                poly_free(this, &d[i]);
            cipher_free(this, c1_B);
            cipher_free(this, c2_B);
            delete c1_B;
            delete c2_B;
            delete [] cstar_Q;
            delete [] cstar_B;
            delete [] cmult;
            delete u;
            delete e1;
            delete e2;

            poly_free(this, m_aux);
            delete m_aux;

            alt_ctxs.clear();
        }

};

__host__ SecretKey* bfv_new_sk(BFVContext *ctx);

#endif