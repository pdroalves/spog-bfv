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
#ifndef FVContext_H
#define FVContext_H

#define ntl_safe_scaling(A, B) to_ZZ(NTL::round(to_RR(A) / to_RR(B)))

#include <cuPoly/arithmetic/context.h>
#include <cuPoly/cuda/sampler.h>
#include <SPOG/arithmetic/ciphertext.h>

#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
using namespace rapidjson;
typedef Document json;

/////////////////////
// Params
// 
// This struct will store all non-key settings.
// 
// R = Z[x] / \phi_nphi(x)
// 
// q and t are the ciphertext and plaintext coefficient bases, respectively
// 
typedef struct{
    ZZ q;
    uint64_t t;
    int nphi;
} Params;

/////////////////////
// Keys            //
/////////////////////

// Public Key type
typedef struct{
    poly_t b;
    poly_t a;
} PublicKey;

// Secret Key type
typedef struct{
    poly_t s;
} SecretKey;

// evk type
typedef struct{
    poly_t *b = NULL;
    poly_t *a = NULL;
} EvaluationKey;

// A composite of all types of keys
typedef struct{
    PublicKey pk;
    SecretKey sk;
    EvaluationKey evk;
} Keys;

__host__ void keys_init(Context *ctx, Keys *k);
__host__ void keys_free(Context *ctx, Keys *k);

/**
 * @brief      This is a specialization of Context focused on handling FV's related data 
 */
class FVContext : public Context{

    private:
        Params params;

        std::vector<Context*> alt_b_ctxs; // Alternative contexts at base B
        std::vector<Context*> alt_ctxs; // Alternative contexts at base Q
        
    public: 
        // Using delta as a polynomial implies in a much faster and simpler
        // multiplication during encryption.
        poly_t delta; // Encrypt (Pre-comp)

        poly_t *cstar_Q; // Mul (Pre-comp): Store the product of c1 by c2 in base q
        poly_t *cstar_B; // Mul (Pre-comp):Store the product of c1 by c2 in base b
        poly_t *cmult; //
    
        poly_t *u; // Encryption
        poly_t *e1, *e2; // Encryption


        cipher_t *c1_B; // Mul (Pre-comp):Base B
        cipher_t *c2_B; // Mul (Pre-comp):Base B


        // FV decrypt
        poly_t *m_aux;

        PublicKey pk;
        SecretKey sk;
        EvaluationKey evk;

        poly_t *d = new poly_t[COPRIMES_BUCKET_SIZE]; // Relin


        FVContext( ZZ q,
            uint64_t t,
            int nphi){
            /////////////////////////////////////////////
            // Each FV object works on its own context //
            /////////////////////////////////////////////
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

        FVContext( Params p ) : FVContext(p.q, p.t, p.nphi){

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
         * @brief      Clear all the keys stored in the FV object
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
         * @brief      Synchronizes all related streams
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
         * @brief      Synchronizes all related streams
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
        * @brief      Return a Params struct containing q, t, and nphi
        *
        * @return     The parameters.
        */
        Params get_params(){
            return (Params){params.q, params.t, params.nphi};
        }

        ~FVContext(){
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

__host__ SecretKey* fv_new_sk(FVContext *ctx);

#endif