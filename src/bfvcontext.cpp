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
#include <SPOG-BFV/bfvcontext.h>

// Allocates memory for a Keys object
__host__ void keys_init(Context *ctx, Keys *k){

	const int n = CUDAEngine::RNSPrimes.size();
	k->evk.a = new poly_t[n];
	k->evk.b = new poly_t[n];

	for(int i = 0; i < n; i++){
		poly_init(ctx, k->evk.a);
		poly_init(ctx, k->evk.b);
	}
}
// Releases memory of a Keys object
__host__ void keys_free(Context *ctx, Keys *k){
	poly_free(ctx, &k->sk.s);
	poly_free(ctx, &k->pk.a);
	poly_free(ctx, &k->pk.b);
	
	const int n = CUDAEngine::RNSPrimes.size();
	for(int i = 0; i < n; i++){
		poly_free(ctx, &k->evk.a[i]);
		poly_free(ctx, &k->evk.b[i]);
	}
	delete [] k->evk.a;
	delete [] k->evk.b;
}

__host__ void keys_sk_free(Context *ctx, SecretKey *sk){
	poly_free(ctx, &sk->s);
}

__host__ SecretKey* bfv_new_sk(BFVContext *ctx){
    SecretKey *sk = new SecretKey;
    poly_init(ctx, &sk->s);

    ////////////////
    // Secret key //
    ////////////////
	// Low-norm secret key
	Sampler::sample(ctx, &sk->s, NARROW);
	poly_copy_to_device(ctx, &sk->s);
    return sk;
};


__host__ void BFVContext::load_keys(const json & k){
	poly_copy(this, &keys->sk.s, poly_import(this, k["sk"]["s"].GetString()));
	poly_copy(this, &keys->pk.a, poly_import(this, k["pk"]["a"].GetString()));
	poly_copy(this, &keys->pk.b, poly_import(this, k["pk"]["b"].GetString()));

	for(unsigned int i = 0; i < CUDAEngine::RNSPrimes.size(); i++){
		poly_copy(this, &keys->evk.a[i], poly_import(this, k["evk"]["a"][i].GetString()));
		poly_copy(this, &keys->evk.b[i], poly_import(this, k["evk"]["b"][i].GetString()));
	}
}

__host__ json BFVContext::export_keys(){

	std::string sks = poly_export(this, &keys->sk.s);
	std::string pka = poly_export(this, &keys->pk.a);
	std::string pkb = poly_export(this, &keys->pk.b);

	json jkeys;
	jkeys.SetObject();
	jkeys.AddMember("sk", Value{}.SetObject(), jkeys.GetAllocator());
	jkeys.AddMember("pk", Value{}.SetObject(), jkeys.GetAllocator());
	jkeys["sk"].AddMember("s", Value{}.SetString(sks.c_str(), sks.length(), jkeys.GetAllocator()), jkeys.GetAllocator());
	jkeys["pk"].AddMember("a", Value{}.SetString(pka.c_str(), pka.length(), jkeys.GetAllocator()), jkeys.GetAllocator());
	jkeys["pk"].AddMember("b", Value{}.SetString(pkb.c_str(), pkb.length(), jkeys.GetAllocator()), jkeys.GetAllocator());

	// evk
	Value evka(kArrayType);
	Value evkb(kArrayType);
	for(unsigned int i = 0; i < CUDAEngine::RNSPrimes.size(); i++){
		std::string a = poly_export(this, &keys->evk.a[i]);
		std::string b = poly_export(this, &keys->evk.b[i]);

		evka.PushBack(Value{}.SetString(a.c_str(), a.length(), jkeys.GetAllocator()), jkeys.GetAllocator());
		evkb.PushBack(Value{}.SetString(b.c_str(), b.length(), jkeys.GetAllocator()), jkeys.GetAllocator());
	}
	jkeys.AddMember("evk", Value{}.SetObject(), jkeys.GetAllocator());
	jkeys["evk"].AddMember("a", evka, jkeys.GetAllocator());
	jkeys["evk"].AddMember("b", evkb, jkeys.GetAllocator());
	return jkeys;
}

__host__ void BFVContext::clear_keys(){
	poly_clear(this, &keys->sk.s);
	poly_clear(this, &keys->pk.a);
	poly_clear(this, &keys->pk.b);

	const int k = CUDAEngine::RNSPrimes.size();
	for(int i = 0; i < k; i++){
		poly_clear(this, &keys->evk.a[i]);
		poly_clear(this, &keys->evk.b[i]);
	}
}