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

#include "spog.hh"

poly_t* SPOG::to_poly_t(SPOGPoly *m){
	poly_t *poly_m = new poly_t();
	for(int i = 0; i < m->data.size(); i++)
		poly_set_coeff(poly_m, i, to_ZZ(m->data[i].c_str()));
	return poly_m;
}


cipher_t* SPOG::to_cipher_t(SPOGCipher *c){
	cipher_t *cipher_c = new cipher_t();
	cipher_c->c[0] = to_poly_t(c->c0);
	cipher_c->c[1] = to_poly_t(c->c1);
	return cipher_c;
}

SPOGPoly* to_SPOGPoly(poly_t *p){
	SPOGPoly *sp = new SPOGPoly();

	for (int i = 0; i <= poly_get_deg(p); i++){
		std::ostringstream oss;
   		oss << poly_get_coeff(p, i);
   		sp->data.push_back(oss.str());
	}
	return sp;
}

SPOGCipher* to_SPOGCipher(cipher_t *ct){
	SPOGCipher *sc = new SPOGCipher();
	sc->c0 = to_SPOGPoly(ct->c[0]);
	sc->c1 = to_SPOGPoly(ct->c[1]);
	return sc;
}

SPOGCipher* SPOG::encrypt(SPOGPoly *m){
	return to_SPOGCipher(cipher->encrypt(*to_poly_t(m)));
}
SPOGPoly* SPOG::decrypt(SPOGCipher *ct){
	return to_SPOGPoly(cipher->decrypt(*to_cipher_t(ct)));
}
SPOGCipher* SPOG::add(SPOGCipher *a, SPOGCipher *b){
	return to_SPOGCipher(cipher->add(*to_cipher_t(a), *to_cipher_t(b)));
}
SPOGCipher* SPOG::mul(SPOGCipher *a, SPOGCipher *b){
	return to_SPOGCipher(cipher->mul(*to_cipher_t(a), *to_cipher_t(b)));
}