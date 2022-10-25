#include <SPOG/arithmetic/ciphertext.h>

__host__ void cipher_init(Context *ctx, cipher_t *c, int base){
	poly_init(ctx, &c->c[0], base);
	poly_init(ctx, &c->c[1], base);
}

__host__ void cipher_clear(Context *ctx, cipher_t *c){
	poly_clear(ctx, &c->c[0]);
	poly_clear(ctx, &c->c[1]);
}

__host__ void cipher_free(Context *ctx, cipher_t *c){
	poly_free(ctx, &c->c[0]);
	poly_free(ctx, &c->c[1]);
}

__host__ void cipher_copy(Context *ctx, cipher_t *b, cipher_t *a){
	poly_copy(ctx, &b->c[0], &a->c[0]);
	poly_copy(ctx, &b->c[1], &a->c[1]);
}

__host__ std::string cipher_to_string(Context *ctx, cipher_t *c){
	return std::string("c0: ") +
	poly_to_string(ctx, &c->c[0]) +
	"\n" +
	"c1: " +
	poly_to_string(ctx, &c->c[1]);
}

std::vector<std::string> cipher_export(Context *ctx, cipher_t *ct){
	std::vector<std::string> v;
	v.push_back(poly_export(ctx, &ct->c[0]));
	v.push_back(poly_export(ctx, &ct->c[1]));
	v.push_back(to_string(ct->level));

	return v;
}

cipher_t* cipher_import(Context *ctx, std::vector<std::string> v){
	cipher_t *ct = new cipher_t;
	cipher_init(ctx, ct);
	ct->c[0] = *poly_import(ctx, v[0]);
	ct->c[1] = *poly_import(ctx, v[1]);
	ct->level = std::stoi(v[2]);

	return ct;
}