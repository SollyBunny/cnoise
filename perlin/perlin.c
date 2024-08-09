#include "perlin.h"

#include <math.h>
#include <stdint.h>

// Permutation table, the second half is a mirror of the first half.
static unsigned char p[512] = {
	151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69, 142,
	8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203,
	117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175, 74,
	165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230, 220,
	105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132,
	187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186,
	3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59,
	227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163, 70,
	221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178,
	185, 112, 104, 218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241,
	81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204, 176,
	115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195,
	78, 66, 215, 61, 156, 180,

	151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69, 142,
	8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203,
	117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175, 74,
	165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230, 220,
	105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132,
	187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186,
	3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59,
	227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163, 70,
	221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178,
	185, 112, 104, 218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241,
	81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204, 176,
	115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195,
	78, 66, 215, 61, 156, 180,
};

static inline perlinfloat_t lerp(perlinfloat_t a, perlinfloat_t b, perlinfloat_t t) {
	return a + t * (b - a);
}

static inline perlinfloat_t fade(perlinfloat_t t) {
	return t * t * t * (t * (t * (perlinfloat_t)6.0 - (perlinfloat_t)15.0) + (perlinfloat_t)10.0);
}

static inline perlinfloat_t dot_grad(int hash, perlinfloat_t xf) {
	// In 1D case, the gradient may be either 1 or -1.
	// The distance vector is the input offset (relative to the smallest bound).
	return (hash & 0x1) ? xf : -xf;
}

static const perlinfloat_t dot_grad3_gradients[8][2] = {
	{ 1,  1},
	{ 1,  0},
	{ 1, -1},
	{ 0, -1},
	{-1, -1},
	{-1,  0},
	{-1,  1},
	{ 0,  1}
};
static inline perlinfloat_t dot_grad3(int hash, perlinfloat_t xf, perlinfloat_t yf) {
	// In 2D case, the gradient may be any of 8 direction vectors pointing to the
	// edges of a unit-square. The distance vector is the input offset (relative to
	// the smallest bound).
	// switch (hash & 0x7) {
	// 	case 0x0: return  xf + yf;
	// 	case 0x1: return  xf;
	// 	case 0x2: return  xf - yf;
	// 	case 0x3: return -yf;
	// 	case 0x4: return -xf - yf;
	// 	case 0x5: return -xf;
	// 	case 0x6: return -xf + yf;
	// 	case 0x7: return  yf;
	// 	default:  return  (perlinfloat_t)0.0;
	// }
	int index = hash & 0x7;
	perlinfloat_t gx = dot_grad3_gradients[index][0];
	perlinfloat_t gy = dot_grad3_gradients[index][1];
	return gx * xf + gy * yf;
}

static const perlinfloat_t dot_grad4_gradients[16][3] = {
	{ 1.0,  1.0, 0.0}, {-1.0,  1.0, 0.0}, { 1.0, -1.0, 0.0}, {-1.0, -1.0, 0.0},
	{ 1.0,  0.0, 1.0}, {-1.0,  0.0, 1.0}, { 1.0,  0.0, -1.0}, {-1.0,  0.0, -1.0},
	{ 0.0,  1.0, 1.0}, { 0.0, -1.0, 1.0}, { 0.0,  1.0, -1.0}, { 0.0, -1.0, -1.0},
	{ 1.0,  1.0, 0.0}, {-1.0,  0.0, 1.0}, { 0.0,  1.0, -1.0}, { 0.0, -1.0, -1.0}
};
static inline perlinfloat_t dot_grad4(int hash, perlinfloat_t xf, perlinfloat_t yf, perlinfloat_t zf) {
	// In 3D case, the gradient may be any of 12 direction vectors pointing to the edges
	// of a unit-cube (rounded to 16 with duplications). The distance vector is the input
	// offset (relative to the smallest bound).
	/*switch (hash & 0xF) {
		case 0x0: return  xf + yf;
		case 0x1: return -xf + yf;
		case 0x2: return  xf - yf;
		case 0x3: return -xf - yf;
		case 0x4: return  xf + zf;
		case 0x5: return -xf + zf;
		case 0x6: return  xf - zf;
		case 0x7: return -xf - zf;
		case 0x8: return  yf + zf;
		case 0x9: return -yf + zf;
		case 0xA: return  yf - zf;
		case 0xB: return -yf - zf;
		case 0xC: return  yf + xf;
		case 0xD: return -yf + zf;
		case 0xE: return  yf - xf;
		case 0xF: return -yf - zf;
		default:  return  (perlinfloat_t)0.0;
	}*/
	const perlinfloat_t *g = dot_grad4_gradients[hash & 0xF];
	return g[0] * xf + g[1] * yf + g[2] * zf;
}

perlinfloat_t perlin1D(perlinfloat_t x) {
	// Left coordinate of the unit-line that contains the input.
	const int32_t xi0 = perlinfloor(x);

	// Input location in the unit-line.
	const perlinfloat_t xf0 = x - (perlinfloat_t)xi0;
	const perlinfloat_t xf1 = xf0 - (perlinfloat_t)1.0;

	// Wrap to range 0-255.
	const int32_t xi = xi0 & 0xFF;

	// Apply the fade function to the location.
	perlinfloat_t const u = fade(xf0);

	// Generate hash values for each point of the unit-line.
	const int32_t h0 = p[xi + 0];
	const int32_t h1 = p[xi + 1];

	// Linearly interpolate between dot products of each gradient with its distance to the input location.
	return lerp(dot_grad(h0, xf0), dot_grad(h1, xf1), u);
}

perlinfloat_t perlin2D(perlinfloat_t x, perlinfloat_t y) {
	// perlinfloat_top-left coordinates of the unit-square.
	const int32_t xi0 = (int32_t)(perlinfloor(x)) & 0xFF;
	const int32_t yi0 = (int32_t)(perlinfloor(y)) & 0xFF;

	// Input location in the unit-square.
	perlinfloat_t const xf0 = x - (perlinfloat_t)xi0;
	perlinfloat_t const yf0 = y - (perlinfloat_t)yi0;
	perlinfloat_t const xf1 = xf0 - (perlinfloat_t)1.0;
	perlinfloat_t const yf1 = yf0 - (perlinfloat_t)1.0;

	// Wrap to range 0-255.
	const int32_t xi = xi0 & 0xFF;
	const int32_t yi = yi0 & 0xFF;

	// Apply the fade function to the location.
	const perlinfloat_t u = fade(xf0);
	const perlinfloat_t v = fade(yf0);

	// Generate hash values for each point of the unit-square.
	const int32_t h00 = p[p[xi + 0] + yi + 0];
	const int32_t h01 = p[p[xi + 0] + yi + 1];
	const int32_t h10 = p[p[xi + 1] + yi + 0];
	const int32_t h11 = p[p[xi + 1] + yi + 1];

	// Linearly interpolate between dot products of each gradient with its distance to the input location.
	const perlinfloat_t x1 = lerp(dot_grad3(h00, xf0, yf0), dot_grad3(h10, xf1, yf0), u);
	const perlinfloat_t x2 = lerp(dot_grad3(h01, xf0, yf1), dot_grad3(h11, xf1, yf1), u);
	return lerp(x1, x2, v);
}

perlinfloat_t perlin3D(perlinfloat_t x, perlinfloat_t y, perlinfloat_t z) {
	// perlinfloat_top-left coordinates of the unit-cube.
	const int32_t xi0 = perlinfloor(x);
	const int32_t yi0 = perlinfloor(y);
	const int32_t zi0 = perlinfloor(z);

	// Input location in the unit-cube.
	const perlinfloat_t xf0 = x - (perlinfloat_t)xi0;
	const perlinfloat_t yf0 = y - (perlinfloat_t)yi0;
	const perlinfloat_t zf0 = z - (perlinfloat_t)zi0;
	const perlinfloat_t xf1 = xf0 - (perlinfloat_t)1.0;
	const perlinfloat_t yf1 = yf0 - (perlinfloat_t)1.0;
	const perlinfloat_t zf1 = zf0 - (perlinfloat_t)1.0;

	// Wrap to range 0-255.
	const int32_t xi = xi0 & 0xFF;
	const int32_t yi = yi0 & 0xFF;
	const int32_t zi = zi0 & 0xFF;

	// Apply the fade function to the location.
	const perlinfloat_t u = fade(xf0);
	const perlinfloat_t v = fade(yf0);
	const perlinfloat_t w = fade(zf0);

	// Generate hash values for each point of the unit-cube.
	const int32_t h000 = p[p[p[xi + 0] + yi + 0] + zi + 0];
	const int32_t h001 = p[p[p[xi + 0] + yi + 0] + zi + 1];
	const int32_t h010 = p[p[p[xi + 0] + yi + 1] + zi + 0];
	const int32_t h011 = p[p[p[xi + 0] + yi + 1] + zi + 1];
	const int32_t h100 = p[p[p[xi + 1] + yi + 0] + zi + 0];
	const int32_t h101 = p[p[p[xi + 1] + yi + 0] + zi + 1];
	const int32_t h110 = p[p[p[xi + 1] + yi + 1] + zi + 0];
	const int32_t h111 = p[p[p[xi + 1] + yi + 1] + zi + 1];

	// Linearly interpolate between dot products of each gradient with its distance to the input location.
	const perlinfloat_t x11 = lerp(dot_grad4(h000, xf0, yf0, zf0), dot_grad4(h100, xf1, yf0, zf0), u);
	const perlinfloat_t x12 = lerp(dot_grad4(h010, xf0, yf1, zf0), dot_grad4(h110, xf1, yf1, zf0), u);
	const perlinfloat_t x21 = lerp(dot_grad4(h001, xf0, yf0, zf1), dot_grad4(h101, xf1, yf0, zf1), u);
	const perlinfloat_t x22 = lerp(dot_grad4(h011, xf0, yf1, zf1), dot_grad4(h111, xf1, yf1, zf1), u);

	const perlinfloat_t y1 = lerp(x11, x12, v);
	const perlinfloat_t y2 = lerp(x21, x22, v);

	return lerp(y1, y2, w);
}

