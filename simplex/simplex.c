#include "simplex.h"

#include <math.h>
#include <stdint.h>

static const uint8_t perm[256] = {
	151, 160, 137, 91, 90, 15,
	131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23,
	190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33,
	88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175, 74, 165, 71, 134, 139, 48, 27, 166,
	77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244,
	102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200, 196,
	135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226, 250, 124, 123,
	5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42,
	223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
	129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104, 218, 246, 97, 228,
	251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14, 239, 107,
	49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254,
	138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180
};

static inline uint8_t hash(int32_t i) {
	return perm[(uint8_t)i];
}

static inline simplexfloat_t grad1(int32_t hash, simplexfloat_t x) {
	const simplexfloat_t grad = (hash & 7) + 1;
	// NOTE: if your noise is not looking noisey enough, reenable this line
	// if ((h & 8) != 0) grad = -grad; // Set a random sign for the gradient
	return grad * x;              // Multiply the gradient with the distance
}

static inline simplexfloat_t grad2(int32_t hash, simplexfloat_t x, simplexfloat_t y) {
	const uint8_t h = hash & 0x3F;  // Convert low 3 bits of hash code
	const simplexfloat_t u = h < 4 ? x : y;  // into 8 simple gradient directions,
	const simplexfloat_t v = h < 4 ? y : x;
	return ((h & 1) ? -u : u) + ((h & 2) ? -2.0f * v : 2.0f * v); // and compute the dot product with (x,y).
}

static inline simplexfloat_t grad3(int32_t hash, simplexfloat_t x, simplexfloat_t y, simplexfloat_t z) {
	const uint8_t h = hash & 15;     // Convert low 4 bits of hash code into 12 simple
	const simplexfloat_t u = h < 8 ? x : y;
	const simplexfloat_t v = h < 4 ? y : h == 12 || h == 14 ? x : z; // Fix repeats at h = 12 to 15
	return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
}

simplexfloat_t simplex1D(simplexfloat_t x) {
	simplexfloat_t n0, n1;   // Noise contributions from the two "corners"

	// No need to skew the input space in 1D

	// Corners coordinates (nearest integer values):
	int32_t i0 = simplexfloor(x);
	int32_t i1 = i0 + 1;
	// Distances to corners (between 0 and 1):
	simplexfloat_t x0 = x - i0;
	simplexfloat_t x1 = x0 - 1.0f;

	// Calculate the contribution from the first corner
	simplexfloat_t t0 = 1.0f - x0*x0;
	t0 *= t0;
	n0 = t0 * t0 * grad1(hash(i0), x0);

	// Calculate the contribution from the second corner
	simplexfloat_t t1 = 1.0f - x1*x1;
	t1 *= t1;
	n1 = t1 * t1 * grad1(hash(i1), x1);

	// The maximum value of this noise is 8*(3/4)^4 = 2.53125
	// A factor of 0.395 scales to fit exactly within [-1,1]
	return 0.395f * (n0 + n1);
}

simplexfloat_t simplex2D(simplexfloat_t x, simplexfloat_t y) {
	simplexfloat_t n0, n1, n2;   // Noise contributions from the three corners

	// Skewing/Unskewing factors for 2D
	static const simplexfloat_t F2 = 0.366025403f;  // F2 = (sqrt(3) - 1) / 2
	static const simplexfloat_t G2 = 0.211324865f;  // G2 = (3 - sqrt(3)) / 6   = F2 / (1 + 2 * K)

	// Skew the input space to determine which simplex cell we're in
	const simplexfloat_t s = (x + y) * F2;  // Hairy factor for 2D
	const simplexfloat_t xs = x + s;
	const simplexfloat_t ys = y + s;
	const int32_t i = (int32_t)simplexfloor(xs);
	const int32_t j = (int32_t)simplexfloor(ys);

	// Unskew the cell origin back to (x,y) space
	const simplexfloat_t t = (simplexfloat_t)(i + j) * G2;
	const simplexfloat_t X0 = i - t;
	const simplexfloat_t Y0 = j - t;
	const simplexfloat_t x0 = x - X0;  // The x,y distances from the cell origin
	const simplexfloat_t y0 = y - Y0;

	// For the 2D case, the simplex shape is an equilateral triangle.
	// Determine which simplex we are in.
	int32_t i1, j1;  // Offsets for second (middle) corner of simplex in (i,j) coords
	if (x0 > y0) {   // lower triangle, XY order: (0,0)->(1,0)->(1,1)
		i1 = 1;
		j1 = 0;
	} else {   // upper triangle, YX order: (0,0)->(0,1)->(1,1)
		i1 = 0;
		j1 = 1;
	}

	const simplexfloat_t x1 = x0 - i1 + G2;            // Offsets for middle corner in (x,y) unskewed coords
	const simplexfloat_t y1 = y0 - j1 + G2;
	const simplexfloat_t x2 = x0 - 1.0f + 2.0f * G2;   // Offsets for last corner in (x,y) unskewed coords
	const simplexfloat_t y2 = y0 - 1.0f + 2.0f * G2;

	// Work out the hashed gradient indices of the three simplex corners
	const int32_t gi0 = hash(i + hash(j));
	const int32_t gi1 = hash(i + i1 + hash(j + j1));
	const int32_t gi2 = hash(i + 1 + hash(j + 1));

	// Calculate the contribution from the first corner
	simplexfloat_t t0 = 0.5f - x0*x0 - y0*y0;
	if (t0 < 0.0f) {
		n0 = 0.0f;
	} else {
		t0 *= t0;
		n0 = t0 * t0 * grad2(gi0, x0, y0);
	}

	// Calculate the contribution from the second corner
	simplexfloat_t t1 = 0.5f - x1*x1 - y1*y1;
	if (t1 < 0.0f) {
		n1 = 0.0f;
	} else {
		t1 *= t1;
		n1 = t1 * t1 * grad2(gi1, x1, y1);
	}

	// Calculate the contribution from the third corner
	simplexfloat_t t2 = 0.5f - x2*x2 - y2*y2;
	if (t2 < 0.0f) {
		n2 = 0.0f;
	} else {
		t2 *= t2;
		n2 = t2 * t2 * grad2(gi2, x2, y2);
	}

	// Add contributions from each corner to get the final noise value.
	// The result is scaled to return values in the interval [-1,1].
	return 45.23065f * (n0 + n1 + n2);
}

simplexfloat_t simplex3D(simplexfloat_t x, simplexfloat_t y, simplexfloat_t z) {
	simplexfloat_t n0, n1, n2, n3; // Noise contributions from the four corners

	// Skewing/Unskewing factors for 3D
	static const simplexfloat_t F3 = 1.0f / 3.0f;
	static const simplexfloat_t G3 = 1.0f / 6.0f;

	// Skew the input space to determine which simplex cell we're in
	simplexfloat_t s = (x + y + z) * F3; // Very nice and simple skew factor for 3D
	int32_t i = (int32_t)simplexfloor(x + s);
	int32_t j = (int32_t)simplexfloor(y + s);
	int32_t k = (int32_t)simplexfloor(z + s);
	simplexfloat_t t = (i + j + k) * G3;
	simplexfloat_t X0 = i - t; // Unskew the cell origin back to (x,y,z) space
	simplexfloat_t Y0 = j - t;
	simplexfloat_t Z0 = k - t;
	simplexfloat_t x0 = x - X0; // The x,y,z distances from the cell origin
	simplexfloat_t y0 = y - Y0;
	simplexfloat_t z0 = z - Z0;

	// For the 3D case, the simplex shape is a slightly irregular tetrahedron.
	// Determine which simplex we are in.
	int32_t i1, j1, k1; // Offsets for second corner of simplex in (i,j,k) coords
	int32_t i2, j2, k2; // Offsets for third corner of simplex in (i,j,k) coords
	if (x0 >= y0) {
		if (y0 >= z0) {
			i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 1; k2 = 0; // X Y Z order
		} else if (x0 >= z0) {
			i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 0; k2 = 1; // X Z Y order
		} else {
			i1 = 0; j1 = 0; k1 = 1; i2 = 1; j2 = 0; k2 = 1; // Z X Y order
		}
	} else { // x0<y0
		if (y0 < z0) {
			i1 = 0; j1 = 0; k1 = 1; i2 = 0; j2 = 1; k2 = 1; // Z Y X order
		} else if (x0 < z0) {
			i1 = 0; j1 = 1; k1 = 0; i2 = 0; j2 = 1; k2 = 1; // Y Z X order
		} else {
			i1 = 0; j1 = 1; k1 = 0; i2 = 1; j2 = 1; k2 = 0; // Y X Z order
		}
	}

	simplexfloat_t x1 = x0 - i1 + G3; // Offsets for second corner in (x,y,z) coords
	simplexfloat_t y1 = y0 - j1 + G3;
	simplexfloat_t z1 = z0 - k1 + G3;
	simplexfloat_t x2 = x0 - i2 + 2.0f * G3; // Offsets for third corner in (x,y,z) coords
	simplexfloat_t y2 = y0 - j2 + 2.0f * G3;
	simplexfloat_t z2 = z0 - k2 + 2.0f * G3;
	simplexfloat_t x3 = x0 - 1.0f + 3.0f * G3; // Offsets for last corner in (x,y,z) coords
	simplexfloat_t y3 = y0 - 1.0f + 3.0f * G3;
	simplexfloat_t z3 = z0 - 1.0f + 3.0f * G3;

	// Work out the hashed gradient indices of the four simplex corners
	const uint32_t gi0 = (uint32_t)hash(i + hash(j + hash(k)));
	const uint32_t gi1 = (uint32_t)hash(i + i1 + hash(j + j1 + hash(k + k1)));
	const uint32_t gi2 = (uint32_t)hash(i + i2 + hash(j + j2 + hash(k + k2)));
	const uint32_t gi3 = (uint32_t)hash(i + 1 + hash(j + 1 + hash(k + 1)));

	// Calculate the contribution from the four corners
	simplexfloat_t t0 = 0.6f - x0*x0 - y0*y0 - z0*z0;
	if (t0 < 0) {
		n0 = 0.0;
	} else {
		t0 *= t0;
		n0 = t0 * t0 * grad3(gi0, x0, y0, z0);
	}
	simplexfloat_t t1 = 0.6f - x1*x1 - y1*y1 - z1*z1;
	if (t1 < 0) {
		n1 = 0.0;
	} else {
		t1 *= t1;
		n1 = t1 * t1 * grad3(gi1, x1, y1, z1);
	}
	simplexfloat_t t2 = 0.6f - x2*x2 - y2*y2 - z2*z2;
	if (t2 < 0) {
		n2 = 0.0;
	} else {
		t2 *= t2;
		n2 = t2 * t2 * grad3(gi2, x2, y2, z2);
	}
	simplexfloat_t t3 = 0.6f - x3*x3 - y3*y3 - z3*z3;
	if (t3 < 0) {
		n3 = 0.0;
	} else {
		t3 *= t3;
		n3 = t3 * t3 * grad3(gi3, x3, y3, z3);
	}
	// Add contributions from each corner to get the final noise value.
	// The result is scaled to stay just inside [-1,1]
	return 32.0f * (n0 + n1 + n2 + n3);
}

// TODO: convert this part to C

// /**
//  * Fractal/Fractional Brownian Motion (fBm) summation of 1D Perlin Simplex noise
//  *
//  * @param[in] octaves   number of fraction of noise to sum
//  * @param[in] x         simplexfloat_t coordinate
//  *
//  * @return Noise value in the range[-1; 1], value of 0 on all integer coordinates.
//  */
// simplexfloat_t SimplexNoise::fractal(size_t octaves, simplexfloat_t x) const {
//     simplexfloat_t output    = 0.f;
//     simplexfloat_t denom     = 0.f;
//     simplexfloat_t frequency = mFrequency;
//     simplexfloat_t amplitude = mAmplitude;

//     for (size_t i = 0; i < octaves; i++) {
//         output += (amplitude * noise(x * frequency));
//         denom += amplitude;

//         frequency *= mLacunarity;
//         amplitude *= mPersistence;
//     }

//     return (output / denom);
// }

// /**
//  * Fractal/Fractional Brownian Motion (fBm) summation of 2D Perlin Simplex noise
//  *
//  * @param[in] octaves   number of fraction of noise to sum
//  * @param[in] x         x simplexfloat_t coordinate
//  * @param[in] y         y simplexfloat_t coordinate
//  *
//  * @return Noise value in the range[-1; 1], value of 0 on all integer coordinates.
//  */
// simplexfloat_t SimplexNoise::fractal(size_t octaves, simplexfloat_t x, simplexfloat_t y) const {
//     simplexfloat_t output = 0.f;
//     simplexfloat_t denom  = 0.f;
//     simplexfloat_t frequency = mFrequency;
//     simplexfloat_t amplitude = mAmplitude;

//     for (size_t i = 0; i < octaves; i++) {
//         output += (amplitude * noise(x * frequency, y * frequency));
//         denom += amplitude;

//         frequency *= mLacunarity;
//         amplitude *= mPersistence;
//     }

//     return (output / denom);
// }

// /**
//  * Fractal/Fractional Brownian Motion (fBm) summation of 3D Perlin Simplex noise
//  *
//  * @param[in] octaves   number of fraction of noise to sum
//  * @param[in] x         x simplexfloat_t coordinate
//  * @param[in] y         y simplexfloat_t coordinate
//  * @param[in] z         z simplexfloat_t coordinate
//  *
//  * @return Noise value in the range[-1; 1], value of 0 on all integer coordinates.
//  */
// simplexfloat_t SimplexNoise::fractal(size_t octaves, simplexfloat_t x, simplexfloat_t y, simplexfloat_t z) const {
//     simplexfloat_t output = 0.f;
//     simplexfloat_t denom  = 0.f;
//     simplexfloat_t frequency = mFrequency;
//     simplexfloat_t amplitude = mAmplitude;

//     for (size_t i = 0; i < octaves; i++) {
//         output += (amplitude * noise(x * frequency, y * frequency, z * frequency));
//         denom += amplitude;

//         frequency *= mLacunarity;
//         amplitude *= mPersistence;
//     }

//     return (output / denom);
// }

