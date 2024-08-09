// Test it compiles :P

#include <stdio.h>

#include "../perlin/perlin.h"
#include "../simplex/simplex.h"

#define JOIN(a, b) (a ## b)
#define TESTNOISE(func, x, y, z) \
	printf(#func "1D: %f\n", JOIN(func, 1D)(x)); \
	printf(#func "2D: %f\n", JOIN(func, 2D)(x, y)); \
	printf(#func "3D: %f\n", JOIN(func, 3D)(x, y, z))

int main() {
	float x = 0.5f, y = 0.1f, z = 0.05f;
	TESTNOISE(perlin, x, y, z);
	TESTNOISE(simplex, x, y, z);
}
