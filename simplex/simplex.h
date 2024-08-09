/*
 * SimplexNoise by Sebastien Rombauts (https://github.com/SRombauts/SimplexNoise)
 * Translated to C & optimized by SollyBunny (https://github.com/sollybunny/cnoise)
 * Distributed under MIT License (http://opensource.org/licenses/MIT)
 */

/*
 * Copyright (c) 2014-2018 Sebastien Rombauts (sebastien.rombauts@gmail.com)
 *
 * This C++ implementation is based on the speed-improved Java version 2012-03-09
 * by Stefan Gustavson (original Java source code in the public domain).
 * http://webstaff.itn.liu.se/~stegu/simplexnoise/SimplexNoise.java:
 * - Based on example code by Stefan Gustavson (stegu@itn.liu.se).
 * - Optimisations by Peter Eastman (peastman@drizzle.stanford.edu).
 * - Better rank ordering method by Stefan Gustavson in 2012.
 *
 * This implementation is "Simplex Noise" as presented by
 * Ken Perlin at a relatively obscure and not often cited course
 * session "Real-Time Shading" at Siggraph 2001 (before real
 * time shading actually took on), under the title "hardware noise".
 * The 3D function is numerically equivalent to his Java reference
 * code available in the PDF course notes, although I re-implemented
 * it from scratch to get more readable code. The 1D, 2D and 4D cases
 * were implemented from scratch by me from Ken Perlin's text.
 *
 * Distributed under the MIT License (MIT) (See accompanying file LICENSE.txt
 * or copy at http://opensource.org/licenses/MIT)
 */

#ifndef simplexfloat_t
	#define simplexfloat_t float
#endif
#ifndef simplexfloor
	#define simplexfloor(x) floorf(x)
#endif

simplexfloat_t simplex1D(simplexfloat_t x);
simplexfloat_t simplex2D(simplexfloat_t x, simplexfloat_t y);
simplexfloat_t simplex3D(simplexfloat_t x, simplexfloat_t y, simplexfloat_t z);
