/*
 * db-perlin by daniilsjb (https://github.com/daniilsjb/perlin-noise)
 * Translated to C and optimized by SollyBunny (https://github.com/sollybunny/cnoise)
 * Distributed under MIT License (http://opensource.org/licenses/MIT)
 */

/*
 * The following is an implementation of Ken Perlin's Improved Noise in 1D, 2D, and 3D.
 * This code has no external dependencies and as such may easily be used as a library
 * in other projects.
 *
 * I wrote this with the primary goal of having a bit of fun and learning more about the
 * famous algorithm used everywhere in procedural generation. Ultimately, my goal was to
 * use this implementation in several other projects (it's always good to have a noise
 * generator lying around). I hope it could be useful to other people, too!
 *
 */

/*
 * The implementation was based on this article:
 * - https://flafla2.github.io/2014/08/09/perlinnoise.html
 *
 * A reference implementation in Java by Ken Perlin, the author of the algorithm:
 * - https://mrl.cs.nyu.edu/~perlin/noise/
 *
 * Here are some alternative implementations that were used as inspirations:
 * - https://github.com/nothings/stb/blob/master/stb_perlin.h
 * - https://github.com/stegu/perlin-noise/blob/master/src/noise1234.c
 */

/*
 * MIT License
 * 
 * Copyright (c) 2020-2024 Daniils Buts
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef perlinfloat_t
	#define perlinfloat_t float
#endif
#ifndef perlinfloor
	#define perlinfloor(x) floorf(x)
#endif

perlinfloat_t perlin1D(perlinfloat_t x);
perlinfloat_t perlin2D(perlinfloat_t x, perlinfloat_t y);
perlinfloat_t perlin3D(perlinfloat_t x, perlinfloat_t y, perlinfloat_t z);

