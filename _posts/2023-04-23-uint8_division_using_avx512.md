---
layout: blog_post
title: Dividing 8-bit Uints with AVX-512VBMI
---

Torbjorn Granlund and Peter L. Montgomery are the authors behind a paper that is
well-known by those interested in numerical techniques and/or optimization. This
paper, *Division by Invariant Integers using Multiplication*, provides a
technique and proof pertaining thereto for quickly computing quotients in the
event that you know what the divisor is beforehand. The basic idea behind this
technique is nothing more than elementary math: `n / d` is the same as `n * (1 /
d)`. If the reciprocal of `d` is computed beforehand, then the division can
transformed into multiplication. Crucially, multiplication instructions are
generally cheaper than corresponding division instructions. Hence, this ends up
being a faster way to compute a quotient, at least at the point of computing the
quotient.

If you're working with floating-point numbers then computing `n * (1 / d)` is
simple, at least if you don't care about the minor error this introduces.
However, it's more complicated if you're working with integer types. The
reciprocal of an integer falls into the range of (0, 1] for positive integers,
and [-1, 1] for non-zero integers. The problem is that these values are not
themselves integers in the general case (with the obvious exceptions for -1 and
1). Hence, to put this idea into practice, alternative representations for the
reciprocal of `d` have to be considered.

This is just what Granlund and Montgomery's paper offers, alongside the formulas
for how to compute and use this alternative representation naturally. It's more
accurate to say that they provide two such approaches, for both unsigned and
signed integers. This approach can be roughly described as emulating a
floating-point number. Those interested in the fine details, are encouraged to
read through the paper. However, the only part of that paper that's really
important here are the following formulas which pertain to the technique for
unsigned integers.

Once the value of `d` is known, three variables, `m`, `sh1`, and `sh2` should be
computed as:
```c++
    l = N - lzcnt(d - 1)
    
    m = (double_width_uint((1 << l) - d) << N) / (d + 1)
    sh1 = min(l, 1)
    sh2 = l - sh1

```

Here, N refers to how many bits wide the type of `n` and `d` is.
`double_width_uint` is straightforwardly an unsigned integer type which is twice
as wide as the type of `n` or `d`.

Note that l is just an intermediate variable that does not need to be kept
around for the following step.

When some number `n` should be divided by `d`, then the following formula should
be evaluated:

```c++
    t1 = mulhi(m, n)
    uint q = t1 + ((n - t1) >> sh1) >> sh2
```

Here, mulhi refers to the high half of performing a widening multiplication.
That is equivalent to `(double_width_uint(m) * double_width_uint(n)) >> N`.

`q` here denotes the quotient `n / d`.

Note that the computation of `m` involves a division of integers twice the width
of either `n` or `d`. This means that this approach actually requires more work
than simply computing `n / d`. Therefore, this technique is really only
beneficial if you're performing multiple divisions by `d` such that decrease in
the runtime of these divisions outweights the upfront cost. Or rather, that's
what you'll commonly hear when this technique is brought up. The other case
where it's useful is when you can quickly compute the value of `m` by
alternative means.

## The Part Where AVX-512 Comes In
Even with the latest extensions to the x86 ISA, there are no vectorized integer
division instructions, meaning that vectorized division must be emulated in
software. However, software division is a fairly expensive algorithm, not to
mention somewhat inconvenient to implement in a vectorized fashion, particularly
for 8-bit ints for which x86 generally has lackluster vectorization support.

Fortunately, AVX-512VBMI introduced an instruction called `vpermi2b`. This
instruction fundamentally allows you to perform a vectorized table lookup. It
takes in two vector registers each containing 64 bytes, which are collectively
used as an 128-entry lookup table. The instruction then also take another vector
register which contains 64 indices into the aforementioned lookup table. Now,
128 is only one power of two away from 256, the number of unique values that a
byte may take on, and hence the number of unique values of `m` (if you ignore
the degenerate case of `d == 0` of course). Although it's not ideal, it's
feasible to emulate a 256-entry table lookup by using two `vpermi2b`s, a
`vpmovb2m`, and a `vpblendmb`.

The idea is that the `vpermi2b` instruction only looks at the low 7 bits of each
index in the index vector, hence we need some way to factor in that last 8th
bit. The `vpmovb2m` produces a mask specifically based on the most-significant
bit of each byte in a vector register, allowing us to easily extract that high
bit. This mask can then be used as one of he parameters to the `vpblendmb`
instruction. The blend's other inputs would simply be the results of invoking
the `vpermi2b` instruction on the high and low halves of a 256-byte lookup table
that contains all the possible values of `m`. Assuming that the lookup table is
already loaded, then that means it takes only four instructions to compute m.

Now, there another detail that's a little difficult to compute, `l`. AVX-512
does not have leading-zero count instructions for 8-bit integers which raises
the question of how to quickly compute `l`. But before you go exploring the
obvious approach, I'd like to point out that `l` may be equivalently computed as
`bit_width(x - 1)` where `bit_width` comes from C++20's
[`<bit>`](https://en.cppreference.com/w/cpp/numeric/bit_width) header. I chose
to go with this alternative framing of the problem instead because it's slightly
cheaper to compute compared to literally computing it via the formula from the
original paper.

Subtracting one from a number is simple but computing `bit_width` has some
complexity to it. Thankfully, we've just discussed three instructions that are
very useful for this kind of thing. We can use a very similar approach as was
used to compute `m`. We can note that if the high bit of `x` is set, then
`bit_width(x)` should evaluate to `8` and that if it's not set, then it should
evaluate to the bit width of the low 7 bits. Thankfully, we can check if the
high bit of each byte is set with `vpmovb2m`, compute the bit width of those 7
bits with `vpermi2b`, and combine those results with `vpblendmb`.

With a convenient way to compute `m` and `l`, it's possible to create a
vectorized implementation that might just end up being practical, and maybe even
a little faster than the alternatives.

## The Obvious Alternatives
Speaking of alternatives, it's worth discussing which specifically those are.
There are at least three to which this approach could reasonably be compared.
The first would be performing division in a scalar fashion that uses the
hardware `div` instruction. The second would be a vectorized long division
approach. And the last would be a vectorized long division approach that also
features early termination.

The long division algorithms would fundamentally take the same approach to
division as children are taught in elementary, just in base 2. This means an
iterative algorithm based on repeated subtraction that produces one bit of the
quotient with each iteration. 

Note that it's possible to reach a point where no more iterations are necessary
once the dividend becomes smaller than the divider. Should this happen, all
subsequent bits of the quotient would evaluate to 0. Also note that if the
quotient has some number of leading zeros, then the same amount of iterations
can be skipped from the start of the loop. 

However, given that we're dealing with vectors with 64 lanes, the probability
that all of them will meet either of theses criteria simultaneously for a random
set of inputs is borderline non-existent. Not to mention that even if that's
ignored and we instead assume that there is a good chance of that actually
happening, then we'd be introducing a very unpredictable set of branches into
the algorithm. Of course, the branches are in reality quite predictable in that
we know they'll almost never get taken since if even one lane does not meet the
criteria, then the loop iterations cannot be skipped. Regardless, the mere
presence of the branches and the instructions required to compute their
conditions will make the algorithm slower by some amount.

That said, theoretically, if you happen to find yourself in the frankly
fantasy-like scenario that you happen to need to compute the quotients of a long
list of unsigned 8-bit integers, whose quotients will all have around the same
number of leading and/or trailing zeros, then this might just end up being
faster.

## What About AVX-512FP16?
Another alternative that's worth bringing up would be to widen the inputs to
16-bit integers, convert to 16-bit floats, leverage the 16-bit floating-point
division, convert back to 16-bit ints, and narrow back to 8-bit floats. This
approach would involve far fewer instructions than those previously mentioned,
and would potentially better leverage the hardware . There is therefore some
degree of intuition that suggests that it may be a much faster approach.

Currently, AVX-512FP16 is only available on Golden Cove cores, which can only be
found on certain early Alderlake CPUs and Sapphire Rapids CPUs. Unfortunately, I
do not have access to any such machines to experiment on.

However, there is at least one detail that casts doubt on possibility of this
approach panning out in practice, at least as things currently are. Checking
uops.info for Alderlake's performance of the
[`vdivph`](https://www.uops.info/html-instr/VDIVPH_ZMM_ZMM_ZMM.html#ADL-P)
instruction and comparing its results to that of the
[vdivps](https://www.uops.info/html-instr/VDIVPS_ZMM_ZMM_ZMM.html#ADL-P)
instruction, it appears that 16-bit floating-point division actually has
performance characteristics that are roughly 2x worse than those for 32-bit
division. Although the site does not currently have info for Sapphire Rapids
CPUs, since they're both using Golden Cove cores, presumably it would be much
the same story. 

This casts some doubt on the efficacy of this approach. Part of the intuition
for why this approach might be faster that wasn't explicitly stated before was
the idea that with a smaller float and hence a smaller significand, the 16-bit
division instructions would be cheaper than their 32-bit counterparts, but with
latencies of up to 41 cycles, the other approaches probably still have a
fighting chance.

If we're lucky, this will change in future microarchitectures but as it stands,
emulating 8-bit integer division using 16-bit floating-point division doesn't
seem like it'll necessarily be a silver bullet.

## Some Simple Benchmarks & Code
To compare the approaches that I could run, I set up a simple and frankly
unrealistic set of benchmarks where a list of dividends and divisors that fit
into the L1D cache of my Icelake 1065G7 are processed repeatedly.

The code is available on [Compiler Explorer](https://godbolt.org/z/dEYhr8Mh3). I
do however admit that these implementations are not necessarily ideal as they
stand, and there are aspects of these implementations which are worth refining
further. For example, `vgf2p8affineqb` could be used to emulate 8-bit shifts,
however I opted to not use them here because in previously benchmarks I've ran,
they proved inferior to using 16-bit shifts and masking. Theoretically, the
could perform better in the future, or on different microarchitectures e.g. on
Zen 4 their latency is only 3 cycles and have a CPI of 0.5 instead of 5 and 1.0
respectively for my CPU. Miscellaneous other such details could also be tweaked,
but I wished to focus more on the idea of accelerate the Granlund-Montgomery
technique on machines with AVX-512VBMI.

The code was compiled using GCC 12.2.0 with the `-O3 -march=icelake-client`
flags. The code was then run 50 times and the following statistics were computed
from the results.

|                | Scalar Division | Lookup Table | Long Division | Early Return Long Division |
|----------------|-----------------|--------------|---------------|----------------------------|
| Mean           | 421,620us       | 22,986us     | 37,340us      | 51,115us                   |
| Std. Dev.      | 1574us          | 121us        | 184us         | 265us                      |
| Improvement    | 1.00            | 18.34        | 11.29         | 8.25                       |

From these numbers, we can see that at the very least there is a potential for
the lookup table based approach to perform better than the alternatives against
it was pitted. However, it does fundamentally rely on the lookup tables being in
cache. The long division algorithm by comparison do not rely on reading from
memory at all and its performance is still a large improvement over scalar code. 

We can see that the long division algorithm did not benefit from early
termination which is to be expected given that the numerators and denominators
were generated at random with a uniform distribution. Notably, the increase in
its runtime over the regular long division algorithm is almost perfectly
proportional to the increase in the number of instructions: `(140 / 102) *
37340us  = 51249us ≈ 51115us` despite the crudeness of this as a performance
estimation technique. 

The early termination approach does have some potential. In an alternate version
of the test where all the numerator (chosen from [0, 127]) were smaller than the
denominators (chosen from [128, 255]), the following results were obtained:

|                | Scalar Division | Lookup Table | Long Division | Early Return Long Division |
|----------------|-----------------|--------------|---------------|----------------------------|
| Mean           | 435,028us       | 23,775us     | 38,761us      | 27us                       |
| Std. Dev.      | 1734us          | 191us        | 263us         | 2us                        |
| Improvement    | 1.00            | 18.30        | 11.22         | 16112.15                   |

Here, the long division with early return runs through the data at a
substantially greater rate, which shouldn't be surprising given the fact that
there is no work to be done anyways. But this just isn't representative of what
you'd expect to encounter in practice so it's perhaps best left ignored.

Overall, the lookup table accelerated version of the Granlund-Montgomery
approach appears to deliver the best results, at least under these ideal, and
not necessarily realistic circumstances. That said, in the grand scheme of
things, the truth is that having to churn through a large amount of unsigned
8-bit integer divisions is not something most people ever have to do. I imagine
that the number of people that would be interested in this technique is far
greater than those who would actually benefit from it.
