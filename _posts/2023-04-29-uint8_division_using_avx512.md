---
layout: blog_post
title: Dividing 8-bit Uints with AVX-512VBMI
description: Exploring how AVX-512VBMI can be used to perform Granlund-Montgomery division on 8-bit uints, and how a simpler more naive algorithm beats the hardware div instruction by up to ~30x .
---

Torbjorn Granlund and Peter L. Montgomery are the authors behind a paper that is
well-known by those interested in numerical techniques and/or optimization. This
paper, *Division by Invariant Integers using Multiplication*, provides a
technique, and proof for the correctness thereof, for quickly computing
quotients in the event that you know what the divisor is beforehand. The basic
idea behind this technique is nothing more than elementary math: `n / d` is the
same as `n * (1 / d)`. If the reciprocal of `d` is computed beforehand, then the
problem of division can transformed into one of multiplication. Crucially,
multiplication instructions are generally cheaper than corresponding division
instructions. Therefore, this technique ends up being a potentially faster way
to perform division, at least at the point of computing the quotient.

If you're working with floating-point numbers then computing `n * (1 / d)` is
simple, at least if you don't care about the minor error this introduces.
However, it's more complicated if you're working with integer types. The
reciprocal of an integer is in general not itself an integer, meaning it cannot
be straightforwardly represented using integral types. Hence, to put this idea
into practice, alternative representations for the reciprocal of `d` have to be
considered.

This is just what Granlund and Montgomery's paper offers. It's more accurate to
say that they provide two such approaches, one for unsigned integers and one for
signed integers. This approach can be roughly described as emulating a
floating-point number. Those interested in the fine details are encouraged to
read through the paper. However, the only part of it that's really important
here are the following formulas which pertain to the technique for unsigned
integers.

Once the value of `d` is known, three variables, `m`, `sh1`, and `sh2` should be
computed as follows:
```c++
    l = N - lzcnt(d - 1)
    
    m = (double_width_uint((1 << l) - d) << N) / (d + 1)
    sh1 = min(l, 1)
    sh2 = l - sh1

```

Here `N` refers to how many bits wide the type of `n` and `d` is.
`double_width_uint` is straightforwardly an unsigned integer type which is twice
as wide as the type of `n` or `d`.

Note that `l` is just an intermediate variable that does not need to be kept
around for the following step.

When some number `n` should be divided by `d`, then the following formula should
be evaluated:

```c++
    t1 = mulhi(m, n)
    uint q = t1 + ((n - t1) >> sh1) >> sh2
```

Here, `mulhi` refers to the high half of performing a widening multiplication.
That is equivalent to `(double_width_uint(m) * double_width_uint(n)) >> N`.

`q` here denotes the quotient of interest, `n / d`.

Note that the computation of `m` involves dividing two integers which are twice
the width of either `n` or `d`. This means that this approach actually requires
more work overall compared to simply computing `n / d` directly. Therefore, this
technique is really only beneficial if you're performing multiple divisions by
`d` such that decrease in the runtime of these divisions outweighs the upfront
cost. Or rather, that's what you'll commonly hear about this technique. The
other case where it's useful is when you can quickly compute the value of `m` by
alternative means.

## The Part Where AVX-512 Comes In
Even with the latest extensions to the x86 ISA, there are no vectorized integer
division instructions, meaning that vectorized division must be emulated in
software. However, software division is generally fairly expensive and somewhat
inconvenient to implement.  This is even moreso the case when it must be
vectorized for 8-bit ints since x86 lacks vectorized instructions for shifts and
multiplications.

Fortunately, AVX-512VBMI introduced an instruction called `vpermi2b`. This
instruction fundamentally allows you to perform a vectorized table lookup. It
takes in two vector registers each containing 64 bytes, which are collectively
used as an 128-entry lookup table. The instruction then also take another vector
register which contains 64 indices into the aforementioned lookup table. Note
that 128 is only one power of two away from 256, the number of unique values
that a byte may take on, and hence the number of unique values of `m` (if you
ignore the degenerate case of `d == 0` of course). Although it's not ideal, it's
feasible to emulate a 256-entry table lookup by using two `vpermi2b`s, a
`vpmovb2m`, and a `vpblendmb`.

The idea is that the `vpermi2b` instruction only looks at the low 7 bits of each
index in the index vector, hence we need some way to factor in that last 8th
bit. The `vpmovb2m` produces a mask specifically based on the most-significant
bit of each byte in a vector register, allowing us to easily extract that high
bit. This mask can then be used as one of the parameters to the `vpblendmb`
instruction. The blend's other inputs would simply be the results of invoking
the `vpermi2b` instruction on the high and low halves of a 256-byte lookup table
containing all possible values of `m`. Assuming that the lookup table is already
loaded, then that means `m` can be computed in as little as four instructions.

There is another detail that's a little difficult to compute, `l`. AVX-512 does
not have leading-zero count instructions for 8-bit integers which raises the
question of how to quickly compute `l`. But before you go exploring the
straightforward approach, I'd like to point out that `l` may be equivalently
computed as `bit_width(x - 1)` where `bit_width` comes from C++20's
[`<bit>`](https://en.cppreference.com/w/cpp/numeric/bit_width) header. This
alternative is slightly cheaper to compute compared to following the formula
from the original paper.

Subtracting one from a number is simple but computing `bit_width` has some
complexity to it. Thankfully, we've just discussed three instructions that are
very useful for this kind of thing. We can use a very similar approach as was
used to compute `m`. We can note that if the high bit of `x` is set, then
`bit_width(x)` should evaluate to `8` and that if it's not set, then it should
evaluate to the bit width of the low 7 bits. Thankfully, we can check if the
high bit of each byte is set with `vpmovb2m`, compute the bit width of those 7
bits with `vpermi2b`, and combine those results with a `vpblendmb`.

With a convenient way to compute both `m` and `l`, it's feasible to create a
vectorized implementation of 8-bit integer division which relies upon the
Granlund-Montgomery technique.

## Simplifying it Even Further
Early, it was mentioned that the approach taken by the paper can be crudely
described as emulating a float, however there is a simpler alternative,
emulating a fixed-point number.
[u/YumiYumiYumi](https://www.reddit.com/user/yumiYumiYumi/) suggested this
simplification to me as well as example code of a refined implementation. The
following section is an explanation of it.

This approach removes the lead to compute `l`, and use the altogether, but it
does fundamentally mean that more work must be done in computing `m` (which to
be clear is now just the fixed-point reciprocal of `d` instead of the
pseudo-mantissa which it previously was), and it must also be widened to be a
16-bit value instead of an 8-bit value. This widening is necessary to ensure
that the reciprocal has a sufficient amount of fractional bits to be able to
accurately compute the quotient.

Intuitively, a 16-bit value would seem to suggest that four uses of `vpermi2b`
would be necessary, two for each of the low and high bytes of `m`, following the
steps previously described. However this is not the case if you leverage a
couple pieces of additional information. If the value of `d` is greater than
128, then the high byte of `m` will always be `0000'0001`. 127 of the 255
non-degenerate values of `d` meet this criteria, leaving 128 to handle via a
lookup. A simple blend can be used to choose between the  `0000'0001` and the
value produced by the lookup. The mask which is passed to this blend instruction
may be produced by a `vpermi2b` if 1 is subtracted from the value of `d` first.
In the reference code, this subtraction is performed upfront, affecting the
layout of the data used in lookup table. The low bytes of the reciprocals can be
computed using two `vpermi2b` instructions as was discussed in the prior
section.

A further consideration is that the value of sh1 will only ever be `0` or
`false` when `d` equals 1, which is trivially handled by a blend at the end of
the division subroutine. Additionally, since the `d` was decremented upfront,
testing for whether `d` equal one can at this point be performed with just a
`vptestnmb`.

## The Alternatives Methods
It's worth discussing which specifically the alternatives are. There are at
least four to compare against. The first would be performing division in a
scalar fashion that uses the native `div` instruction. The second would be a
vectorized long division approach. A related third would a vectorized long
division approach that also features early termination. And the last would to
fall back to using the vectorized division instructions that *are* available,
the ones for 32-bit floats.

The long division algorithms would fundamentally take the same approach to
division as children are taught in elementary, just in base 2. This means an
iterative algorithm based on repeated subtraction that produces one bit of the
quotient with each iteration. 

Note that it's possible to reach a point where no more iterations are necessary
once the dividend becomes smaller than the divisor. Should this happen, all
subsequent bits of the quotient would evaluate to 0. Also note that if the
quotient has some number of leading zeros, then the same number of iterations
could be skipped upfront.

However, given that we're dealing with vectors with 64 lanes, the probability
that all of them will meet either of theses criteria simultaneously for a random
set of inputs is very low, practically non-existent. Not to mention that even if
that's ignored and we instead assume that there is a good chance of that
actually happening, then we'd still be introducing a very unpredictable set of
branches into the algorithm. Of course, the branches are in reality quite
predictable in that we know they'll almost never get taken since if even one
lane does not meet the criteria, then the loop iterations cannot be skipped.
Regardless, the mere presence of the branches and the instructions required to
compute their conditions will make the algorithm slower by some amount.

That said, theoretically, if you happen to find yourself in the frankly
fantasy-like scenario that you need to compute the quotients of a long list of
unsigned 8-bit integers, whose quotients will all have around the same number of
leading and/or trailing zeros, then this might just end up being faster.

Using 32-bit floating-point division instructions, seems like it may be more
likely to perform well. After all, you'd be leveraging a hardware implementation
of division. However, there is a fair bit of work to this approach as well.
You'd need to widen the 8-bit integers to 32-bit integers, convert them to
floats, perform four different division instructions (which have high CPIs in
practice), convert the quotients back to 32-bit floats, and narrow them back
down to 8-bit ints. Even without having to handle any of actual division logic
in software, this doesn't make for a trivial implementation.

## What About AVX-512FP16?
AVX-512FP16 offers vectorized 16-bit division instructions which introduces yet
another alternative. Using 16-bit fp division would involve fewer instructions
than any of the previously mentioned techniques, while still benefiting from a
hardware implementation of division. There is therefore some degree of intuition
that suggests that it may end up being the fastest approach.

Currently, AVX-512FP16 is only available on Golden Cove cores, which can only be
found on Alderlake CPUs as P-cores and Sapphire Rapids CPUs making this a fairly
rare feature at the current time. Further contributing to its rarity is a
complex situation where AVX-512 is not usable on all Alderlakes.
Unfortunately, as much as I would have liked to test the performance of 16-bit
division on these machines, I do not have access to one, nor the means to
procure one.

As an alternative we can instead reference the [uops.info](uops.info)
instruction tables for Alderlake P-cores. Checking the performance
characteristics of the
[`vdivph`](https://www.uops.info/html-instr/VDIVPH_ZMM_ZMM_ZMM.html#ADL-P)
instruction and comparing it to those of the
[vdivps](https://www.uops.info/html-instr/VDIVPS_ZMM_ZMM_ZMM.html#ADL-P)
instruction, it appears that 16-bit fp division is actually inferior to 32-bit
fp division. Roughly speaking, `vdivph` has performance characteristics that are
~2x worse than those for `vdivps`. Although the site does not have info for
Sapphire Rapids CPUs, since they're both using Golden Cove cores, presumably it
would be much the same story. 

This casts some doubt on the efficacy of using 16-bit fp division to emulate
8-bit integer division. An unstated part of the intuition for why this may be
faster was the idea that with a smaller float and hence a smaller significand,
the 16-bit division instructions would be cheaper than their 32-bit
counterparts. However, right now it's not clear whether using two 16-bit
divisions would actually end up being faster than using the four 32-bit
divisions.

If we're lucky, the performance characteristics of 16-bit fp instructions will
improve in future microarchitectures but as it stands, emulating 8-bit integer
division using 16-bit floating-point division doesn't seem like it'll
necessarily be a silver bullet.

## Some Simple Benchmarks & Code
To compare the approaches that I could run, I set up a simple and frankly
unrealistic benchmark where a list of dividends and divisors that fit into an
L1D cache on my Ice Lake 1065G7 are processed repeatedly.

The code for this benchmark and the implementations of the various approaches to
division are available on [Compiler Explorer](https://godbolt.org/z/edfdKnKKf).

In creating these implementations I've tried to ensure that they are reasonable,
but it must be admitted that they are not necessarily as well optimized as they
might theoretically be. If you go through the code, you might for example point
out the possibility of using `vgf2p8affineqb` to perform 8-bit shifts. However I
opted to not use them here because in previous benchmarks I've run on my
machine, they proved inferior to using 16-bit shifts and masking. Theoretically,
that instruction could perform better on newer microarchitectures e.g. on Zen 4
the instruction has a latency of only 3 cycles and has a CPI of 0.5 compared to
the 5 and 1.0 respectively for my Ice Lake. Again however, I don't have access
to such hardware. More broadly speaking, in my attempts to refine such small
details, I found that the differences were small enough to the point were it was
unlikely to change the overall conclusions you'd draw from the data.

The code was compiled using GCC 12.2.0 with the `-O3 -march=icelake-client`
flags. The code was then run 50 times and the following statistics were computed
from the results.

|           | Scalar    | G-M Lookup8 | G-M Lookup16 | Long Div. | Early Ret. Div | fp-32 div. |
|-----------|-----------|-------------|--------------|-----------|----------------|------------|
| Mean      | 421,273us | 22,464us    | 14,091us     | 36,862us  | 50,505us       | 44,497us   |
| Std. Dev. | 1,407us   | 156us       | 163us        | 175us     | 273us          | 242us      |
| Speed-up  | N/A       | 18.75       | 29.90        | 11.43     | 8.34           | 9.47       |
{: .table.wide }

Notably, all approaches delivered a substantial improvement over using the
scalar div instruction, so none of them are terrible.

However, the Granlund-Montgomery approaches stand out as they performs nearly
20x and 30x better than scalar code. However, they do fundamentally rely on the
lookup tables being in cache, something which may ruin its performance in
practical applications where divisions don't have sufficient temporal locality.

In contrast, the long division algorithm does not rely on reading from memory at
all and its performance is still a large improvement over scalar code. It may
end up being faster in applications that involve one-off divisions.

32-bit floating-point division did not perform great. It's performance is
somewhat worse than long-division and is only half as good as the
Granlund-Montgomery approach. 

We can see that the long division algorithm did not benefit from returning early
which is to be expected given that the numerators and denominators were
generated at random with a uniform distribution, demonstrating that as predicted
earlier, iterations were not able to be skipped (modifying the code so that it
prints out a message when iterations are skipped confirms that it didn't happen
even once).

The early termination approach does have some potential however. In an alternate
version of the test where all the dividends (chosen from [0, 127]) were smaller
than the divisors (chosen from [128, 255]), meaning that the quotients would
always be 0, the following results were obtained:

|           | Early Ret. Div |
|-----------|----------------|
| Mean      | 29us           |
| Std. Dev. | 1.85us         |
| Speed-up  | 15432.66       |
{: .table.narrow }

Here, the early-returning long division runs through the data at a substantially
greater rate, which shouldn't be surprising given the fact that there is no
meaningful work to be done. But this just isn't representative of what you'd
expect to encounter in practice so perhaps this potential is best left ignored.

Overall, the lookup table accelerated version of the Granlund-Montgomery
approach appears to be best if you're going to be performing a large amount of
8-bit divisions in series, and the long-division may be best when performing
divisions on an inconsistent basis. That said, in the grand scheme of things,
the truth is that having to churn through any significant amount of unsigned
8-bit integer divisions is not something most people ever have to do. I imagine
that the number of people that would actually benefit from this is quite small.
