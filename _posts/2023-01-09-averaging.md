---
layout: blog_post
title: Integer Averaging, Up to SIMD Midpoint
---

Any programmer will be familiar with the formula for the average of two numbers:  

```code_block
    (a + b) / 2
```

And any competent programmer will know that when it comes to software, this
isn't the way to do it. We know that the intermediate sum could overflow and
that in order to write a function that works in the general case, some
workaround must be employed.  

If you're looking for those workarounds, I'm not necessarily here to offer them.
[Raymond Chen](https://devblogs.microsoft.com/oldnewthing/20220207-00/?p=106223)
has done a better job than I could, at least for unsigned averages performed in
a scalar fashion.   

Instead, I'd like to point out that there are dedicated integer averaging
instructions on x86, ARM, and even Nvidia GPUs, and AMD GPUs. This sounds like
it has the potential to remove the need for any workaround, and surely that must
be why I disregarded them in the previous paragraph. Unfortunately, that is not
the case.  

So let's take a look at these instructions, and let's see what issues arise.

## ISA Averaging Support
Before actually doing that however, it's important to note that all of the
averaging instructions that will be mentioned here are SIMD instructions. If
you're not already familiar the concept, it helps to contrast with regular
scalar instructions. A scalar instruction operates on one set of inputs at a
time and produces one set of outputs, e.g. takes the average of of one pair of
integers and produce one average. On the other hand, SIMD instructions operate
on multiple sets of inputs and produce multiple sets of outputs (At least
usually. Sometimes they just produce a single output if they perform some kind
of reduction for example), e.g. take 16 pairs of integers and produces 16
averages in one instruction. The kind of SIMD that would be relevant here would
be that of SWAR, SIMD within a register. Here, instructions operated on vector
registers that are multiple times larger than the data being processed. On CPUs,
128-bit vector registers are quite common. These 128-bit vector registers can
generally be packed with values any of the native integer or floating-point
types. i.e. an 128-bit register could contain sixteen 8-bit values, eight 16-bit
values, four 32-bit values, or two 64-bit values. On GPUs, 32-bit can often be
packed with 8-bit or 16-bit values. 

On x86, there are averaging instructions which operate on unsigned 8-bit
integers, `(v)pavgb` and unsigned 16-bit integers, `(v)pavgw`. With the SSE2,
AVX2, and AVX-512BW extensions, these instructions are available for 128-bit,
256-bit, and 512-bit vector registers respectively. In the event that the
average of the two integers cannot be represented exactly because there is a
remainder of 1, these instructions will round the result upwards.  

On ARM, there are averaging instructions called halving additions, `UHADD`,
`SHADD`, available with NEON and SVE2. These instructions are available for
unsigned and signed integers 8, 16, or 32 bits in size. The Neon instructions
only operate on 64-bit and 128-bit vector registers, which is likely narrower
than their x86 counterparts. However, SVE2 vector registers may be larger than
that. Unlike the vector registers that you'd see on x86 or those used by Neon,
these registers are not a fixed size. Their size is a detail of the particular
hardware implementation on which the compiled code is run. They could
theoretically be as wide as 2048 bits but in practice consumer grade
implementations still only offer 128-bit registers. With regards to rounding,
there are actually two different versions of these halving add instructions. One
set which will round downwards and another set, so-called rounding halving
instructions, which will round upwards. These rounding rules apply to both the
unsigned and signed versions of these instructions.  

On Nvidia GPUs, there are averaging instructions (or at least CUDA intrinsics
which highly suggest the presence thereof, `__vavgs2`, `__vavgs4`, `__vavgu2`,
`__vavgu4`) which process unsigned and signed integers, 8 or 16 bits in size,
packed into 32-bit registers. The unsigned averages will round upwards while the
signed averages will round away from zero.  

(I should note that the following paragraph is based on AMD's documentation for
their GCN3, GCN5, and RDNA2 ISAs, but I've been unable to verify my
understanding on actual hardware)  

On AMD GPUs since at least the GCN3 microarchitecture, there is an instruction,
`V_LERP_U8`, which performs averages between unsigned 8-bit integers packed into
32-bit registers. Interestingly, this instruction allows the programmer to
choose between rounding up and rounding down on a per-lane basis. The
instruction takes a third operand whose low-bit in each 8-bit integer determines
the rounding scheme.  

Of these, it seems that ARM is the only that takes a general-purpose approach to
averaging. It not only has averaging instructions for both unsigned and signed
integers, it's the only one to support averaging 32-bit ints, which is almost
certainly the most common size of integer, and it also allows the programmer to
choose how they'd like the average to round. The instructions available on other
platforms are presumably motivated by the desire to accelerate image/texture
processing given that they operate on 8-bit or 16-bit integers, typically
unsigned, which are of course commonly used for representing the color channels
of images/texture maps. When contrasted with each other like this, their
offerings can feel lacking or incomplete when compared to ARM. And even then,
ARM doesn't have 64-bit averaging instructions so there's at least one more
obvious thing you could ask for. 

## Rounding

We might also note that there's a fair bit of variety in the way that these
instructions handle rounding. Some of the unsigned averages round up while
others round down. For the signed averages, some round downwards, some round
upwards, and some round away from zero. There's a surprising amount of
inconsistency here.  

I think it should be noted that a similar set of gaps exists when talking about
the common bit-hacks for averages that you most frequently come across online.
Most discussion of these hacks focuses specifically on unsigned integers, and
the main focus is on how to avoid overflow. There is often a tacit assumption
that you'll want to round downwards, and less attention is paid to the
possibility that you'd want to average signed integers. 

However, there are certainly valid reasons why you may want to round in
different ways. The range of rounding schemes found in the aforementioned
averaging instructions testifies to this. We can recognize that the designers of
each of those instructions would have had to make a conscious decision about how
to round yet these various designers seem to have reached various different
conclusions about the matter. We even saw that there was some support for
different rounding schemes even within the same platform and so we can say that
the designers recognized the utility of different rounding schemes. 

But that very thing is the crux of the matter. The purpose for which the
averaging is done will affect which rounding scheme is desirable. This places a
limit to the utility of not only these instructions, but also any individual
averaging bit-hack.  

One of the simplest reasons why you may want to go with a particular rounding
scheme is simply as a matter of convention, so that your program exhibits
results which are consistent with another system. .

There are certain contexts where you may want to round downwards to prevent new
"stuff" from being created. For example, if the two values you're averaging
represent money, it would not be wise to introduce an extra unit of currency
here and there. 

Rounding upwards can be useful in situations where underestimates are not
desirable. Computations involving some kind of capacity or budget can can
benefit from mild over-estimates to ensure that the computed quantity is always
sufficient.

Furthermore, we should note that there are easily more rounding schemes that you
could be interested in using that haven't been brought up yet. In particular,
for signed integers there's rounding towards zero, which is consistent with the
rounding of the expression `(x + y) / 2` when `x` and `y` are signed integers.
Beyond that there are still more possible rounding schemes such as rounding
towards the nearest even/odd integer, and there's even stochastic rounding where
the rounded value is chosen randomly. I will however not touch upon these last
few rounding schemes any further.

## std::midpoint

For both unsigned and signed integers, there is another notable rounding scheme,
one that which is consistent with C++ 20's `std::midpoint`. This function could
be described as an averaging function of sorts, but it's got a rounding scheme
that's rather interesting. If the midpoint of the two inputs to this function
cannot be exactly represented, the result rounds towards the first input. This
makes this the first rounding scheme we've come across that exhibits asymmetric
behavior with regards to its arguments. It's probably worth asking why someone
would want this because symmetric behavior seems like it would be much more
intuitively useful. Surely you don't want your results changing just because you
did something as trivial as get the order of the arguments backwards, right? It
sounds like a very subtle bug just waiting to happen. Well, consider the
following scenario:  

You're making a tile-based game where the tiles are arranged into a square grid.
On this grid, various game objects are placed, and you'd like to find the tile
in the middle of two objects. The obvious solution would be to take the average
of the tile coordinates for both objects. The following illustrates what it
would look like if you attempted to solve this problem using an averaging
function that rounds downwards. The green tile represents one object and the red
tiles represent the other object which which the average is being taken. The
yellow tiles correspond to the location produced by the averaging function.  

![Game grid with downwards rounding
average](/assets/images/grid_downwards.svg){: class="centered_image"}  

Well, that turned out less than ideal. There is an asymmetry in the result that
is ironically caused by the symmetry of the averaging function. To resolve this,
we can instead use `std::midpoint` and bias towards the first input. In this
case, the assumption will be made that the coordinates corresponding to the
green tile are passed as the first parameter. The result of this would instead
look like so:  

![Game grid with midpoint](/assets/images/grid_midpoint.svg){:
class="centered_image"}  

As can be seen here, the behavior is now more consistent in the sense that the
results are symmetrical around the green tile.  

We can also note that this is also an issue (although probably a smaller and
more subtle one) if using floating-point numbers. And more broadly speaking,
this is really an issue for anyone who wants to take the average of two points
in any coordinate system, be that for the purposes of simulations, graphics,
game design, or whatever else.  

## Towards Implementations

Alright, so now that `std::midpoint` has been justified, let's get back to
rounding.

For unsigned integers, we'll say that there are three possible rounding schemes
of interest: down, up, and first-biased. For signed integers, we'll say that are
five: down, up, towards zero, away from zero, and first-biased.  

If we assume that all of these rounding schemes are desirable, we are faced with
the problem of finding how to implement averaging functions that exhibit each of
these rounding behaviors for both unsigned and signed integers, and for at least
the common integer sizes of 8, 16, 32, and 64 bits. Not only do we still face
the problem of avoiding overflow, but there is also another challenge of
achieving the desired rounding behavior. And of course, ideally we'd want them
to be fast, and to that end it makes sense to leverage the existing SIMD
averaging instructions that were previously mentioned. Although this sounds like
a simple problem that has exploded into dozens of mildly more complex problems
there are only subtle differences between the functions at the end of the day,
and they can mostly all be solved using the same general approach.

To the end of establishing that approach, let's take a look at averaging two
unsigned integers with downwards rounding.

As already mentioned, bit hacks for averaging often assume unsigned integers and
a downwards rounding scheme, meaning that creating averaging functions that meet
that criteria is quite an easy task. In particular, there are two common bit
hacks for this:  

```code_block
    (x >> 1) + (y >> 1) + (x & y & 0x1)
    (x & y) + (x ^ y) >> 1
```

This first solution should be fairly intuitive to anyone comfortable with bit
manipulation and who's willing to think it over for a short while, so I won't
explain it here, but I will provide somewhat of a motivation later on.  

The second solution is less intuitive so I'd like to offer an explanation
because understanding how this hack works is necessary for understanding a later
paragraph on how certain ARM instructions can be used to compute 64-bit
averages. If you're not interested in that, then you can just skip this all. 
 
There are at least two ways of understanding the bit hack that I know of but
I'll offer only one. What this approach fundamentally does is leverage the fact
that averaging without overflow becomes a simple matter under two different
circumstances.

Firstly, if the two inputs are the same number, then finding the average is
trivial because it's just that same number.  

Secondly, if the two inputs don't have any set bits in common i.e. `(x & y ==
0)`, then when adding the two numbers together, the carried values are always 0,
meaning that overflow cannot happen. Under this assumption, it's safe to average
the numbers by just adding them together and dividing by 2 (or right shifting by
1) as would normally be done. Furthermore, the addition itself can be simplified
to either a bitwise OR, or a bitwise XOR. (To convince yourself of this, look at
the truth table for bitwise addition, and ignore the row where A and B are both
1.) So the formula for averaging under this assumption could be either `(x | y)
>> 1` or `(x ^ y) >> 1`, but we'll go with the one that uses a XOR for reasons
that will be explained shortly.  

What we now need to do is find a way to split up the problem of averaging two
different integers in the general case, to just these two special cases where
the solutions are trivial.  

Since we already have the solution in front of us, let's just take a look at it.
The first half, `x & y` produces a new integer that consists of only those bits
which are set in both `x` and `y` so it decays to the first special case. On the
other hand, `x ^ y` produces a new integer that consists of only those set bits
which differ between `x` and `y` so it decays to the second special case. So we
can understand this averaging bit hack as the average of what `x` and `y` have
in common plus the average of what `x` and `y` don't have in common, which is
the average of `x` and `y`. 

While these bit hacks are often presented in the context of unsigned averages,
they also work for averaging signed integers if the goal is to round downwards.
The only difference would be that the logical right shift would have to be
replaced with an arithmetic right shift, something which is already handled in a
programming language like C or C++ if you simply change the types of `x` and `y`
to be signed integers.

### Using a Table for Intuition

At this point. I'd like to introduce a way to reach the first formula I
mentioned `((x >> 1) + (y >> 1) + (x & y & 0x1))`, in a way that while not
mathematically rigorous, provides a lot of visual intuition and is also
flexible, making it a handy tool for quickly finding ways to implement all those
averaging functions mentioned earlier.

What we can do is generate a table showing the error between a "naive"
implementation of an averaging function, and a baseline function that delivers
the desired result by whatever inefficient or non-general means necessary.  

The naive function will just be an attempt to take the average of two integers
by rearranging the formula for the average of two numbers `(x + y) >> 1` to
become `(x >> 1) + (y >> 1)` even though this modified formula is still
imperfect in that it fails if both inputs are odd:  
```code_block
    std::uint32_t average_naive(std::uint32_t a, std::uint32_t b) {
    	return (a >> 1) + (b >> 1);
    }
```

The baseline will just cast to a wider type to avoid overflow:
  
```code_block
    std::uint32_t average_downards(std::uint32_t a, std::uint32_t b) {
    	return (std::uint64_t{a} + std::uint64_t{b}) / 2;
    }  
```

Ideally, we'd really want to take an exhaustive look at all possible inputs, but
given the large size of the domain for integers wider than 8-bits, that's not
exactly feasible. Instead, what we'll just take a look at a small subset of it,
and operate under the assumption that any patterns observed within that subset
generalize to the full range of inputs. As it turns out, the assumption does
hold, and if you were so inclined, you could prove that they do.  

So, let's print out what both of these functions produce when both inputs are in
the range of `[0, 8)`

Naive average:  

|   |   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|---|
|   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| 0 | 0 | 0 | 1 | 1 | 2 | 2 | 3 | 3 |
| 1 | 0 | 0 | 1 | 1 | 2 | 2 | 3 | 3 |
| 2 | 1 | 1 | 2 | 2 | 3 | 3 | 4 | 4 |
| 3 | 1 | 1 | 2 | 2 | 3 | 3 | 4 | 4 |
| 4 | 2 | 2 | 3 | 3 | 4 | 4 | 5 | 5 |
| 5 | 2 | 2 | 3 | 3 | 4 | 4 | 5 | 5 |
| 6 | 3 | 3 | 4 | 4 | 5 | 5 | 6 | 6 |
| 7 | 3 | 3 | 4 | 4 | 5 | 5 | 6 | 6 |  
 
Downwards rounding average:  

|   |   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|---|
|   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| 0 | 0 | 0 | 1 | 1 | 2 | 2 | 3 | 3 |
| 1 | 0 | 1 | 1 | 2 | 2 | 3 | 3 | 4 |
| 2 | 1 | 1 | 2 | 2 | 3 | 3 | 4 | 4 |
| 3 | 1 | 2 | 2 | 3 | 3 | 4 | 4 | 5 |
| 4 | 2 | 2 | 3 | 3 | 4 | 4 | 5 | 5 |
| 5 | 2 | 3 | 3 | 4 | 4 | 5 | 5 | 6 |
| 6 | 3 | 3 | 4 | 4 | 5 | 5 | 6 | 6 |
| 7 | 3 | 4 | 4 | 5 | 5 | 6 | 6 | 7 |


Subtracting the naive output table from the baseline downwards rounding table
produces:

Downwards rounding - naive:  

|   |   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|---|
|   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 |
| 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 |
| 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 |
| 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 |

Taking a look at this table, there's a clear pattern to the error. With such a
simple pattern, it should be fairly easy to adjust the naive averaging function
to produce the same results as the baseline function by adding a corrective term
to it. We can note that the error is 1 if both inputs to the averaging function
are odd, and 0 otherwise. We can of course check if an integer is odd by
checking if the low bit is set. So expressing that in code that would be `((x &
0x1) == 1) && ((y & 0x1) == 1)` relying on the implicit conversions from bools
to ints. But that can be easily simplified to just `(x & y & 0x1)`. So with the
addition of the corrective term to the naive average, the solution becomes the
familiar:  

```code_block
    (a >> 1) + (b >> 1) + (a & b & 0x1)
```

So let's continue this and see what happens if we do this with an upwards
rounding function instead of a downwards rounding function.

Upwards rounding - naive:  

|   |   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|---|
|   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| 0 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 |
| 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 2 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 |
| 3 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 4 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 |
| 5 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 6 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 |
| 7 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

Again, we see another clear pattern. We can see that the error is 1 if either
input is odd, so the corrective term could be determined to be quite similar to
the previous one:  

```code_block
    (a >> 1) + (b >> 1) + (a | b & 0x1)
```

So it's neat that we've managed to get these solutions in such a visually
intuitive way, but can this be used to derive a corrective term for
`std::midpoint`? As it turns out, the answer is yes.  

This is the output of subtracting the naive function from the output of
`std::midpoint`.
 
|   |   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|---|
|   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| 0 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 |
| 1 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 2 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 1 |
| 3 | 0 | 1 | 0 | 1 | 1 | 1 | 1 | 1 |
| 4 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 |
| 5 | 0 | 1 | 0 | 1 | 0 | 1 | 1 | 1 |
| 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| 7 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 |

This table shows another pattern, although, a more complex one than the
previous. Taking a look at the lower-left half of the table, there is the same
pattern that we saw when we wanted to round downwards, and taking at look at the
upper-right half of the table there's that pattern that we saw when we wanted to
round upwards. The break between these individual patterns comes across the
diagonal entries of table. The diagonal entries themselves are accurately
computed by either pattern, so there's some leeway in how we choose to define
that boundary as `y < x` and `y <= x` would both work, but I'll go with `x < y`.
We can note that in the case that both inputs are odd, the error is always 1. In
the case that only one input is odd, then the error is 1 if and only if `y < x`.
Turing this into code we get that the corrective term is:  

```code_block
    (a & b) | ((b < a) & (a | b)) & 0x1
```

And just like that, we've found a way to implement averaging functions for all
the unsigned rounding schemes by simply adding a corrective term to a naive
implementation of an averaging function. Naturally, the next thing would be to
try to use this as a way to find ways to implement averaging functions for
signed integer as well. But note that there's no inherent reason that we have to
base these implementations on the naive averaging function. We could also add
corrective terms to the averaging instructions which the hardware may provide in
order to potentially reach a more efficient solution. 

### Corrective Terms:

To ease the process of generating all these tables, I've written a short program
for it. In particular, it prints out the tables for all the possible desired
rounding schemes, for all the rounding schemes exhibited by the
instructions/baselines mentioned earlier, making it easy to coerce any of those
hardware/bit hack implementations into the desired behavior.

The program is available on [Compiler
Explorer](https://godbolt.org/z/9G6dWzsxE). The idea was to redirect the output
to a CSV file which you could then open in spreadsheet software to get a more
visually appealing look at its results.

The following are the corrective terms derived from all of the tables the
program outputs. I believe that all of the following less-than comparisons can
be replaced with a less-than-or-equal comparison, except for the very last one.
Similarly, `x | y` in both `midpoint - naive` corrective terms can be replace
with `x ^ y`.

Also, note that a `-` is used before all comparisons. This can come across as
somewhat redundant, even inefficient, if thinking about scalar code where false
and true convert to `0` and `1` as you could simply remove the `-` from in from
of the comparison and the `& 0x1` from the end and still get the same result.
However, when it comes to comparisons in many SIMD instruction sets, the results
of comparisons are different. Either each lane has all bits cleared or all bits
set, i.e. `0`, or `-1`. The negative sign out in front is meant to reflect this.
Note that this should not be conflated with other negative signs in the
corrective terms. Those are there because negation is acaully part of the
correction.

Additionally, note that some corrective terms involve negating `x` or `y`. This
can be problematic as if either is equal to the smallest representable value for
a given integer type, and without converting to a larger type, attempting to
negate the value will cause overflow. Therefore some additional handling for
this edge case may be necessary. 

#### Unsigned Corrective Terms
* downwards-naive: `+(x & y & 0x1)`
* downwards-upwards: `-(x ^ y & 0x1)`
* upwards-naive: `+(x | y & 0x1)`
* upwards-downwards: `+(x ^ y & 0x1)`
* midpoint - naive: `+((-(y < x) & (x | y)) | (x & y) & 0x1))`
* midpoint - downwards: `+(-(y < x) & (x ^ y) & 0x1)`
* midpoint - upwards: `-((x < y) & (x ^ y) & 0x1)`

#### Signed Corrective Terms
* downwards-naive: `+(x & y & 0x1)`
* downwards-upwards: `-(x ^ y & 0x1)`
* downwards-away from zero: `-(-(-x < y) & (x ^ y) & 0x1)`
* upwards-naive: `+(x | y & x01)`
* upwards-downwards: `+(x ^ y & 0x1)`
* upwards-away from zero: `+(-(x < -y) & (x ^ y) & 0x1)`
* towards zero-naive: `+((-(x < -y) & (x | y)) | (x & y) & 0x1)`
* towards zero-downwards: `+(-(x < -y) & (x ^ y) & 0x1)`
* towards zero-upwards: `-(-(-x < y) & (x ^ y) & 0x1)`
* towards zero-away from zero: `+ -(-x < y) ? -(x ^ y) : +(x ^ y)`
* away from zero-naive: `+((-(-x < y) & (x & y)) | (x | y) & 0x1)`
* away from zero-downwards: `+(-(-x < y) & (x ^ y) & 0x1)`
* away from zero-upwards: `-(-(x < -y) & (x ^ y) & 0x1)`
* midpoint-naive: `+((-(y < x) & (x | y)) | (x & y) & 0x1)`
* midpoint-downwards: `+(-(y < x) & (x ^ y) & 0x1)`
* midpoint-upwards: `-(-(x < y) & (x ^ y) & 0x1)`
* midpoint-away from zero: `(0 < x) ? -(abs(x) < abs(y)) : +(abs(x) < abs(y))`

Now, to be clear, all of this work is strictly unnecessary. If you were so
inclined, you could derive these corrective terms from first principles in a
mathematically rigorous way. The main reason I did this was because this problem
caught my attention late at night one day (I think it would have actually been
morning at the time) meaning  I was too tired to think through all of these and
it was much easier to write the table generating program, which has since been
touched up to be more presentable.

So with all of these corrective terms in hand, implementing these averaging
functions, at least in scalar code, is straight forward. Just add the
appropriate corrective term to whichever baseline average you have available to
you.

## Vectorizing std::midpoint

The original problem that got me thinking about averages was that I wanted to
write SIMD vectorized implementations of `std::midpoint`. This would have been
for vectors of both unsigned and signed integers, of all the standard widths. 

Of course that raised the question of how these functions could be implemented.
Obviously, the first thing I did was look up "std::midpoint implementation"
online, and found Marshall Clow's CppCon talk on `std::midpoint` in which he
goes through his process of discovery for his implementation. However, the
solution he shared during the presentation cannot be adapted for vectorization
without some drawbacks.

```code_block
    template<class Integer>
    constexpr Integer midpoint(Integer a, Integer b) noexcept {
    	using U = std::make_unsigned_t<Integer>;
    
    	int sign = 1;
    	U m = a;
    	U M = b;
    	if (a > b)
    	{
    		sign = -1;
    		m = b;
    		M = a;
    	}
    	return a + sign * Integer(U(M-m) >> 1);
    }
```

Arguably the most readily apparent issue is that the code uses an if statement
to conditionally swap three values around. Conditional statements like this
generally pose a challenge when it comes to vectorization as conditional logic
is almost contradictory to the idea of SIMD. The entire point of conditional
logic is to change which instructions are executed based on input, but SIMD is
all about processing multiple pieces data with a uniform set of instructions. If
all the lanes end up following one side of a branch then it's not much a problem
but if different lanes need to follow different paths, then SIMD starts to break
down. Branchless programming techniques therefore become incredibly important in
situations like these.  
  
In this particular case, the branch isn't a horrible challenge to eliminate. In
his presentation, Clow notes that the branch compiles down to conditional moves
when targeting x86, a good sign that eliminating the branch should be easy. The
if statement itself fundamentally does two things. Firstly, it just ensure that
the value of `sign` is negative if `a` is greater than `b` and positive
otherwise. Secondly, it ensures that `m` and `M` are the minimum and maximum of
`a` and `b` respectively. The first part of this could be replaced with a blend
instruction (a selection between two values using a mask, essentially a
vectorized ternary operator), and the second part could be done via vectorized
min and max instructions. After that, the rest of the code can also be
straightforwardly vectorized with the corresponding vector instructions for
addition, subtraction, shifts, and multiplication.

At least, that is if you're targeting an x86 processor with AVX-512 or an ARM
processor with Neon and don't care about 64-bit ints.

Unfortunately, the landscape of computer hardware is quite complex. The previous
explanation about how to vectorize Clow's solution is heavily flawed in that it
assumes the presence of CPU features which either don't necessarily exist, or if
they do, aren't guaranteed to be available on all architectures. Additionally,
it's a little lacking in the performance department.

x86 does not have vectorized comparison instructions for all native integers
until AVX-512BW. Prior to that, it only has signed comparison instructions for
8, 16, and 32-bit integers. x86 also doesn't have blend instructions until
SSE4.1 and while fairly widespread today, I don't want to make that assumption.
x86 doesn't have vectorized min and max instructions for all integers until
AVX-512F. With just SSE2, there are only unsigned 8-bit and signed 16-bit
min/max instructions available, although the gaps are filled up to 32-bit
integers with SSE4.1. x86 also doesn't have vectorized shift instructions or
multiplication instructions for 8-bit integers at all, and it doesn't have
multiplication instructions for 64-bit integers until AVX-512DQ (I may also note
that multiplication is a bit on the expensive side since, at least on Icelake
processors like my own, 16/32/64 bit multiplications have a latency of 5/10/15
cycles respectively). This is all to say that there are lots of gaps in the
individual instructions that go into this solution that means that it's not
actually easy to implement.

ARM's Neon instruction set is more well-rounded, but it still suffers from the
fact that it doesn't have the 64-bit min/max or multiplication instructions
required by Clow's solution.

The absence of these instructions can be be worked around by emulating them
using other instructions. Unsigned comparisons could be emulated using
corresponding signed comparisons after offsetting the input values. At least on
x86, before they came with SSE4.2, 64-bit comparisons can fall back to scalar
code without too much of a drawback since there are only 2 of them per 128-bit
vector. Blend instructions can be emulated by three bitwise operations, either
an `and`, `or`, and `andnot` or two `xor` ops, and an `and`. The min/max
instructions can be emulated via blends. Vectorized 8-bit shift instructions
could be emulated using 16-bit shifts with masking to clear bits that shouldn't
be set. As far as the multiplication, since it's always either by `1` or `-1`,
it's just conditionally negating an integer and hence it can be replaced with
the bit hack `(n ^ mask) - mask` where `mask` is `-1` if `a > b` and `0`
otherwise. With these tricks up your sleeve you could theoretically create
implementations of midpoint for vectors of all integers regardless of the which
CPU features are available to you.

Individually, any of these workarounds isn't atrocious, but when you consider
that depending on which CPU features you'd like to assume, you may have to use
multiple in combination with each other, the cumulative cost of these
workarounds will have you starting to wonder if there isn't a more performant
alternative.

Clow's solution stands the best chance to be most performant under vectorization
when, for the particular integers you're working with, there are native
vectorized comparison instructions, min/max instructions, and shift
instructions. Vectorized addition, subtraction, and bitwise XOR operations are
supported pretty universally for all integers so while those instructions are
are also required, it's less meaningful to point out their need.   
 
As a concrete example of what that might look like, we can consider a midpoint
implementation for signed 16-bit integers on an x86 CPU with SSE4.1:  

```code_block
    __m128i midpoint_i16(__m128i x, __m128i y) {
        auto m = _mm_min_epi16(x, y);
        auto M = _mm_max_epi16(x, y);
    
        auto sign = _mm_cmpgt_epi16(x, y);
    
        auto tmp0 = _mm_srli_epi16(_mm_sub_epi16(M, m), 1);
        auto tmp1 = _mm_xor_si128(tmp0, sign);
        auto correction = _mm_sub_epi16(tmp1, sign);
    
        return _mm_add_epi16(x, tmp1);
    }
```

It compiles down to the following twelve instructions when compiled with GCC,
and the result is much the same when compiled with Clang, the only difference
being the order of some of the instructions. 

```code_block
    movdqa  xmm4, xmm0
    movdqa  xmm2, xmm1
    movdqa  xmm3, xmm0
    pminsw  xmm4, xmm1
    pcmpgtw xmm3, xmm2
    movdqa  xmm1, xmm0
    pmaxsw  xmm1, xmm2
    psubw   xmm1, xmm4
    psrlw   xmm1, 1
    psubw   xmm0, xmm3
    pxor    xmm1, xmm3
    paddw   xmm0, xmm1
```

However, for reasons previously mentioned, assembly this terse is far from
always possible so we might begin to look for alternative approaches. For
example, you might begin to wonder if those averaging instructions and the
relevant corrective terms that were mentioned earlier could be used to make a
more terse implementation of `std::midpoint`. It looks like using one of those
averaging instructions could save you at least three instructions, and many of
the corrective terms are relatively cheap to implement, just a few bitwise ops.
This was actually the idea that I had in my head when I thought of writing the
program that output the difference tables. Originally, I produced the table for
all 256x256 possible pairs of inputs when operating on unsigned 8-bit integers.
With the pattern clear I decided cut back to a subset of that to make things
more manageable. 

For reference, the upper-left corner of that table is as follows:  
 
|   |    |    |    |    |    |    |    |   |
|---|----|----|----|----|----|----|----|---|
|   | 0  | 1  | 2  | 3  | 4  | 5  | 6  | 7 |
| 0 | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0 |
| 1 | -1 | 0  | 0  | 0  | 0  | 0  | 0  | 0 |
| 2 | 0  | -1 | 0  | 0  | 0  | 0  | 0  | 0 |
| 3 | -1 | 0  | -1 | 0  | 0  | 0  | 0  | 0 |
| 4 | 0  | -1 | 0  | -1 | 0  | 0  | 0  | 0 |
| 5 | -1 | 0  | -1 | 0  | -1 | 0  | 0  | 0 |
| 6 | 0  | -1 | 0  | -1 | 0  | -1 | 0  | 0 |
| 7 | -1 | 0  | -1 | 0  | -1 | 0  | -1 | 0 |

As a reminder, the corrective term for `midpoint - upwards` for unsigned
integers that was mentioned earlier and the one that can be seen from the above
table is: 

```code_block
    -(-(x < y) & (x ^ y) & 0x1)
```

So how does this approach compare to a vectorized version of Clow's solution?
Well, we can compare the earlier midpoint implementation to one for unsigned
16-bit integers on the same target platform. This may seem a bit suspect since
the earlier implementation was for signed 16-bit integers and now we're
comparing to an implementation for unsigned 16-bit integers. However, the idea
here is to give each approach the best chance since each individually may not
necessarily be optimal across all possible CPU feature sets or integer types.
Additionally, if unsigned and signed versions of instructions exist, they'll
almost certainly have the same cost (I can't think of a counterexample but I'm
unwilling to claim that this is always the case). This comparison is actually
somewhat biased in favor of the earlier signed implementation since recall that
with only SSE4.1, x86 does not have vectorized unsigned comparisons. However,
further recall two important details. The first being that the less-than
comparison in the corrective term can be replaced with a less-than-or-equal
comparison, and the second being that SSE4.1 offers an unsigned 16-bit min
instruction. This opens the possibility to emulate a less-than-or-equal
comparison as if the minimum of two integers `x` and `y` is `x`, we know that
`x` is less than or equal to `y`, and we *do* have 16-bit equality comparison
instructions available to us as early as SSE2. 

```
    __m128i midpoint_u16(__m128i x, __m128i y) {
        auto avg = _mm_avg_epu16(x, y);
    
        auto c0 = _mm_cmpeq_epi16(x, _mm_min_epu16(x, y));
        auto c1 = _mm_and_si128(c0, _mm_xor_si128(x, y));
        auto c2 = _mm_and_si128(c1, _mm_set1_epi16(0x1));
    
        return _mm_sub_epi16(avg, c2);
    }
```

The generated assembly code is shorter than the previous code while still not
generally relying on expensive instructions, which is generally a good sign.

```code_block
    movdqa  xmm2, xmm0
    pavgw   xmm0, xmm1
    movdqa  xmm3, xmm2
    pminuw  xmm3, xmm1
    pxor    xmm1, xmm2
    pand    xmm1, XMMWORD PTR .LC0[rip]
    pcmpeqw xmm2, xmm3
    pand    xmm1, xmm2
    psubw   xmm0, xmm1
```

Of course, it does involve a load, and there is a possibility that it could lead
to a cache miss. But if you're using the averaging function with sufficient
temporal locality then that won't be much of an issue, and  it's almost
certainly not an issue if you're using this code inside of a loop as the load
could just happen once up front. Not to mention that if targeting newer
architectures, the load may not be necessary at all so the code gen could
potentially be even better than that shown here.

A rough comparison between these two implementations can be made through the use
of the uiCA simulator. Since the idea was to target a machine with SSE4.1
support I chose to have the simulation target the Sandy Lake microarchitecture
as it's the oldest that that the simulator supports and it doesn't have AVX2,
the next major extension to x86 that touches upon integer manipulation. With a
couple additional `vmovq` instructions to simulate loading values into `xmm0`
and `xmm1`, thereby avoiding introducing dependencies that would skew the
results, the predicted throughput of the first assembly snippet is
[5.15](https://bit.ly/3jvQ7N5), while for the second it's
[4.09](https://bit.ly/3VygMWY), a decent improvement.  

Granted, this isn't a benchmark, and there are several caveats. There are
several more ISA extensions and integers of different sizes to be considered.
Not to mention that when it comes to such short snippets of code, the hardware
and the context in which they're used may impact performance more than any
individual change to the code itself. But the purpose of this has more to do
with demonstrating that there's merit to exploring multiple different approaches
to implementing vectorized versions of `std::midpoint`, and more broadly
speaking, the other averaging functions. 

## Optimization Opportunities and Little Tricks

The following are a list of potential optimizations that could be used in
implementing these averaging functions. Unfortunately, I can't say that I have a
magic bullet for implementing each possible averaging function on every possible
architecture but these tricks combined with what I've mentioned so far should be
enough to have at least one way to implement each.

### Common Terms
Note that some of the corrective terms feature `x ^ y` and/or `x & y` in them.
Also note that the second bit hack for downwards rounding averages, `(x & y) +
(x ^ y) >> 1` also features these terms so there's a small potential for reuse
if implementing some of those averaging functions as corrections to that bit
hack. Also note that I mentioned that in some of the corrective terms `x | y`
could be substituted for `x ^ y`.  
 
For example, if implementing midpoint in terms fo the downwards rounding
bit-hack, you could do the following:

```code_block
    std::uint32_t midpoint(std::uint32_t x, std::uint32_t y) {
        auto x_and_y = x & y;
        auto x_xor_y = x ^ y;
        auto avg = (x_xor_y >> 1) + (x_and_y);
        auto bias = (y < x) & x_xor_y;
    
        return avg + bias;
    }
```

If the corrective term and the uncorrected average both happen to have a common
term in them then of course your compiler will figure it out, but note that the
compiler will only be able to do this if you choose a corrective term and
baseline average implementation that have shared terms to begin with.

### Comparisons
There was a short paragraph explaining the motivation for putting a `-` before
all the comparisons in the corrective terms. Reading through it may have
suggested to you one possible optimization opportunity. In particular, if you
have `0` and `-1` and you'd like to end up with `0`, and `1`, then just negate
the numbers. Zero out a register (which is basically free given the zeroing
idioms unit on modern CPUs) and then use a subtract instruction to compute `0 -
x`, thereby removing the need for the `& 0x1` at the end of the corrective
terms. In my dabbling so far, this has not lead to a performance increase but
I'd be surprised if it doesn't make a difference somewhere. 

### Using unsigned operations for signed averages and vice versa

Even though x86 features unsigned averaging instructions, it's still somewhat of
a challenge to implement signed 8-bit averages due to the lack of 8-bit shifting
instructions. Division by two is of course an important part of computing
averages and not being able to do it quickly via a shift poses a pretty big
challenge for creating performant implementations.  
 
It was mentioned previously that unsigned comparisons could be emulated by
offsetting the inputs before using a signed comparison. In particular, if
operating on N-bit integers, either adding or subtracting `1 << (N - 1)` to both
inputs would achieve the desired result. Note however that all this would do is
flip the leading bit so this can be somewhat simplified by using a bitwise xor
instead of addition or subtraction instructions. While addition/subtraction
instructions generally have the same latency as a xor instruction of just 1, xor
instructions sometimes have slightly better throughputs. While it's still not
ideal, there are worse things in the world than a couple xor instructions.

It's already been mentioned that unsigned min/max instructions can be used to
perform less-than-or-equal or greater-than-or-equal comparisons.

Broadly speaking though, this same trick can be applied to abuse many
unsigned/signed integer instructions for the purpose of signed/unsigned integer
manipulation. In particular, on x86, you could use the unsigned 8-bit and 16-bit
average instructions as part of implementations of signed 8-bit and 16-bit
averaging functions at the cost of 2 xor operations before and 1 xor operation
afterwards.

Alternatively, there is another trick that could be used in such a case. Note
that if you're dealing with signed integers, and both inputs are positive or
both inputs are negative, then using the unsigned averaging instructions will
deliver the desired results. In the case that one input is positive and one is
negative, the error is easily corrected. As it turns out, the error is just that
the sign bit is flipped. So we just need to flip it when the two inputs have
different signs: `((x ^ y) & 0x80) ^ avg(x, y)`. In terms of the number of
instructions required, this isn't any cheaper than offsetting using xors like
mentioned earlier, however it could be beneficial in other ways. One is that
it's amenable to the potential optimization mentioned in the next section.
Beyond that, it might just turn out to schedule better, depending on context.

### X86's ternary logic

When it comes to x86 machines that have AVX-512F, the corrective terms can
generally be computed in fewer instructions thanks to the
`vpternlogd`/`vpternlogq` instructions. What these instructions do is allow the
programmer to perform any 3-operand logical operation using the three
corresponding bits from three different registers. So it's essentially a
generalization of the familiar bitwise operations like OR, AND, and XOR, to
instead take in three operands, and for the output to be a choice within the
programmer's hands. It's essentially a Swiss army knife for performing logical
operations. The behavior of the instruction is dependent on an 8-bit immediate
value passed to the instruction. The nth bit of this immediate value corresponds
to the output of the nth row in a truth table with 3 inputs.   
 
If we call that immediate value Z, then that means that the truth table for the
instruction will be as follows:
 
| A | B | C |  Z |
|---|---|---|----|
| 0 | 0 | 0 | Z0 |
| 0 | 0 | 1 | Z1 |
| 0 | 1 | 0 | Z2 |
| 0 | 1 | 1 | Z3 |
| 1 | 0 | 0 | Z4 |
| 1 | 0 | 1 | Z5 |
| 1 | 1 | 0 | Z6 |
| 1 | 1 | 1 | Z7 |

So for example, if we wanted the instruction to perform `A ^ B & C` , then we'd
fill in the Z column with the values that this expression would produce given
the specified inputs and we'd get `0b00101000` or `0x28`. Computing this
expression in particular is useful since many corrective terms involve `x ^ y &
0x1`.

GCC and Clang both seem to handle this already however so doing it manually is a
bit redundant unless directly writing assembly.

### AVX-512 Masking

One of the major features of AVX-512 is the addition of mask registers. The
majority of AVX-512's instructions can be used in conjunction with these mask
registers and one of the ways that they can be used is to zero out lanes in the
vector register if the corresponding bit in the mask isn't set.

This is useful here because many corrective terms feature comparisons and
AVX-512's comparison instructions all produce masks. This is unlike earlier
instruction sets where comparisons simply populated each lane in a regular
vector register with all ones or all zeros. And it so happens this zero-ing out
behavior is exactly what we're doing with the comparisons in the corrective
terms.

Since you can use mask registers with just about any instruction, there's a lot
of potential places where it could be used, and to a large extent, it doesn't
seem to matter too much, however you probably don't want to do it too early. If
you do, then an instruction that could otherwise be executed in parallel with
the comparison instruction will instead have to wait for the comparison to
complete. I've found that just using it one of very last instruction possible to
work well.

### 64-bit Averages on ARM64

ARM's averaging instructions are neat, but since they're not available for
64-bit integers, those have to be implemented in software. The classic bit-hack
for averaging `(x & y) + (x ^ y) >> 1` explained earlier can be used for this
purpose. Although the trick only requires 4 explicit operations as is, it can be
further refined on ARM thanks to the `USRA`/`SSRA` instructions. These allow for
a combination of a logical/arithmetic right shift, and a subsequent addition.
These are available for 64-bit integers, allowing downwards rounding averages to
tbe be computed in just three explicit operations.

```code_block
    uint64x2_t average_downwards(uint64x2_t x, uint64x2_t y) {
        return vsraq_n_u64(vandq_u64(x, y), veorq_u64(x, y), 1);
    }
    
    int64x2_t average_downwards(int64x2_t x, int64x2_t y) {
        return vsra_n_s64(vand_s64(x, y), veor_s64(x, y), 1);
    }
```

When it comes to averaging with upwards rounding, ARM makes that easy too. In
particular ARM features the instructions, `URSHR` and `SRSHR` which perform what
it calls rounding right shifts. We know that shifting right is the same as
dividing by two with the understanding that the result is rounded downwards.
Well, these instructions are equivalent to dividing by two with the
understanding that the result is rounded upwards instead. What's even better is
that these rounding right shifts can be performed in combination with a
subsequent addition just as can be done with regular right shifts by using the
`URSRA` and  `SRSRA` instructions. The result is that it again only takes three
explicit operations to average while rounding upwards.

```code_block
    uint64x2_t average_upwards(uint64x2_t x, uint64x2_t y) {
        return vrsraq_n_u64(vandq_u64(x, y), veorq_u64(x, y), 1);
    }
    
    int64x2_t average_upwards(int64x2_t x, int64x2_t y) {
        return vsra_n_s64(vand_s64(x, y), veor_s64(x, y), 1);
    }
```
