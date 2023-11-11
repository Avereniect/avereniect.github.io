---
layout: blog_post
title: Levenshtein Edit Distance with AVX-512
description: A discussion of how changing the memory layout of the table used in Levenshtein edit distance can make it more SIMD friendly and how this can be leveraged with AVX-512. 
---

Levenshtein edit distance is a one of the most famous of dynamic programming
problems. For anyone not particularly familiar with the algorithm or perhaps
just in need of refresher, I'd suggest checking out [What's a Creel's video on
the algorithm](https://www.youtube.com/watch?v=Cu7Tl7FGigQ). At the very least,
it'll be helpful for understanding the rest of the writeup.

A short while back I watched the video for a second time and given my fondness
for SIMD vectorization, I thought it would be fun to apply AVX-512 to this
problem.

## Opportunities for Parallelism
Before any algorithm can meaningfully benefit from SIMD vectorization, an
opportunity for parallelism must be found. One way to find such opportunities is
to identify the things which limit our ability to parallelize the algorithm, and
to then to find the circumstances under which those limitations don't apply.

Looking at the tabulated Levenstein edit distance algorithm, it's fundamentally
based on the filling a two-dimensional table. In order to compute a table entry,
three entries must be known beforehand: the one above the current cell, the one
to the left of the current cell, and the one to the upper-left of the current
cell. If we wish to compute the value of multiple cells in parallel then those
cells must simultaneously satisfy these requirements. 

Taking a look at the "cabbages" vs. "rabbit" example from What's a Creel's
Video, this would mean that if we wanted to compute the value for the cell
marked X, then the values for cells A, B, and C would be necessary to
have beforehand.

![Table with diagonals](/assets/images/cost_table_dependencies.svg){:
class="centered_image"}

Before the main table-filling loop begins, the base cases are handled by
populating the cells within the top row and the left-most column with the index
of their column/row respectively. These base cases are also illustrated in the
previous image.

The only cell that meets the evaluation criteria immediately following this
initialization is the cell at (1, 1). However, once that cell is evaluated, the
cell to the right, and the cell below can be evaluated in parallel. This
statement remains true for subsequent iterations, which produces a pattern where
the cells that can be evaluated in parallel are arranged into ascending
diagonals within the table. This pattern is illustrated in the following image
where the cells with the same colors can be evaluated at the same time.

![Table with diagonals](/assets/images/cost_table_diagonals.svg){:
class="centered_image"}

This fact becomes the opportunity for parallelism which we can exploit through
the use of SIMD vectorization.

## Memory Layout
If attempting to create a parallelized implementation that follows this approach
there is one particularly important detail with regards to memory layout.
Ideally, the memory layout of the diagonals would be contiguous so that all it
takes to populate a vector register is a single SIMD load instruction (which
operates on contiguous bytes in memory).

The most obvious way to go about accomplishing this would be to diagonalize the
memory layout of the table. This is to say that the table's memory layout should
follow the diagonals which were previously illustrated. The table as a whole
could still be contained within a single contiguous allocation. Within this
allocation, the diagonals would be stored contiguously, starting from the
top-left to the bottom-right of table.

While we're pondering memory layout, we might as well note another optimization.
We can note that at any given time, only the two previous diagonals are required
to populate the current diagonal. This means that instead of creating an
allocation large enough to store the entire table, it need only be large enough
to store the previous two diagonals and the current diagonal. In a moment we'll
see that it' actually possible to further reduce this to keeping around only
enough memory to store two diagonals. The memory to store these diagonals could
be allocated upfront as a single contiguous buffer equal to three/two times the
maximum size of a diagonal (which we'll also look at how to compute in a short
while). For the sake of this discussion, I'll refer to these as diagonal
buffers, although in reality it'll just be a singular large buffer as that would
be the most efficient solution to this problem.

With the diagonal buffers typically being larger than the actual diagonals which
populate them, this brings up a question of how the diagonals should be packed
into the available memory. Perhaps the most obvious approach would be to pack
each diagonal at the start of the diagonal buffers. After all, this is how this
problem is most-commonly addressed, such as is the case for a dynamically-sized
array for example. This approach could be visualized as follows on the left
where the rows in the illustrations are actually the diagonals from the earlier
table.

<div class=row>
    <img src="/assets/images/diagonals_packed_left.svg" alt="" class="centered_image_narrow" />
    <img src="/assets/images/relative_positions_packed_left.svg" alt="" class="centered_image_narrow" />
</div>

However, this naive solution presents a problem. Although the ultimate goal is
to produce a vectorized algorithm, for now consider a purely scalar approach to
understand what the issue is. Looking at the figure on the right, you may notice
that the positions of A, B, and C relative to X are not consistent. The selected
table entries show the variety you'll find there. If `I` is the index of the
current cell X, then the index of A may either be `I` or `I + 1`. The index of B
may either be `I - 1`, `I`, or `I + 1`, and C's index may be either `I - 1` or
`I`. Although it would be possible to proceed with the current memory layout,
the need to handle this variation adds additional and unnecessary complexity to
the body of the table-filling algorithm.

However, a simple change in the way that diagonals are packed into the diagonal
buffers can eliminate this inconsistency. Again, it's illustrated by colorful
boxes below. Under this alternative scheme, the diagonals whose first element
comes from the left-most column of the table are packed to the right of the
diagonal buffers and those diagonals whose first element comes from the bottom
row of the algorithm are packed to the left. 

<div class=row>
    <img src="/assets/images/diagonals_packed.svg" alt="" class="centered_image_narrow" />
    <img src="/assets/images/relative_positions_packed.svg" alt="" class="centered_image_narrow" />
</div>

Note that now the relative positions of A, B, and C are consistent where they
were previously inconsistent. Indeed they're now consistent for all cells in the
table, not just those illustrated.

Another optimization that comes from this memory layout is that it's possible to
maintain only 2 diagonal buffers instead of 3. Let's say that we're in the
middle of running the algorithm, filling in some diagonal and we're working on
the `I`th entry in the current diagonal. We'll call the previous diagonal `d0`,
and the diagonal before that `d1`. We'll write the result for the `I`th entry
directly into `d1[I]`. This is safe because the value previously at `d1[I]` is
not necessary by this time. If you look at the previous figure(right), you'll
see that `d1[I]` is only used to compute the `I - 1`th entry, that is, the last
entry to be computed. Hence, the value we're overwriting has already served its
purpose and is safe to overwrite.

This optimization can also be applied when vectorizing the algorithm for a
similar reason. Although, the reasoning there isn't necessarily that the values
we're overwriting has been used in the filling of previous cells. It's mostly
that the values we're overwriting are used by the current vector of entries and
hence, the values still get used as intended.

## A Scalar Diagonal Implementation
It's worthwhile to create a purely scalar solution that follows the diagonal
memory layout followed so far. There are more details still need to be fleshed
out and this will serve as an opportunity to address them without having to deal
with the added complexity of writing vectorized code.

First, we must compute the number of diagonals in the table. We can note
that for every entry in the table's left-most column and the bottom-most row
there would be a diagonal whose first element would be that entry. The sum of
these values would be `1 + L0 + 1 + L1` and to remove the double-counted entry
in the lower-left corner of the table we of course subtract one for a final
formula of `1 + L0 + L1`. We'll call this value `k0`.

Furthermore, we need to determine the maximum length of a diagonal in the table so
that the diagonal buffers can be allocated at an appropriate size. It should be
intuitively obvious that this is dependent on the size of the table, and that
the table's size is dependent on the length of the strings being compared. To
that end, let `L0` be the length of the first string and `L1` be the length of
the second string. Then the table would be `L0 + 1` cells wide and `L1 + 1`
cells high. If it's assumed that the input strings are of equal length then the
table is square, and the size of the longest diagonal would straightforwardly be
equal to the table's width or height since the longest diagonal would have a
cell for ever row/column. If the input strings are not equal in length, and the
table is rectangular, then you can imagine that it contains a square sub-table.
This square sub-table's sides would be equal in size to the full table's
smallest dimension. Hence, the length of the largest diagonal(s) in that case
would be `min(1 + L0, 1 + L1)` or `1 + min(L0, L1)`. We'll call this value `k1`.

Then we must also compute at which index within the diagonal buffers each
diagonals begins/ends. However, note that the table cells which are bases cases
can be handled in a scalar fashion once per iteration which runs over the
diagonals. This is more efficient because otherwise there would be a need to
include more logic within the loop which runs over the elements in the current
diagonal. This means that some diagonals may not need to have their very first
or last entries included in the range covered by the indices we're trying to
compute. The last figure once again offers a lot of intuition. The left side of
the diagonals follows a linear function beginning with `k0 - 1` and decreasing
by one each iteration until reaching 0, so the formula for the first index of
the `i`th diagonal would be `max(k0 - i, 0)`. By a similar analysis, the
formula for the index of the last element to be processed, or rather, one past
that element would be `min(k1 - 1, k0 - i)`.

However, these formulas are actually based on the assumption that the first
string is longer than the second string. If the strings are passed in reverse
order, then the table is transposed and the diagonals look like this:

![Transposed diagonals](/assets/images/cost_table_diagonalized_transpose.svg){:class="centered_image_narrow"}

The arrangement of the base case entries isn't as neat, and doesn't follow the
established formula. Although it's possible to handle this through mild
augmentations to the computations of the aforementioned indices, it can also be
straightforwardly avoided by simply swapping the strings upfront at minimal
cost.

However, note that this assumption that the first string is the longer of the
two also simplifies the computation of at least one of the previously mentioned
values. The maximum length of a diagonal can be implemented simply as `1 + L1`.

We might as well note that the first diagonal will always just be a singular 0
and that the second diagonal (should both strings be non-empty) will only ever
contain a pair of 1s. Should either input string be empty then we can just
return the length of the longer string. This means we can cheaply handle these
edge upfront and skip a couple of the algorithm's interior loop.

With those details worked, out we can construct a scalar implementation of the
algorithm, the code of which is available on [Compiler
Explorer](https://godbolt.org/z/o1oKe4n68). The implementation is somewhat
clunky, but ultimately a functional example of diagonalizing the memory layout
and serves as a starting point to construct th e vectorized implementation.

## AVX-512 Implementation
With the scalar implementation on hand, now it's time to actually vectorize the
algorithm. Beyond the changes that have already been made, the real changes are
to the algorithm's interior loop, the one which iterates over the current
diagonal.

We can begin by noting that, as is so often the case when processing an array in
a vectorized fashion, the vector of elements being processed will be fully
populated for all iterations except for the last one. For the very last
iteration we can leverage AVX-512's masked instructions and for all earlier
iterations we can have a separate implementation that's assumes a full vector
and is therefore a little simpler.

In line with this, the we'll compute the number of iterations.

The loading of the values of `A`, `B`, and `C` is a straightforward
vectorization of the scalar approach. The only real difference is the use of
`j_begin + j` to compute the index of the current .

Computing the cost via construction is also straightforwardly vectorizable due
to the existence of the `vpminub` instruction.

The loading of the character vector is the largest challenge which is found
here. Although loading the `ch0` vector is straightforward, loading the `ch1`
vector is marginally more complex because the characters actually have to be in
reverse order for the logic to work appropriately. Reversing the contents of a
vector at a byte-wise granularity can be done using a `vpermb` instruction
rather easily, however note that there is a slight problem that arises when the
vector isn't full. It's necessary to perform a load such that last n elements in
the vector are the first n elements of `str1` so that the reversal of the
vector's elements will put the elements at the right location. (Note that
technically the pointer passed to the load will actually have to be before the
start of `str1`. Technically, the mere computation of this pointer constitutes
UB according to the C++ Standard since it does not point to an object, and does
not point to a memory address within an array or one past the last element,
however in a practical sense, this is not likely to be a real concern).

Again, [the code is available on Compiler
Explorer](https://godbolt.org/z/W4eEdoaP1). 

## Benchmarks
Naturally, the next step is to compare the actual performance of the vectorized
algorithm. However, trying to analyze the performance of these algorithms is
slightly non-trivial because the performance is not a simple linear function of
the size of the input. Additionally, neither is the speedup that we can expect
from this vectorization. It's not until there are 64 non-base-case elements
in a diagonal that the 512-bit vectors will be used to their full extent.

Furthermore, the scalar implementations are liable to have branches whose
predictability will depend on the inputs. Although whether actual branches or
conditional moves are used will itself depend on the compiler and compiler flags
used. I don't attempt to account for this here but the benchmark code should be
easy to modify to this end.

Instead, here I choose to simply compare the performance under the assumptions
that the two input strings are of equal length and there is a 50% chance that
any pair of characters at the same index will be equal. Under these assumption,
the benchmarks were performed for 9 different lengths of inputs strings.

In addition to the typical implementation of the tabulated Levenshtein edit
distance algorithm, I also threw in the variant which maintains only the only
maintains the previous row just to have more context to interpret the vectorized
results against.

I ran the tests on my i7-1065G7, compiled them with GCC 12.1.0. The executable's
outputs were run 50 times and averages computed from that.

The result of [the code](https://godbolt.org/z/M5dbGzvs4) are as follows:

### Full Table Implementation:

| Time(ms)  | 1    | 2    | 4    | 8    | 16   | 32   | 64   | 128  | 255  |
|-----------|------|------|------|------|------|------|------|------|------|
| Mean      | 1.35 | 1.59 | 2.54 | 6.04 | 23.0 | 100  | 452  | 2050 | 8388 |
| Std. Dev. | 0.20 | 4.34 | 97.4 | 0.31 | 0.92 | 2.73 | 7.07 | 19.7 | 80.2 |
{: .table.wide }

### Single Row Implementation:

| Time(ms)  | 1    | 2    | 4    | 8    | 16   | 32   | 64   | 128  | 255  |
|-----------|------|------|------|------|------|------|------|------|------|
| Mean      | 1.30 | 1.61 | 2.52 | 6.05 | 21.4 | 81.6 | 340  | 1397 | 5468 |
| Std. Dev. | 0.19 | 3.84 | 72.5 | 0.25 | 0.86 | 1.99 | 5.86 | 14.5 | 49.2 |
| Speedup   | 1.04 | 0.99 | 1.01 | 1.00	| 1.07 | 1.23 | 1.33 | 1.47 | 1.53 |
{: .table.wide }

### Scalar Diagonal Implementation:

| Time(ms)  | 1    | 2    | 4    | 8    | 16   | 32   | 64   | 128  | 255  |
|-----------|------|------|------|------|------|------|------|------|------|
| Mean      | 1.53 | 1.87 | 3.61 | 9.56 | 31.0 | 105  | 413  | 1654 | 6453 |
| Std. Dev. | 0.23 | 2.88 | 88.6 | 0.41 | 1.02 | 2.9  | 7.99 | 17.2 | 59.7 |
| Speedup   | 0.88 | 0.85 | 0.70 | 0.63 | 0.74 | 0.95 | 1.09 | 1.24 | 1.30 |
{: .table.wide }

### AVX-512 Diagonal Implementation:

| Time(ms)  | 1    | 2    | 4    | 8    | 16   | 32   | 64   | 128  | 255  |
|-----------|------|------|------|------|------|------|------|------|------|
| Mean      | 1.31 | 2.50 | 4.90 | 9.56 | 20.7 | 40.0 | 80.8 | 165  | 349  |
| Std. Dev. | 0.13 | 0.00 | 29.2 | 0.19 | 0.37 | 0.59 | 1.06 | 6.56 | 26.3 |
| Speedup   | 1.03 | 0.64 | 0.52 | 0.63 | 1.11 | 2.50 | 5.59 | 12.4 | 24.0 |
{: .table.wide }

Interestingly enough, it seems that all alterative approaches yield performance
improvements for large enough inputs. For shorter strings there isn't
necessarily a clear benefit to the alternative techniques. The vectorization
approach doesn't appear to perform as well as the full table scalar
implementation until the input strings are at least 16 characters long, although
there is a decent increase in performance as the input strings become
meaningfully larger than that. However, the vectorized algorithm doesn't support
strings longer than 255 so as things are the performance gains appear to be
capped at around 24x.

## Potential Improvements
The current vectorization approach is rather naive from a number of angles,
imposing various limitations. There are however ways to go about working around
this.

### Encoding
It will have stood out to those familiar with text encoding schemes that there
is an assumption made that each character in the string is represented using
exactly a single byte. Although in theory, this algorithm would work with any
8-bit character scheme, it's likely to break in practice. Multi-byte encoding
and variable-width encoding schemes are commonplace. In the case of
constant-width encoding schemes, it's straightforward to modify this algorithm
to operate on 16-bit or 32-bit characters. However, for variable-width encoding
schemes, it's a bit more complex since the vectorized approach fundamentally
relies on an assumption that all characters are represented with a constant
amount of bytes.

One possible solution to this problem would be to pre-process the string
upfront. A map could be used to associate each unique character, codepoint, or
grapheme, whichever is relevant, to some arbitrary integer value. This integer
value would only have to be wide enough to encode as many unique values as there
are keys within the aforementioned mapping. The algorithm could run then on an
array which operates on an array of these mapped integers instead of the
original data. 

### Overflow
Perhaps a more straightforward issue, the algorithm breaks down when either
input string is longer than 255 characters long. Making the strings longer would
require representing a cost of 256, which would overflow an 8-bit integer. In a
practical sense, in many programs strings are generally quite short, and 255
characters may very well be good enough. However, the previous benchmarks show
that there isn't a benefit under those circumstances anyways.

After coming up with the approach shown here, I did look through a handful of
papers which described alternative approaches and some did include approaches
that, if vectorized as done here, could work for strings of any length so long
as the edit distance is no greater than 255.

A more straightforward approach to addressing this problem would be to use
16-bit integers to store the edit distances, however that will likely infringe
upon the performance improvements, as well as add complexity to the
implementation.

### Register Only Implementation
AVX-512 features 32 512-bit registers. Looking through the disassembly on
Compiler Explorer, it's possible to see that the register zmm0-zmm5 are used,
but that still leaves 26 registers free for other purposes. Collective, these
registers could store 1664 bytes worth of information. 
 
The fine details here will clash with the aforementioned encoding issues, but if
we continue with the assumption of one byte per character, that means 1664
characters that could be stored, or two diagonals containing up to 832 bytes
each. It was previously mentioned that the maximum size of a diagonal could be
computed as `1 + min(L1, L2)`. This means that so long as one of the strings
passed to the algorithm is no more than 831 bytes in length, then there is no
need to perform a dynamic memory allocation or perform any load/store
instructions to handle the diagonals. Although at the same time, with the limit
of being only to work with strings up to 255 bytes in length anyways, much of
this potential is not usable with the current approach regardless.

More to the point though, this means that the current algorithm could be
modified such that no diagonals need to be stored to/loaded from memory,
potentially improving performance.

If nothing else, creating such a register-only implementation would be an
interesting problem to deal with next.
