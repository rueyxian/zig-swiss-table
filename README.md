# zig-swiss-table

A possibly SIMD-operated Hash Table in Zig.

## Note

- This repository primarily serves for personal experimenting and learning purposes.
- I did not come up with anything original. All the credit goes to: 
	1. [CppCon 2017: Matt Kulukundis "Designing a Fast, Efficient, Cache-friendly Hash Table, Step by Step"](https://www.youtube.com/watch?v=ncHmEUmJZf4) 
	2. [The standard library hash map](https://ziglang.org/documentation/master/std/#A;std:hash_map).
- The API is exactly the same as the `std.hash_map` as it is a copy-pasta from it, except for changing to `camel_case`, pretending I did some meaningful work.
- The library implements 'arbitrary group positions' as Matt mentioned at the [46:00 timestamp](https://youtu.be/ncHmEUmJZf4?si=dP70kLBB1sSZL3ns&t=2756).
- I haven't done any benchmarking. Who knows, it might be worse than `std.hash_map`.
