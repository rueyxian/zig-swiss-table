# zig-swiss-table

A possibly SIMD-operated Hash Table in Zig.

## Note

- This repository primarily serves for personal experimenting and learning purposes.
- Nothing orignal here. All the credit goes to: 
	1. [CppCon 2017: Matt Kulukundis "Designing a Fast, Efficient, Cache-friendly Hash Table, Step by Step"](https://www.youtube.com/watch?v=ncHmEUmJZf4) 
	2. [The standard library hash map](https://ziglang.org/documentation/master/std/#A;std:hash_map).
- The API is exactly the same as the `std.hash_map` as it is a copy-pasta from it, and changing to `camel_case`, pretending I did some meaningful work.
- This hash map implements 'arbitrary group positions' as Matt mentioned at the [46:00 timestamp](https://youtu.be/ncHmEUmJZf4?si=dP70kLBB1sSZL3ns&t=2756), but not sure if it has been done it correctly.
- This hash map uses 16-lane instructions by default, regardless of the CPU architecture, although I suppose there is an optimal choice for a different architecture.
- No benchmarking has been done. Who knows, it might be worse than `std.hash_map`.


## Example

Declaration works exactly the same as `std.hash_map`. Except it requires an additional 4th parameter for `lane_count`.

```zig
const allocator: Allocator = std.testing.allocator;
const lane_count = 16;
var map = SwissTable([]const u8, u32, StringContext, 80, lane_count).init(allocator);
defer map.deinit();

try map.put_no_clobber("a", 1);
try map.put_no_clobber("b", 2);
try map.put_no_clobber("c", 3);
try map.put_no_clobber("d", 4);
std.debug.assert((try map.fetch_put("b", 5)).?.value == 2);
std.debug.assert((try map.fetch_put("d", 6)).?.value == 4);

std.debug.assert(map.get("a").? == 1);
std.debug.assert(map.get("b").? == 5);
std.debug.assert(map.get("c").? == 3);
std.debug.assert(map.get("d").? == 6);
```

`StringSwissTable` and `AutoSwissTable` use `16` lane count by default, as previously mentioned.

```zig
var map1 = StringSwissTable(u32).init(allocator);
var map2 = AutoSwissTable(u8, u32).init(allocator);
```