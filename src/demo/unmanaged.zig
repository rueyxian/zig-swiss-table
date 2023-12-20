const std = @import("std");
const debug = std.debug;
const mem = std.mem;
const Allocator = std.mem.Allocator;

const SwissTable = @import("swiss_table").SwissTable;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){ .backing_allocator = std.heap.page_allocator };
    defer debug.assert(gpa.deinit() == .ok);
    const allocator = gpa.allocator();

    const Context = struct {
        pub fn hash(_: @This(), s: []const u8) u64 {
            return std.hash.Wyhash.hash(0, s);
        }
        pub fn eql(_: @This(), a: []const u8, b: []const u8) bool {
            return mem.eql(u8, a, b);
        }
    };

    const lane_count = 16;
    var map = SwissTable([]const u8, u32, Context, 80, lane_count).init(allocator);
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
}
