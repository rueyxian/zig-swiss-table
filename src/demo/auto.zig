const std = @import("std");
const debug = std.debug;
const mem = std.mem;
const Allocator = std.mem.Allocator;

const AutoSwissTable = @import("swiss_table").AutoSwissTable;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){ .backing_allocator = std.heap.page_allocator };
    defer debug.assert(gpa.deinit() == .ok);
    const allocator = gpa.allocator();

    var map = AutoSwissTable(u8, u32).init(allocator);
    defer map.deinit();

    try map.put_no_clobber('a', 1);
    try map.put_no_clobber('b', 2);
    try map.put_no_clobber('c', 3);
    try map.put_no_clobber('d', 4);
    std.debug.assert((try map.fetch_put('b', 5)).?.value == 2);
    std.debug.assert((try map.fetch_put('d', 6)).?.value == 4);

    std.debug.assert(map.get('a').? == 1);
    std.debug.assert(map.get('b').? == 5);
    std.debug.assert(map.get('c').? == 3);
    std.debug.assert(map.get('d').? == 6);
}
