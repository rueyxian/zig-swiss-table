const std = @import("std");
const builtin = @import("builtin");
const debug = std.debug;
const print = std.debug.print;
const testing = std.testing;
const mem = std.mem;
const math = std.math;
const meta = std.meta;
const Allocator = std.mem.Allocator;

fn verify_context(
    comptime RawContext: type,
    comptime PseudoKey: type,
    comptime Key: type,
    comptime Hash: type,
    comptime is_array: bool,
) void {
    _ = is_array;
    _ = Hash;
    _ = Key;
    _ = PseudoKey;
    _ = RawContext;
    //
}

pub const Processor = union(enum) {
    scalar: u8,
    vector: u8,
};

pub fn HashMapUnmanaged(
    comptime K: type,
    comptime V: type,
    comptime Context: type,
    comptime max_load_percentage: u64,
    comptime processor: Processor,
) type {
    if (max_load_percentage <= 0 or max_load_percentage >= 100) {
        @compileError("`max_load_percentage` must be between 0 and 100.");
    }
    switch (processor) {
        inline else => |n| if (@popCount(n) != 1) {
            @compileError("`processor` must be a power of 2.");
        },
    }

    comptime verify_context(Context, K, K, u64, false);

    return struct {
        const Self = @This();

        memory: ?[*]u8 = null,
        size: Size = 0,
        capacity: Size = 0,
        available: Size = 0,

        const MIN_CAPACITY = @min(8, Group.lane_count);

        pub const Size = u32;

        //              |   0x   |   0x   |   0x   |   0x   |   0x   |   0x   |   0x   |   0x   |
        // Hash       : <--------------------------------- u64 --------------------------------->
        // Fingerprint: <-- u7 ->
        // Hashcode   :         <---------------------------- u57 ------------------------------>
        pub const Hash = u64;
        const Fingerprint = u7;
        const Hashcode = u57;
        comptime {
            const hash_bits = @typeInfo(Hash).Int.bits;
            const fp_bits = @typeInfo(Fingerprint).Int.bits;
            const hc_bits = @typeInfo(Hashcode).Int.bits;
            debug.assert(hash_bits == fp_bits + hc_bits);
        }
        fn get_hashcode(hash: Hash) Hashcode {
            return @intCast(hash & math.maxInt(Hashcode));
        }
        fn get_fingerprint(hash: Hash) Fingerprint {
            return @intCast(hash >> @typeInfo(Hashcode).Int.bits);
        }

        const Header = struct {
            keys: [*]K,
            values: [*]V,
            // capacity: Size,
        };

        //              |  0b  |  0b  |  0b  |  0b  |  0b  |  0b  |  0b  |  0b  |
        // Metadata   : <-------------------------- u8 ------------------------->
        // Used       : <- u1 ->
        // Fingerprint:        <--------------------- u7 ----------------------->
        const Metadata = u8;
        const METADATA_FREE: Metadata = 0b1000000;
        const METADATA_TOMBSTONE: Metadata = 0b1000001;

        const Group = blk: {
            switch (processor) {
                .scalar => |n| break :blk ScalarGroup(n),
                .vector => |n| break :blk VectorGroup(n),
            }
            unreachable;
        };

        fn ScalarGroup(comptime n: comptime_int) type {
            debug.assert(@popCount(n) == 1);

            return struct {
                data: Data,

                const lane_count = n;
                const Data = meta.Int(.unsigned, lane_count);
                const Mask = meta.Int(.unsigned, lane_count);

                comptime {
                    debug.assert(lane_count == @bitSizeOf(Data));
                }

                inline fn from_metadata(metadata: [*]const Metadata) @This() {
                    _ = metadata;
                    // TODO
                    unreachable;
                }

                inline fn match_fingerprint(group: @This(), fingerprint: Fingerprint) Mask {
                    _ = fingerprint;
                    _ = group;
                    // TODO
                    unreachable;
                }

                inline fn match_free(group: @This()) Mask {
                    _ = group;
                    // TODO
                    unreachable;
                }

                inline fn match_unused(group: @This()) Mask {
                    _ = group;
                    // TODO
                    unreachable;
                }

                inline fn match_used(group: @This()) Mask {
                    _ = group;
                    // TODO
                    unreachable;
                }
            };
        }

        fn VectorGroup(comptime n: u8) type {
            debug.assert(@popCount(n) == 1);

            return struct {
                // TODO consider holding pointer instead. But then we have to some work on the alignment
                data: Data,

                const lane_count = n;
                const Data = @Vector(lane_count, Metadata);
                const Mask = meta.Int(.unsigned, lane_count);

                comptime {
                    debug.assert(lane_count == @sizeOf(Data));
                }

                inline fn from_metadata(metadata: [*]const Metadata) @This() {
                    var data: Data = @bitCast(metadata[0..lane_count].*);
                    if (builtin.cpu.arch.endian() == .Big) data = @byteSwap(data);
                    return @This(){ .data = data };
                }

                inline fn match_fingerprint(group: @This(), fingerprint: Fingerprint) Mask {
                    const truths = group.data == @as(Data, @splat(fingerprint));
                    return @bitCast(truths);
                }

                inline fn match_free(group: @This()) Mask {
                    const truths = group.data == @as(Data, @splat(METADATA_FREE));
                    return @bitCast(truths);
                }

                inline fn match_unused(group: @This()) Mask {
                    const truths = group.data >= @as(Data, @splat(METADATA_FREE));
                    return @bitCast(truths);
                }

                inline fn match_used(group: @This()) Mask {
                    const truths = group.data < @as(Data, @splat(METADATA_FREE));
                    return @bitCast(truths);
                }
            };
        }

        fn header(self: *Self) *Header {
            return @ptrCast(@alignCast(self.memory.?));
        }

        fn metadatas(self: *Self) ?[*]Metadata {
            const memory = self.memory orelse return null;
            return memory + @sizeOf(Header);
        }

        // const Metadata = packed struct {
        //     fingerprint: Fingerprint,
        //     used: u1,

        //     const FREE: Metadata = .{ .fingerprint = 0, .used = 0 };
        //     const TOMBSTONE: Metadata = .{ .fingerprint = 1, .used = 0 };

        //     fn take_fingerprint(hash: Hash) Metadata {
        //         const hash_bits = @typeInfo(Hash).Int.bits;
        //         const fp_bits = @typeInfo(Fingerprint).Int.bits;
        //         return .{ .fingerprint = @truncate(hash >> (hash_bits - fp_bits)), .used = 1 };
        //     }
        // };
        // comptime {
        //     debug.assert(@sizeOf(Metadata) == @sizeOf(u8)); // 1
        //     debug.assert(@alignOf(Metadata) == @alignOf(u8)); // 1
        // }

        // fn get_groups(self: *Self) [*]Group {
        //     return @ptrCast(@alignCast(self.memory.? + 1));
        // }

        fn keys(self: *Self) [*]K {
            return self.header().keys;
        }

        fn values(self: *Self) [*]V {
            return self.header().values;
        }

        // pub fn get_capacity(self: *Self) Size {
        //     return self.get_header().capacity;
        // }

        // fn get_group_count(self: *Self) Size {
        //     return self.get_capacity();
        // }

        // fn cap_for_size(size: Size) Size {
        //     var new_cap: u32 = @truncate((@as(u64, size) * 100) / max_load_percentage + 1);
        //     new_cap = math.ceilPowerOfTwo(u32, new_cap) catch unreachable;
        //     return new_cap;
        // }

        fn allocate(self: *Self, allocator: Allocator, capacity: Size) Allocator.Error!void {
            debug.assert(@popCount(capacity) == 1 and capacity >= MIN_CAPACITY);
            debug.assert(@sizeOf([*]K) != 0);
            debug.assert(@sizeOf([*]V) != 0);

            const metadata_start = mem.alignForward(usize, @sizeOf(Header), @alignOf(Metadata));
            const metadata_end = metadata_start + capacity * @sizeOf(Metadata);

            const keys_start = mem.alignForward(usize, metadata_end, @alignOf(K));
            const keys_end = keys_start + capacity * @sizeOf(K);

            const vals_start = mem.alignForward(usize, keys_end, @alignOf(V));
            const vals_end = vals_start + (capacity + @sizeOf(V));

            const max_align = comptime @max(@alignOf(Header), @alignOf(K), @alignOf(V));
            const total_size = mem.alignForward(usize, vals_end, max_align);

            const memory = try allocator.alignedAlloc(u8, max_align, total_size);
            const addr = @intFromPtr(memory.ptr);

            const hdr: *Header = @ptrFromInt(addr);
            hdr.keys = @ptrFromInt(addr + keys_start);
            hdr.values = @ptrFromInt(addr + vals_start);
            // hdr.capacity = capacity;

            self.memory = @ptrFromInt(addr);
            self.capacity = capacity;

            // self.init_metadata();
            // self.clear_retaining_capacity();
        }

        fn deallocate(self: *Self, allocator: Allocator) void {
            if (self.memory == null) return;
            const cap = self.capacity;
            debug.assert(@popCount(cap) == 1 and cap >= MIN_CAPACITY);

            const metadata_start = mem.alignForward(usize, @sizeOf(Header), @alignOf(Metadata));
            const metadata_end = metadata_start + cap * @sizeOf(Metadata);

            const keys_start = mem.alignForward(usize, metadata_end, @alignOf(K));
            const keys_end = keys_start + cap * @sizeOf(K);

            const vals_start = mem.alignForward(usize, keys_end, @alignOf(V));
            const vals_end = vals_start + (cap + @sizeOf(V));

            const max_align = comptime @max(@alignOf(Header), @alignOf(K), @alignOf(V));
            const total_size = mem.alignForward(usize, vals_end, max_align);

            const slice = @as([*]align(max_align) u8, @alignCast(self.memory.?))[0..total_size];
            allocator.free(slice);

            // self.memory = null;
            // self.capacity = 0;
            // self.available = 0;
        }

        fn init_metadata(self: *Self) void {
            const metadata = self.metadatas().?;
            var slice: []Metadata = @ptrCast(metadata[0..(@sizeOf(Metadata) * self.capacity)]);
            @memset(slice, METADATA_FREE);
        }

        inline fn get_index(self: Self, key: anytype, ctx: anytype) ?usize {
            comptime verify_context(@TypeOf(ctx, @TypeOf(key), K, Hash, false));
            if (self.size == 0) {
                return null;
            }

            const hash = ctx.hash(key);
            if (@TypeOf(hash) != Hash) {
                @compileError("Context " ++ @typeName(@TypeOf(ctx)) ++ " has a generic hash function that returns the wrong type! " ++ @typeName(Hash) ++ " was expected, but found " ++ @typeName(@TypeOf(hash)));
            }

            const hashcode: Hashcode = get_hashcode(hash);
            const fingerprint = get_fingerprint(hash);

            debug.assert(@popCount(self.capacity) == 1);
            const cap_mask = self.capacity - 1;
            var group_count = self.capacity / Group.lane_count;
            var aligned_idx: usize = mem.alignBackward(usize, hashcode & cap_mask, Group.lane_count);

            while (group_count != 0) : ({
                aligned_idx = (aligned_idx + Group.lane_count) & cap_mask;
                group_count -= 1;
            }) {
                const group = Group.from_metadata(self.metadatas().? + aligned_idx);
                var fp_matches = group.match_fingerprint(fingerprint);
                while (fp_matches != 0) : (fp_matches &= fp_matches - 1) {
                    const idx = aligned_idx + @ctz(fp_matches);
                    const key_ptr = &self.keys()[idx];
                    const eql = ctx.eql(key, key_ptr.*);
                    if (@TypeOf(eql) != bool) {
                        @compileError("Context " ++ @typeName(@TypeOf(ctx)) ++ " has a generic eql function that returns the wrong type! bool was expected, but found " ++ @typeName(@TypeOf(eql)));
                    }
                    if (!eql) continue;
                    return idx;
                }
                if (group.match_free() != 0) {
                    break;
                }
            }
            return null;
        }

        fn capacity_for_size(size: Size) Size {
            debug.assert(size >= MIN_CAPACITY);
            // TODO
            // In the stdlib hash_map: `var new_cap: Size = @truncate((@as(u64, size) * 100) / max_load_percentage + 1)`;
            // not sure if this could work. But it seems isn't as optimal compare with just `+ 1`
            var new_cap: Size = @intCast(math.divCeil(u64, @as(u4, size * 100), max_load_percentage));
            return math.ceilPowerOfTwoAssert(Size, new_cap);
        }

        fn grow_if_needed(self: *Self, allocator: Allocator, required_additional: Size, ctx: Context) Allocator.Error!void {
            if (required_additional > self.available) {
                // NOTE: The stdlib hash_map derives `load` from `max_load_percentage` and `self.available`.
                // If not mistaken, this is uncessary. Because `load` basically means `self.size`
                const new_size = self.size + required_additional;
                const new_cap = self.capacity_for_size(new_size);
                debug.assert(self.capacity != new_cap);
                try self.grow(allocator, new_cap, ctx);
            }
        }

        fn grow(self: *Self, allocator: Allocator, new_capacity: Size, ctx: Context) Allocator.Error!void {
            @setCold(true);
            const new_cap = @max(new_capacity, MIN_CAPACITY);
            debug.assert(new_cap > self.capacity);
            debug.assert(@popCount(new_cap) == 1);

            var map = Self{};
            defer map.deinit();
            try map.allocate(allocator, new_capacity);
            map.init_metadata();

            if (self.size != 0) {
                copy_and_rehash(&map, self, ctx);
            }
            mem.swap(Self, self, &map);
        }

        fn copy_and_rehash(noalias dst: *Self, noalias src: *Self, ctx: anytype) void {
            debug.assert(dst.size >= @as(Size, @intCast((src.capacity * max_load_percentage) / 100)));
            defer debug.assert(dst.size == src.size);

            var aligned_idx: usize = 0;
            var group_count = src.capacity / Group.lane_count;
            while (group_count != 0) : ({
                aligned_idx += Group.lane_count;
                group_count -= 1;
            }) {
                const group = Group.from_metadata(src.metadatas().? + aligned_idx);
                var used_matches = group.match_used();
                while (used_matches != 0) : (used_matches &= used_matches - 1) {
                    const idx = aligned_idx + @ctz(used_matches);
                    const key = src.keys()[idx];
                    const val = src.values()[idx];
                    dst.put_assume_capacity_no_clobber_context(key, val, ctx);
                }
            }
        }

        pub const Entry = struct {
            key_ptr: *K,
            value_ptr: *V,
        };

        pub const KV = struct {
            key: K,
            value: V,
        };

        pub const GetOrPutResult = struct {
            key_ptr: *K,
            value_ptr: *V,
            found_existing: bool,
        };

        pub fn deinit(self: *Self, allocator: Allocator) void {
            self.deallocate(allocator);
        }

        pub fn clear_retaining_capacity(self: *Self) void {
            if (self.metadatas() == null) return;
            self.init_metadata();
            self.size = 0;
            self.available = @intCast((self.capacity * max_load_percentage) / 100);
        }

        pub fn clear_and_free(self: *Self, allocator: Allocator) void {
            self.deallocate(allocator);
        }

        pub fn count(self: *const Self) Size {
            return self.size;
        }

        pub fn contains(self: *const Self, key: K) bool {
            return self.contains_context(key, undefined);
        }

        pub fn contains_context(self: *const Self, key: K, ctx: Context) bool {
            return self.contains_adapted(key, ctx);
        }

        pub fn contains_adapted(self: *const Self, key: anytype, ctx: anytype) bool {
            return self.get_index(key, ctx) != null;
        }

        pub fn put_no_clobber(self: *Self, allocator: Allocator, key: K, value: K) Allocator.Error!void {
            if (@sizeOf(Context) != 0) {
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call put_no_clobber_context instead.");
            }
            self.put_no_clobber_context(allocator, key, value, undefined);
        }

        pub fn put_no_clobber_context(self: *Self, allocator: Allocator, key: K, value: K, ctx: Context) Allocator.Error!void {
            debug.assert(!self.contains_context(key, ctx));
            try self.grow_if_needed(allocator, 1, ctx);

            self.put_assume_capacity_no_clobber_context(key, value, ctx);
        }

        pub fn put_assume_capacity(self: *Self, key: K, value: V) void {
            if (@sizeOf(Context) != 0)
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call put_assume_capacity_context instead.");
            return self.put_assume_capacity_context(key, value, undefined);
        }

        pub fn put_assume_capacity_context(self: *Self, key: K, value: V, ctx: Context) void {
            const gop = self.get_or_put_assume_capacity_context(key, ctx);
            gop.value_ptr.* = value;
        }

        pub fn put_assume_capacity_no_clobber(self: *Self, key: K, value: V) void {
            if (@sizeOf(Context) != 0) {
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call put_assume_capacity_no_clobber_context instead.");
            }
            self.put_assume_capacity_no_clobber_context(key, value, undefined);
        }

        pub fn put_assume_capacity_no_clobber_context(self: *Self, key: K, value: V, ctx: Context) void {
            debug.assert(!self.contains_context(key, ctx));
            const hash = ctx.hash(key);
            const hash_code = get_hashcode(hash);
            const cap_mask = self.capacity - 1;

            var aligned_idx = mem.alignBackward(usize, hash_code & cap_mask, Group.lane_count);

            const idx = while (true) : ({
                aligned_idx = (aligned_idx + Group.lane_count) & cap_mask;
            }) {
                const group = Group.from_metadata(self.metadatas().? + aligned_idx);
                const unused_matches = group.match_unused();
                if (unused_matches == 0) continue;
                break aligned_idx + @ctz(unused_matches);
            };
            debug.assert(self.available > 0);
            self.metadatas().?[idx] = get_fingerprint(hash);
            self.keys()[idx] = key;
            self.values()[idx] = value;
            self.available -= 1;
            self.size += 1;
        }

        pub fn get_or_put_assume_capacity(self: *Self, key: K) GetOrPutResult {
            if (@sizeOf(Context) != 0) {
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call `get_or_put_assume_capacity_context` instead.");
            }
            return self.get_or_put_assume_capacity_context(key, undefined);
        }

        pub fn get_or_put_assume_capacity_context(self: *Self, key: K, ctx: Context) GetOrPutResult {
            var gop = self.get_or_put_assume_capacity_adapted(key, ctx);
            if (!gop.found_existing) {
                gop.key_ptr.* = key;
            }
            return gop;
        }

        pub fn get_or_put_assume_capacity_adapted(self: *Self, key: anytype, ctx: anytype) GetOrPutResult {
            comptime verify_context(@TypeOf(ctx), @TypeOf(key), K, Hash, false);

            const hash = ctx.hash(key);
            if (@TypeOf(hash) != Hash) {
                @compileError("Context " ++ @typeName(@TypeOf(ctx)) ++ " has a generic hash function that returns the wrong type! " ++ @typeName(Hash) ++ " was expected, but found " ++ @typeName(@TypeOf(hash)));
            }

            const hashcode: Hashcode = get_hashcode(hash);
            const fingerprint: Fingerprint = get_fingerprint(hash);

            debug.assert(@popCount(self.capacity) == 1);
            const cap_mask = self.capacity - 1;
            var group_count = self.capacity / Group.lane_count;
            var aligned_idx: usize = mem.alignBackward(usize, hashcode & cap_mask, Group.lane_count);
            var opt_idx: ?usize = null;

            const idx = while (group_count != 0) : ({
                aligned_idx = (aligned_idx + Group.lane_count) & cap_mask;
                group_count -= 1;
            }) {
                const group = Group.from_metadata(self.metadatas().? + aligned_idx);

                print("group: {any}\n", .{group.data});
                var fp_matches = group.match_fingerprint(fingerprint);
                while (fp_matches != 0) : (fp_matches &= fp_matches - 1) {
                    const idx = aligned_idx + @ctz(fp_matches);
                    const key_ptr = &self.keys()[idx];
                    const eql = ctx.eql(key, key_ptr.*);
                    if (@TypeOf(eql) != bool) {
                        @compileError("Context " ++ @typeName(@TypeOf(ctx)) ++ " has a generic eql function that returns the wrong type! bool was expected, but found " ++ @typeName(@TypeOf(eql)));
                    }
                    if (!eql) continue;
                    return GetOrPutResult{
                        .key_ptr = key_ptr,
                        .value_ptr = &self.values()[idx],
                        .found_existing = true,
                    };
                }
                if (opt_idx == null) blk: {
                    const unused_matches = group.match_unused();
                    if (unused_matches == 0) break :blk;
                    opt_idx = aligned_idx + @ctz(unused_matches);
                }
                if (group.match_free() != 0) {
                    break opt_idx.?;
                }
            } else unreachable;

            debug.assert(self.available > 0);

            self.size += 1;
            self.available -= 1;
            self.metadatas().?[idx] = fingerprint;

            const key_ptr = &self.keys()[idx];
            const value_ptr = &self.values()[idx];
            key_ptr.* = undefined;
            value_ptr.* = undefined;

            return GetOrPutResult{
                .key_ptr = key_ptr,
                .value_ptr = value_ptr,
                .found_existing = false,
            };
        }
    };
}

test "draft" {
    // if (true) return error.SkipZigTest;

    //
    const allocator = testing.allocator;
    // const allocator = std.heap.page_allocator;

    const Context = struct {
        pub fn hash(self: @This(), s: []const u8) u64 {
            _ = self;
            return std.hash.Wyhash.hash(0, s);
        }
        pub fn eql(self: @This(), a: []const u8, b: []const u8) bool {
            _ = self;
            return mem.eql(u8, a, b);
        }
    };
    // var hm = HashMapUnmanaged([]const u8, u32, Context, 80){};
    // var hm = HashMapUnmanaged(u128, u32, Context, 80){};
    const HM = HashMapUnmanaged([]const u8, u32, Context, 80, .{ .vector = 16 });
    // _ = HM;

    var hm = HM{};
    // var hm = HashMapUnmanaged(u8, u16, Context, 80){};
    defer hm.deinit(allocator);

    // try hm.allocate(allocator, 16);
    try hm.allocate(allocator, 64);
    // hm.init_metadata();
    hm.clear_retaining_capacity();

    // print("cap: {any}\n", .{HM.cap_for_size(0)});
    // print("Mask: {any}\n", .{HM.Scan.Mask});

    {
        var gop = hm.get_or_put_assume_capacity_adapted("hello", Context{});
        gop.key_ptr.* = "hello";
        print("found_existing: {}\n", .{gop.found_existing});
    }

    {
        var gop = hm.get_or_put_assume_capacity_adapted("hello", Context{});
        print("found_existing: {}\n", .{gop.found_existing});
    }

    // print("@sizeOf([]const u8): {any}\n", .{@sizeOf([]const u8)});

    // print("header: {any}\n", .{hm.get_header()});
    // print("cap   : {any}\n", .{hm.get_capacity()});
    // print("metadata: {any}\n", .{hm.get_metadata()});

    // const groups = hm.get_groups();
    // print("groups: {any}\n", .{groups});

    {
        const addr = @intFromPtr(hm.metadatas().?);
        print("addr: {}\n", .{addr});
        print("is align 16: {}\n", .{addr % 16 == 0});
        print("is align 32: {}\n", .{addr % 32 == 0});

        // const Group = @Vector(16, u8);

        // var metadata = hm.get_metadata();
        // var addr = @intFromPtr(metadata);

        // print("is div 16: {}\n", .{addr % 16});

        // var g_ptr: *Group = @ptrFromInt(addr + 8);
        // _ = g_ptr;
    }
}
