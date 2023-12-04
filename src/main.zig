const std = @import("std");
const builtin = @import("builtin");
const debug = std.debug;
const print = std.debug.print;
const mem = std.mem;
const math = std.math;
const meta = std.meta;
const Allocator = std.mem.Allocator;

pub const default_max_load_percentage = std.hash_map.default_max_load_percentage;

// TODO: I have no idea what is the optimal choice for each cpu architecture
pub const default_processor: Processor = .{ .vector = 16 };

pub const Processor = union(enum) {
    scalar: u8,
    vector: u8,
};

pub fn AutoHashMap(comptime K: type, comptime V: type) type {
    return HashMap(K, V, AutoContext(K), default_max_load_percentage, default_processor);
}

pub fn AutoHashMapUnmanaged(comptime K: type, comptime V: type) type {
    return HashMapUnmanaged(K, V, AutoContext(K), default_max_load_percentage, default_processor);
}

pub const AutoContext = std.hash_map.AutoContext;

pub fn StringHashMap(comptime V: type) type {
    return HashMap([]const u8, V, StringContext, default_max_load_percentage, default_processor);
}

pub fn StringHashMapUnmanaged(comptime V: type) type {
    return HashMapUnmanaged([]const u8, V, StringContext, default_max_load_percentage, default_processor);
}

pub const StringContext = std.hash_map.StringContext;

pub const eqlString = std.hash_map.eqlString;

pub const hashString = std.hash_map.hashString;

pub const StringIndexContext = std.hash_map.StringIndexContext;

pub const StringIndexAdapter = std.hash_map.StringIndexAdapter;

// pub const verifyContext = std.hash_map.verifyContext;

pub fn verifyContext(
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
}

pub fn HashMap(
    comptime K: type,
    comptime V: type,
    comptime Context: type,
    comptime max_load_percentage: u64,
    comptime processor: Processor,
) type {
    return struct {
        unmanaged: Unmanaged,
        allocator: Allocator,
        ctx: Context,

        const Self = @This();
        pub const Unmanaged = HashMapUnmanaged(K, V, Context, max_load_percentage, processor);

        pub const Size = Unmanaged.Size;
        pub const Hash = Unmanaged.Hash;
        pub const Entry = Unmanaged.Entry;
        pub const KV = Unmanaged.KV;
        pub const GetOrPutResult = Unmanaged.GetOrPutResult;
        pub const Iterator = Unmanaged.Iterator;
        pub const KeyIterator = Unmanaged.KeyIterator;
        pub const ValueIterator = Unmanaged.ValueIterator;

        /// Create a managed hash map with an empty context.
        /// If the context is not zero-sized, you must use
        /// initContext(allocator, ctx) instead.
        pub fn init(allocator: Allocator) Self {
            if (@sizeOf(Context) != 0) {
                @compileError("Context must be specified! Call `init_context` instead.");
            }
            return initContext(allocator, undefined);
        }

        /// Create a managed hash map with a context
        pub fn initContext(allocator: Allocator, ctx: Context) Self {
            return Self{
                .unmanaged = .{},
                .allocator = allocator,
                .ctx = ctx,
            };
        }

        /// Release the backing array and invalidate this map.
        /// This does *not* deinit keys, values, or the context!
        /// If your keys or values need to be released, ensure
        /// that that is done before calling this function.
        pub fn deinit(self: *Self) void {
            self.unmanaged.deinit(self.allocator);
            self.* = undefined;
        }

        /// Empty the map, but keep the backing allocation for future use.
        /// This does *not* free keys or values! Be sure to
        /// release them if they need deinitialization before
        /// calling this function.
        pub fn clearRetainingCapacity(self: *Self) void {
            return self.unmanaged.clearRetainingCapacity();
        }

        /// Empty the map and release the backing allocation.
        /// This does *not* free keys or values! Be sure to
        /// release them if they need deinitialization before
        /// calling this function.
        pub fn clearAndFree(self: *Self) void {
            return self.unmanaged.clearAndFree(self.allocator);
        }

        /// Return the number of items in the map.
        pub fn count(self: Self) Size {
            return self.unmanaged.count();
        }

        /// Create an iterator over the entries in the map.
        /// The iterator is invalidated if the map is modified.
        pub fn iterator(self: *const Self) Iterator {
            return self.unmanaged.iterator();
        }

        /// Create an iterator over the keys in the map.
        /// The iterator is invalidated if the map is modified.
        pub fn keyIterator(self: *const Self) KeyIterator {
            return self.unmanaged.keyIterator();
        }

        /// Create an iterator over the values in the map.
        /// The iterator is invalidated if the map is modified.
        pub fn valueIterator(self: *const Self) ValueIterator {
            return self.unmanaged.valueIterator();
        }

        /// If key exists this function cannot fail.
        /// If there is an existing item with `key`, then the result
        /// `Entry` pointers point to it, and found_existing is true.
        /// Otherwise, puts a new item with undefined value, and
        /// the `Entry` pointers point to it. Caller should then initialize
        /// the value (but not the key).
        pub fn getOrPut(self: *Self, key: K) Allocator.Error!GetOrPutResult {
            return self.unmanaged.getOrPutContext(self.allocator, key, self.ctx);
        }

        /// If key exists this function cannot fail.
        /// If there is an existing item with `key`, then the result
        /// `Entry` pointers point to it, and found_existing is true.
        /// Otherwise, puts a new item with undefined key and value, and
        /// the `Entry` pointers point to it. Caller must then initialize
        /// the key and value.
        pub fn getOrPutAdapted(self: *Self, key: anytype, ctx: anytype) Allocator.Error!GetOrPutResult {
            return self.unmanaged.getOrPutContextAdapted(self.allocator, key, ctx, self.ctx);
        }

        /// If there is an existing item with `key`, then the result
        /// `Entry` pointers point to it, and found_existing is true.
        /// Otherwise, puts a new item with undefined value, and
        /// the `Entry` pointers point to it. Caller should then initialize
        /// the value (but not the key).
        /// If a new entry needs to be stored, this function asserts there
        /// is enough capacity to store it.
        pub fn getOrPutAssumeCapacity(self: *Self, key: K) GetOrPutResult {
            return self.unmanaged.getOrPutAssumeCapacityContext(key, self.ctx);
        }

        /// If there is an existing item with `key`, then the result
        /// `Entry` pointers point to it, and found_existing is true.
        /// Otherwise, puts a new item with undefined value, and
        /// the `Entry` pointers point to it. Caller must then initialize
        /// the key and value.
        /// If a new entry needs to be stored, this function asserts there
        /// is enough capacity to store it.
        pub fn getOrPutAssumeCapacityAdapted(self: *Self, key: anytype, ctx: anytype) GetOrPutResult {
            return self.unmanaged.getOrPutAssumeCapacityAdapted(key, ctx);
        }

        pub fn getOrPutValue(self: *Self, key: K, value: V) Allocator.Error!Entry {
            return self.unmanaged.getOrPutValueContext(self.allocator, key, value, self.ctx);
        }

        /// Increases capacity, guaranteeing that insertions up until the
        /// `expected_count` will not cause an allocation, and therefore cannot fail.
        pub fn ensureTotalCapacity(self: *Self, expected_count: Size) Allocator.Error!void {
            return self.unmanaged.ensureTotalCapacityContext(self.allocator, expected_count, self.ctx);
        }

        /// Increases capacity, guaranteeing that insertions up until
        /// `additional_count` **more** items will not cause an allocation, and
        /// therefore cannot fail.
        pub fn ensureUnusedCapacity(self: *Self, additional_count: Size) Allocator.Error!void {
            return self.unmanaged.ensureUnusedCapacityContext(self.allocator, additional_count, self.ctx);
        }

        /// Returns the number of total elements which may be present before it is
        /// no longer guaranteed that no allocations will be performed.
        pub fn capacity(self: *Self) Size {
            return self.unmanaged.capacity();
        }

        /// Clobbers any existing data. To detect if a put would clobber
        /// existing data, see `getOrPut`.
        pub fn put(self: *Self, key: K, value: V) Allocator.Error!void {
            return self.unmanaged.putContext(self.allocator, key, value, self.ctx);
        }

        /// Inserts a key-value pair into the hash map, asserting that no previous
        /// entry with the same key is already present
        pub fn putNoClobber(self: *Self, key: K, value: V) Allocator.Error!void {
            return self.unmanaged.putNoClobberContext(self.allocator, key, value, self.ctx);
        }

        /// Asserts there is enough capacity to store the new key-value pair.
        /// Clobbers any existing data. To detect if a put would clobber
        /// existing data, see `getOrPutAssumeCapacity`.
        pub fn putAssumeCapacity(self: *Self, key: K, value: V) void {
            return self.unmanaged.putAssumeCapacityContext(key, value, self.ctx);
        }

        /// Asserts there is enough capacity to store the new key-value pair.
        /// Asserts that it does not clobber any existing data.
        /// To detect if a put would clobber existing data, see `getOrPutAssumeCapacity`.
        pub fn putAssumeCapacityNoClobber(self: *Self, key: K, value: V) void {
            return self.unmanaged.putAssumeCapacityNoClobberContext(key, value, self.ctx);
        }

        /// Inserts a new `Entry` into the hash map, returning the previous one, if any.
        pub fn fetchPut(self: *Self, key: K, value: V) Allocator.Error!?KV {
            return self.unmanaged.fetchPutContext(self.allocator, key, value, self.ctx);
        }

        /// Inserts a new `Entry` into the hash map, returning the previous one, if any.
        /// If insertion happuns, asserts there is enough capacity without allocating.
        pub fn fetchPutAssumeCapacity(self: *Self, key: K, value: V) ?KV {
            return self.unmanaged.fetchPutAssumeCapacityContext(key, value, self.ctx);
        }

        /// Removes a value from the map and returns the removed kv pair.
        pub fn fetchRemove(self: *Self, key: K) ?KV {
            return self.unmanaged.fetchRemoveContext(key, self.ctx);
        }

        pub fn fetchRemoveAdapted(self: *Self, key: anytype, ctx: anytype) ?KV {
            return self.unmanaged.fetchRemoveAdapted(key, ctx);
        }

        /// Finds the value associated with a key in the map
        pub fn get(self: Self, key: K) ?V {
            return self.unmanaged.getContext(key, self.ctx);
        }
        pub fn getAdapted(self: Self, key: anytype, ctx: anytype) ?V {
            return self.unmanaged.getAdapted(key, ctx);
        }

        pub fn getPtr(self: Self, key: K) ?*V {
            return self.unmanaged.getPtrContext(key, self.ctx);
        }
        pub fn getPtrAdapted(self: Self, key: anytype, ctx: anytype) ?*V {
            return self.unmanaged.getPtrAdapted(key, ctx);
        }

        /// Finds the actual key associated with an adapted key in the map
        pub fn getKey(self: Self, key: K) ?K {
            return self.unmanaged.getKeyContext(key, self.ctx);
        }
        pub fn getKeyAdapted(self: Self, key: anytype, ctx: anytype) ?K {
            return self.unmanaged.getKeyAdapted(key, ctx);
        }

        pub fn getKeyPtr(self: Self, key: K) ?*K {
            return self.unmanaged.getKeyPtrContext(key, self.ctx);
        }
        pub fn getKeyPtrAdapted(self: Self, key: anytype, ctx: anytype) ?*K {
            return self.unmanaged.getKeyPtrAdapted(key, ctx);
        }

        /// Finds the key and value associated with a key in the map
        pub fn getEntry(self: Self, key: K) ?Entry {
            return self.unmanaged.getEntryContext(key, self.ctx);
        }

        pub fn getEntryAdapted(self: Self, key: anytype, ctx: anytype) ?Entry {
            return self.unmanaged.getEntryAdapted(key, ctx);
        }

        /// Check if the map contains a key
        pub fn contains(self: Self, key: K) bool {
            return self.unmanaged.containsContext(key, self.ctx);
        }

        pub fn containsAdapted(self: Self, key: anytype, ctx: anytype) bool {
            return self.unmanaged.containsAdapted(key, ctx);
        }

        /// If there is an `Entry` with a matching key, it is deleted from
        /// the hash map, and this function returns true.  Otherwise this
        /// function returns false.
        pub fn remove(self: *Self, key: K) bool {
            return self.unmanaged.removeContext(key, self.ctx);
        }

        pub fn removeAdapted(self: *Self, key: anytype, ctx: anytype) bool {
            return self.unmanaged.removeAdapted(key, ctx);
        }

        /// Delete the entry with key pointed to by key_ptr from the hash map.
        /// key_ptr is assumed to be a valid pointer to a key that is present
        /// in the hash map.
        pub fn removeByPtr(self: *Self, key_ptr: *K) void {
            self.unmanaged.removeByPtr(key_ptr);
        }

        /// Creates a copy of this map, using the same allocator
        pub fn clone(self: Self) Allocator.Error!Self {
            var other = try self.unmanaged.cloneContext(self.allocator, self.ctx);
            return other.promoteContext(self.allocator, self.ctx);
        }

        /// Creates a copy of this map, using a specified allocator
        pub fn cloneWithAllocator(self: Self, new_allocator: Allocator) Allocator.Error!Self {
            var other = try self.unmanaged.cloneContext(new_allocator, self.ctx);
            return other.promoteContext(new_allocator, self.ctx);
        }

        /// Creates a copy of this map, using a specified context
        pub fn cloneWithContext(self: Self, new_ctx: anytype) Allocator.Error!HashMap(K, V, @TypeOf(new_ctx), max_load_percentage) {
            var other = try self.unmanaged.cloneContext(self.allocator, new_ctx);
            return other.promoteContext(self.allocator, new_ctx);
        }

        /// Creates a copy of this map, using a specified allocator and context.
        pub fn cloneWithAllocatorAndContext(
            self: Self,
            new_allocator: Allocator,
            new_ctx: anytype,
        ) Allocator.Error!HashMap(K, V, @TypeOf(new_ctx), max_load_percentage) {
            var other = try self.unmanaged.cloneContext(new_allocator, new_ctx);
            return other.promoteContext(new_allocator, new_ctx);
        }

        /// Set the map to an empty state, making deinitialization a no-op, and
        /// returning a copy of the original.
        pub fn move(self: *Self) Self {
            const result = self.*;
            self.unmanaged = .{};
            return result;
        }

        fn __debug_print_metadata(self: *const Self) void {
            self.unmanaged._debug_print_metadata();
        }
    };
}

pub fn HashMapUnmanaged(
    comptime K: type,
    comptime V: type,
    comptime Context: type,
    comptime max_load_percentage: u64,
    comptime processor: Processor,
) type {
    comptime verifyContext(Context, K, K, u64, false);
    if (max_load_percentage <= 0 or max_load_percentage >= 100) {
        @compileError("`max_load_percentage` must be between 0 and 100.");
    }
    switch (processor) {
        inline else => |n| if (@popCount(n) != 1) {
            @compileError("`processor` must be a power of 2.");
        },
    }

    return struct {
        metadata: ?[*]u8 = null,
        size: Size = 0,
        available: Size = 0,

        const Self = @This();
        const Managed = HashMap(K, V, Context, max_load_percentage, processor);

        // TODO
        // const MIN_CAPACITY = 8;
        const minimal_capacity = @max(8, Group.lane_count);

        pub const Size = u32;

        // Bytes      : |   0x   |   0x   |   0x   |   0x   |   0x   |   0x   |   0x   |   0x   |
        // Hash       : <--------------------------------- 64 ---------------------------------->
        // Fingerprint: <-- 7 -->
        // Hashcode   :         <---------------------------- 57 ------------------------------->
        pub const Hash = u64;
        const Fingerprint = u7;
        const Hashcode = u57;
        comptime {
            const hash_bits = @typeInfo(Hash).Int.bits;
            const fp_bits = @typeInfo(Fingerprint).Int.bits;
            const hc_bits = @typeInfo(Hashcode).Int.bits;
            debug.assert(hash_bits == fp_bits + hc_bits);
        }
        fn getHashcode(hash: Hash) Hashcode {
            return @intCast(hash & math.maxInt(Hashcode));
        }
        fn getFingerprint(hash: Hash) Fingerprint {
            return @intCast(hash >> @typeInfo(Hashcode).Int.bits);
        }

        const Header = struct {
            keys: [*]K,
            values: [*]V,
            capacity: Size,
        };

        fn header(self: *const Self) *Header {
            return @ptrCast(@as([*]Header, @ptrCast(@alignCast(self.metadata.?))) - 1);
        }

        fn keys(self: *const Self) [*]K {
            return self.header().keys;
        }

        fn values(self: *const Self) [*]V {
            return self.header().values;
        }

        fn capacity(self: *const Self) Size {
            if (self.metadata == null) return 0;
            return self.header().capacity;
        }

        // Bits               : |  0b |  0b |  0b |  0b |  0b |  0b |  0b |  0b |
        // Metadata           : <--------------------- 8 ----------------------->
        // Unused flag        : <- 1 ->
        // Fingerprint        :       <------------------- 7 ------------------->
        //
        // Free (unused)      : |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
        // Tombstone (unused) : |  1  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |
        // Sentinel (unused)  : |  1  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |
        // Used               : |  0  |  X  |  X  |  X  |  X  |  X  |  X  |  X  |
        const Metadata = u8;
        const METADATA_FREE: Metadata = 0b11111111;
        const METADATA_TOMBSTONE: Metadata = 0b10000001;
        const METADATA_SENTINEL: Metadata = 0b10000000;

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
                const MatchMask = meta.Int(.unsigned, lane_count);

                comptime {
                    debug.assert(lane_count == @sizeOf(Data));
                }

                inline fn fromMetadata(metadata: [*]const Metadata) @This() {
                    _ = metadata;
                    // TODO
                    unreachable;
                }

                inline fn matchFingerprint(group: @This(), fingerprint: Fingerprint) MatchMask {
                    _ = fingerprint;
                    _ = group;
                    // TODO
                    unreachable;
                }

                inline fn matchFree(group: @This()) MatchMask {
                    _ = group;
                    // TODO
                    unreachable;
                }

                inline fn matchUnused(group: @This()) MatchMask {
                    _ = group;
                    // TODO
                    unreachable;
                }

                inline fn matchUsed(group: @This()) MatchMask {
                    _ = group;
                    // TODO
                    unreachable;
                }
            };
        }

        fn VectorGroup(comptime n: u8) type {
            debug.assert(@popCount(n) == 1);

            return struct {
                data: Data,

                const lane_count = n;
                const Data = @Vector(lane_count, Metadata);
                const MatchMask = meta.Int(.unsigned, lane_count);

                comptime {
                    debug.assert(lane_count == @sizeOf(Data));
                    debug.assert(lane_count == @alignOf(Data));
                }

                inline fn fromMetadata(metadata: [*]const Metadata) @This() {
                    var data: Data = @bitCast(metadata[0..lane_count].*);
                    if (builtin.cpu.arch.endian() == .Big) data = @byteSwap(data);
                    return @This(){ .data = data };
                }

                inline fn matchFingerprint(group: @This(), fingerprint: Fingerprint) MatchMask {
                    const truths = group.data == @as(Data, @splat(fingerprint));
                    return @bitCast(truths);
                }

                inline fn matchFree(group: @This()) MatchMask {
                    const truths = group.data == @as(Data, @splat(METADATA_FREE));
                    return @bitCast(truths);
                }

                inline fn matchUnused(group: @This()) MatchMask {
                    const truths = group.data > @as(Data, @splat(METADATA_SENTINEL));
                    return @bitCast(truths);
                }

                inline fn matchUsed(group: @This()) MatchMask {
                    const truths = group.data < @as(Data, @splat(METADATA_SENTINEL));
                    return @bitCast(truths);
                }
            };
        }

        fn allocate(self: *Self, allocator: Allocator, new_capacity: Size) Allocator.Error!void {
            debug.assert(@popCount(new_capacity) == 1 and new_capacity >= minimal_capacity);
            debug.assert(@sizeOf([*]K) != 0);
            debug.assert(@sizeOf([*]V) != 0);

            const key_align = if (@sizeOf(K) == 0) 1 else @alignOf(K);
            const val_align = if (@sizeOf(V) == 0) 1 else @alignOf(V);
            const max_align = comptime @max(@alignOf(Header), key_align, val_align);

            const sentinel_count = Group.lane_count;

            const metadata_start = mem.alignForward(usize, @sizeOf(Header), @alignOf(Metadata));
            const metadata_end = metadata_start + ((new_capacity + sentinel_count) * @sizeOf(Metadata));

            const keys_start = mem.alignForward(usize, metadata_end, key_align);
            const keys_end = keys_start + (new_capacity * @sizeOf(K));

            const vals_start = mem.alignForward(usize, keys_end, val_align);
            const vals_end = vals_start + (new_capacity * @sizeOf(V));

            const total_size = mem.alignForward(usize, vals_end, max_align);

            const memory = try allocator.alignedAlloc(u8, max_align, total_size);
            const addr = @intFromPtr(memory.ptr);

            const hdr: *Header = @ptrFromInt(addr);
            hdr.keys = @ptrFromInt(addr + keys_start);
            hdr.values = @ptrFromInt(addr + vals_start);
            hdr.capacity = new_capacity;

            self.metadata = @ptrFromInt(addr + @sizeOf(Header));
        }

        fn deallocate(self: *Self, allocator: Allocator) void {
            if (self.metadata == null) return;
            const cap = self.capacity();
            debug.assert(@popCount(cap) == 1 and cap >= minimal_capacity);

            const key_align = if (@sizeOf(K) == 0) 1 else @alignOf(K);
            const val_align = if (@sizeOf(V) == 0) 1 else @alignOf(V);
            const max_align = comptime @max(@alignOf(Header), key_align, val_align);

            const sentinel_count = Group.lane_count;

            const metadata_start = mem.alignForward(usize, @sizeOf(Header), @alignOf(Metadata));
            const metadata_end = metadata_start + ((cap + sentinel_count) * @sizeOf(Metadata));

            const keys_start = mem.alignForward(usize, metadata_end, key_align);
            const keys_end = keys_start + cap * @sizeOf(K);

            const vals_start = mem.alignForward(usize, keys_end, val_align);
            const vals_end = vals_start + (cap * @sizeOf(V));

            const total_size = mem.alignForward(usize, vals_end, max_align);

            const slice = @as([*]align(max_align) u8, @ptrFromInt(@intFromPtr(self.header())))[0..total_size];
            allocator.free(slice);

            // TODO: `std.hash_map` does this too. Not sure if this is necessary.
            self.metadata = null;
            self.available = 0;
        }

        fn initMetadatas(self: *Self) void {
            const metadata = self.metadata.?;
            const free_end = @sizeOf(Metadata) * self.capacity();
            const sentinel_end = @sizeOf(Metadata) * (self.capacity() + Group.lane_count);
            var slice: []Metadata = @ptrCast(metadata[0..free_end]);
            @memset(slice, METADATA_FREE);
            slice = @ptrCast(metadata[free_end..sentinel_end]);
            @memset(slice, METADATA_SENTINEL);
        }

        inline fn getIndex(self: Self, key: anytype, ctx: anytype) ?usize {
            comptime verifyContext(@TypeOf(ctx), @TypeOf(key), K, Hash, false);
            if (self.size == 0) {
                return null;
            }

            const hash = ctx.hash(key);
            if (@TypeOf(hash) != Hash) {
                @compileError("Context " ++ @typeName(@TypeOf(ctx)) ++ " has a generic hash function that returns the wrong type! " ++ @typeName(Hash) ++ " was expected, but found " ++ @typeName(@TypeOf(hash)));
            }

            const hashcode: Hashcode = getHashcode(hash);
            const fingerprint = getFingerprint(hash);

            debug.assert(@popCount(self.capacity()) == 1);
            var group_count = (self.capacity() / Group.lane_count) + 1;
            const cap_mask = self.capacity() - 1;
            var base_idx: usize = hashcode & cap_mask;

            while (group_count != 0) : ({
                base_idx = (base_idx + Group.lane_count) % (self.capacity() + Group.lane_count);
                base_idx *= @intFromBool(base_idx < self.capacity());
                group_count -= 1;
            }) {
                const group = Group.fromMetadata(self.metadata.? + base_idx);
                var fp_matches = group.matchFingerprint(fingerprint);
                while (fp_matches != 0) : (fp_matches &= fp_matches - 1) {
                    const idx = base_idx + @ctz(fp_matches);
                    const key_ptr = &self.keys()[idx];
                    const eql = ctx.eql(key, key_ptr.*);
                    if (@TypeOf(eql) != bool) {
                        @compileError("Context " ++ @typeName(@TypeOf(ctx)) ++ " has a generic eql function that returns the wrong type! bool was expected, but found " ++ @typeName(@TypeOf(eql)));
                    }
                    if (!eql) continue;
                    return idx;
                }
                if (group.matchFree() != 0) {
                    break;
                }
            }
            return null;
        }

        fn removeByIndex(self: *Self, index: usize) void {
            self.metadata.?[index] = METADATA_TOMBSTONE;
            self.keys()[index] = undefined;
            self.values()[index] = undefined;
            self.size -= 1;
            self.available += 1;
        }

        fn capacityForSize(size: Size) Size {
            var new_cap: Size = @intCast(((@as(u64, size) * 100) / max_load_percentage) + 1);
            new_cap = math.ceilPowerOfTwoAssert(Size, new_cap);
            return @max(new_cap, minimal_capacity);
        }

        fn growIfNeeded(self: *Self, allocator: Allocator, additional_size: Size, ctx: Context) Allocator.Error!void {
            if (additional_size > self.available) {
                const new_size = self.load() + additional_size;
                const new_cap = capacityForSize(new_size);
                debug.assert(self.capacity() != new_cap);
                try self.grow(allocator, new_cap, ctx);
            }
        }

        fn load(self: *const Self) Size {
            const max_load = (self.capacity() * max_load_percentage) / 100;
            debug.assert(max_load >= self.available);
            return @as(Size, @intCast(max_load - self.available));
        }

        fn grow(self: *Self, allocator: Allocator, new_capacity: Size, ctx: Context) Allocator.Error!void {
            @setCold(true);
            debug.assert(new_capacity >= minimal_capacity);
            debug.assert(new_capacity > self.capacity());
            debug.assert(@popCount(new_capacity) == 1);

            var map = Self{};
            defer map.deinit(allocator);
            try map.allocate(allocator, new_capacity);
            map.initMetadatas();
            map.available = @intCast((new_capacity * max_load_percentage) / 100);
            if (self.size != 0) {
                copyAndRehash(&map, self, ctx);
            }
            mem.swap(Self, self, &map);
        }

        fn copyAndRehash(noalias dst: *Self, noalias src: *const Self, ctx: anytype) void {
            debug.assert(dst.size == 0);
            debug.assert(src.size <= @as(Size, @intCast((dst.capacity() * max_load_percentage) / 100)));
            defer debug.assert(dst.size == src.size);
            var aligned_idx: usize = 0;
            var group_count = src.capacity() / Group.lane_count;
            while (group_count != 0 or src.size != dst.size) : ({
                aligned_idx += Group.lane_count;
                group_count -= 1;
            }) {
                const group = Group.fromMetadata(src.metadata.? + aligned_idx);
                var used_matches = group.matchUsed();
                while (used_matches != 0) : (used_matches &= used_matches - 1) {
                    const idx = aligned_idx + @ctz(used_matches);
                    const key = src.keys()[idx];
                    const val = src.values()[idx];
                    dst.putAssumeCapacityNoClobberContext(key, val, ctx);
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

        pub const Iterator = struct {
            hm: *const Self,
            base_idx: Size = 0,
            used: Group.MatchMask = 0,
            pub fn next(it: *Iterator) ?Entry {
                debug.assert(it.base_idx <= it.hm.capacity());
                while (it.used == 0) : (it.base_idx += Group.lane_count) {
                    if (it.base_idx == it.hm.capacity()) {
                        return null;
                    }
                    const group = Group.fromMetadata(it.hm.metadata.? + it.base_idx);
                    it.used = group.matchUsed();
                }
                defer it.used &= it.used - 1;
                const idx = @ctz(it.used);
                return Entry{
                    .key_ptr = &it.hm.keys()[idx],
                    .value_ptr = &it.hm.values()[idx],
                };
            }
        };

        pub const KeyIterator = FieldIterator(K);
        pub const ValueIterator = FieldIterator(V);
        fn FieldIterator(comptime T: type) type {
            return struct {
                metadata: [*]const Metadata,
                items: [*]const T,
                used: Group.MatchMask = 0,
                based_idx: usize,

                pub fn next(it: *@This()) ?*T {
                    while (it.used == 0) {
                        const res = @subWithOverflow(it.based_idx, Group.lane_count);
                        if (res[1] == 1) return null;
                        it.based_idx = res[0];
                        const group = Group.fromMetadata(it.metadata - it.based_idx);
                        it.used = group.matchUsed();
                    }
                    defer it.used &= it.used - 1;
                    const idx = @ctz(it.used);
                    return &it.items[idx];
                }
            };
        }

        pub fn promote(self: Self, allocator: Allocator) Managed {
            if (@sizeOf(Context) != 0) {
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call promoteContext instead.");
            }
            return promoteContext(self, allocator, undefined);
        }

        pub fn promoteContext(self: Self, allocator: Allocator, ctx: Context) Managed {
            return .{
                .unmanaged = self,
                .allocator = allocator,
                .ctx = ctx,
            };
        }

        pub fn deinit(self: *Self, allocator: Allocator) void {
            self.deallocate(allocator);
        }

        pub fn ensureTotalCapacity(self: *Self, allocator: Allocator, new_capacity: Size) Allocator.Error!void {
            if (@sizeOf(Context) != 0) {
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call `ensure_total_capacity_context` instead.");
            }
            return self.ensureTotalCapacityContext(allocator, new_capacity, undefined);
        }
        pub fn ensureTotalCapacityContext(self: *Self, allocator: Allocator, new_size: Size, ctx: Context) Allocator.Error!void {
            if (self.size >= new_size) return;
            const additional_size = new_size - self.size;
            try self.growIfNeeded(allocator, additional_size, ctx);
        }

        pub fn ensureUnusedCapacity(self: *Self, allocator: Allocator, additional_size: Size) Allocator.Error!void {
            if (@sizeOf(Context) != 0) {
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call `ensure_unused_capacity_context` instead.");
            }
            try self.ensureUnusedCapacityContext(allocator, additional_size, undefined);
        }
        pub fn ensureUnusedCapacityContext(self: *Self, allocator: Allocator, additional_size: Size, ctx: Context) Allocator.Error!void {
            try self.growIfNeeded(allocator, additional_size, ctx);
        }

        pub fn clearRetainingCapacity(self: *Self) void {
            if (self.metadata == null) return;
            self.initMetadatas();
            self.size = 0;
            self.available = @intCast((self.capacity() * max_load_percentage) / 100);
        }

        pub fn clearAndFree(self: *Self, allocator: Allocator) void {
            self.deallocate(allocator);
        }

        pub fn count(self: *const Self) Size {
            return self.size;
        }

        pub fn contains(self: *const Self, key: K) bool {
            return self.containsContext(key, undefined);
        }
        pub fn containsContext(self: *const Self, key: K, ctx: Context) bool {
            return self.containsAdapted(key, ctx);
        }
        pub fn containsAdapted(self: *const Self, key: anytype, ctx: anytype) bool {
            return self.getIndex(key, ctx) != null;
        }

        pub fn iterator(self: *const Self) Iterator {
            return Iterator{ .hm = self };
        }

        pub fn keyIterator(self: *const Self) KeyIterator {
            const metadata = self.metadatas() orelse return .{
                .metadata = undefined,
                .items = undefined,
                .based_idx = 0,
            };
            return .{
                .metadata = metadata,
                .items = self.keys(),
                .based_idx = self.capacity(),
            };
        }

        pub fn valueIterator(self: *const Self) ValueIterator {
            const metadata = self.metadatas() orelse return .{
                .metadata = undefined,
                .items = undefined,
                .based_idx = 0,
            };
            return .{
                .metadata = metadata,
                .items = self.values(),
                .based_idx = self.capacity(),
            };
        }

        pub fn clone(self: Self, allocator: Allocator) Allocator.Error!Self {
            if (@sizeOf(Context) != 0) {
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call cloneContext instead.");
            }
            return self.cloneContext(allocator, @as(Context, undefined));
        }

        pub fn cloneContext(self: Self, allocator: Allocator, new_ctx: anytype) Allocator.Error!HashMapUnmanaged(K, V, @TypeOf(new_ctx), max_load_percentage, processor) {
            var other = HashMapUnmanaged(K, V, @TypeOf(new_ctx), max_load_percentage, processor){};
            if (self.size == 0) return other;

            // NOTE:  The cloned map's capacity is derived from the orignal map's size, not capacity.
            // As the orignal map's might have excessive capacity.
            // For this, if the original map size is zero, the cloned map remains unallocated.
            const new_cap = capacityForSize(self.size);
            try other.allocate(allocator, new_cap);
            other.initMetadatas();
            other.available = @intCast((new_cap * max_load_percentage) / 100);
            copyAndRehash(&other, &self, new_ctx);

            return other;
        }

        pub fn getEntry(self: Self, key: K) ?Entry {
            if (@sizeOf(Context) != 0)
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call getEntryContext instead.");
            return self.getEntryContext(key, undefined);
        }
        pub fn getEntryContext(self: Self, key: K, ctx: Context) ?Entry {
            return self.getEntryAdapted(key, ctx);
        }
        pub fn getEntryAdapted(self: Self, key: anytype, ctx: anytype) ?Entry {
            if (self.getIndex(key, ctx)) |idx| {
                return Entry{
                    .key_ptr = &self.keys()[idx],
                    .value_ptr = &self.values()[idx],
                };
            }
            return null;
        }

        pub fn getKeyPtr(self: Self, key: K) ?*K {
            if (@sizeOf(Context) != 0)
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call getKeyPtrContext instead.");
            return self.getKeyPtrContext(key, undefined);
        }
        pub fn getKeyPtrContext(self: Self, key: K, ctx: Context) ?*K {
            return self.getKeyPtrAdapted(key, ctx);
        }
        pub fn getKeyPtrAdapted(self: Self, key: anytype, ctx: anytype) ?*K {
            if (self.getIndex(key, ctx)) |idx| {
                return &self.keys()[idx];
            }
            return null;
        }

        pub fn getKey(self: Self, key: K) ?K {
            if (@sizeOf(Context) != 0)
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call getKeyContext instead.");
            return self.getKeyContext(key, undefined);
        }
        pub fn getKeyContext(self: Self, key: K, ctx: Context) ?K {
            return self.getKeyAdapted(key, ctx);
        }
        pub fn getKeyAdapted(self: Self, key: anytype, ctx: anytype) ?K {
            if (self.getIndex(key, ctx)) |idx| {
                return self.keys()[idx];
            }
            return null;
        }

        pub fn getPtr(self: Self, key: K) ?*V {
            if (@sizeOf(Context) != 0)
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call getPtrContext instead.");
            return self.getPtrContext(key, undefined);
        }
        pub fn getPtrContext(self: Self, key: K, ctx: Context) ?*V {
            return self.getPtrAdapted(key, ctx);
        }
        pub fn getPtrAdapted(self: Self, key: anytype, ctx: anytype) ?*V {
            if (self.getIndex(key, ctx)) |idx| {
                return &self.values()[idx];
            }
            return null;
        }

        pub fn get(self: Self, key: K) ?V {
            if (@sizeOf(Context) != 0) {
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call getContext instead.");
            }
            return self.getContext(key, undefined);
        }
        pub fn getContext(self: Self, key: K, ctx: Context) ?V {
            return self.getAdapted(key, ctx);
        }
        pub fn getAdapted(self: Self, key: anytype, ctx: anytype) ?V {
            if (self.getIndex(key, ctx)) |idx| {
                return self.values()[idx];
            }
            return null;
        }

        pub fn put(self: *Self, allocator: Allocator, key: K, value: V) Allocator.Error!void {
            if (@sizeOf(Context) != 0) {
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call `put_context` instead.");
            }
            return self.putContext(allocator, key, value, undefined);
        }
        pub fn putContext(self: *Self, allocator: Allocator, key: K, value: V, ctx: Context) Allocator.Error!void {
            const result = try self.getOrPutContext(allocator, key, ctx);
            result.value_ptr.* = value;
        }

        pub fn putNoClobber(self: *Self, allocator: Allocator, key: K, value: V) Allocator.Error!void {
            if (@sizeOf(Context) != 0) {
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call `put_no_clobber_context` instead.");
            }
            self.putNoClobberContext(allocator, key, value, undefined);
        }
        pub fn putNoClobberContext(self: *Self, allocator: Allocator, key: K, value: V, ctx: Context) Allocator.Error!void {
            debug.assert(!self.containsContext(key, ctx));
            try self.growIfNeeded(allocator, 1, ctx);

            self.putAssumeCapacityNoClobberContext(key, value, ctx);
        }

        pub fn putAssumeCapacity(self: *Self, key: K, value: V) void {
            if (@sizeOf(Context) != 0)
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call `put_assume_capacity_context` instead.");
            return self.putAssumeCapacityContext(key, value, undefined);
        }
        pub fn putAssumeCapacityContext(self: *Self, key: K, value: V, ctx: Context) void {
            const gop = self.getOrPutAssumeCapacityContext(key, ctx);
            gop.value_ptr.* = value;
        }

        pub fn putAssumeCapacityNoClobber(self: *Self, key: K, value: V) void {
            if (@sizeOf(Context) != 0) {
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call `put_assume_capacity_no_clobber_context` instead.");
            }
            self.putAssumeCapacityNoClobberContext(key, value, undefined);
        }
        pub fn putAssumeCapacityNoClobberContext(self: *Self, key: K, value: V, ctx: Context) void {
            debug.assert(!self.containsContext(key, ctx));
            const hash = ctx.hash(key);
            const hash_code = getHashcode(hash);
            const cap_mask = self.capacity() - 1;

            // TODO: this might not be correct for scalar
            var base_idx = hash_code & cap_mask;

            const idx = while (true) : ({
                base_idx = (base_idx + Group.lane_count) % (self.capacity() + Group.lane_count);
                base_idx *= @intFromBool(base_idx < self.capacity());
            }) {
                const group = Group.fromMetadata(self.metadata.? + base_idx);
                const unused_matches = group.matchUnused();
                if (unused_matches == 0) continue;
                break base_idx + @ctz(unused_matches);
            };

            debug.assert(self.available > 0);
            self.metadata.?[idx] = getFingerprint(hash);
            self.keys()[idx] = key;
            self.values()[idx] = value;
            self.available -= 1;
            self.size += 1;
        }

        pub fn fetchPut(self: *Self, allocator: Allocator, key: K, value: V) Allocator.Error!?KV {
            if (@sizeOf(Context) != 0) {
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call `fetch_put_context` instead.");
            }
            return self.fetchPutContext(allocator, key, value, undefined);
        }
        pub fn fetchPutContext(self: *Self, allocator: Allocator, key: K, value: V, ctx: Context) Allocator.Error!?KV {
            const gop = try self.getOrPutContext(allocator, key, ctx);
            defer gop.value_ptr.* = value;
            if (gop.found_existing) {
                return KV{
                    .key = gop.key_ptr.*,
                    .value = gop.value_ptr.*,
                };
            }
            return null;
        }

        pub fn fetchPutAssumeCapacity(self: *Self, key: K, value: V) ?KV {
            if (@sizeOf(Context) != 0) {
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call `fetch_put_assume_capacity_context` instead.");
            }
            return self.fetchPutAssumeCapacityContext(key, value, undefined);
        }
        pub fn fetchPutAssumeCapacityContext(self: *Self, key: K, value: V, ctx: Context) ?KV {
            const gop = self.getOrPutAssumeCapacityContext(key, ctx);
            defer gop.value_ptr.* = value;
            if (gop.found_existing) {
                return KV{
                    .key = gop.key_ptr.*,
                    .value = gop.value_ptr.*,
                };
            }
            return null;
        }

        pub fn getOrPut(self: *Self, allocator: Allocator, key: K) Allocator.Error!GetOrPutResult {
            if (@sizeOf(Context) != 0) {
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call `get_or_put_context` instead.");
            }
            return self.getOrPutContext(allocator, key, undefined);
        }
        pub fn getOrPutContext(self: *Self, allocator: Allocator, key: K, ctx: Context) Allocator.Error!GetOrPutResult {
            const gop = try self.getOrPutContextAdapted(allocator, key, ctx, ctx);
            if (!gop.found_existing) {
                gop.key_ptr.* = key;
            }
            return gop;
        }
        pub fn getOrPutAdapted(self: *Self, allocator: Allocator, key: anytype, key_ctx: anytype) Allocator.Error!GetOrPutResult {
            if (@sizeOf(Context) != 0) {
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call `get_or_put_context_adapted` instead.");
            }
            return self.getOrPutContextAdapted(allocator, key, key_ctx, undefined);
        }
        pub fn getOrPutContextAdapted(self: *Self, allocator: Allocator, key: anytype, key_ctx: anytype, ctx: Context) Allocator.Error!GetOrPutResult {
            self.growIfNeeded(allocator, 1, ctx) catch |err| {
                // If allocation fails, try to do the lookup anyway.
                // If we find an existing item, we can return it.
                // Otherwise return the error, we could not add another.
                const index = self.getIndex(key, key_ctx) orelse return err;
                return GetOrPutResult{
                    .key_ptr = &self.keys()[index],
                    .value_ptr = &self.values()[index],
                    .found_existing = true,
                };
            };
            return self.getOrPutAssumeCapacityAdapted(key, key_ctx);
        }

        pub fn getOrPutValue(self: *Self, allocator: Allocator, key: K, value: V) Allocator.Error!Entry {
            if (@sizeOf(Context) != 0) {
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call getOrPutValueContext instead.");
            }
            return self.getOrPutValueContext(allocator, key, value, undefined);
        }
        pub fn getOrPutValueContext(self: *Self, allocator: Allocator, key: K, value: V, ctx: Context) Allocator.Error!Entry {
            const gop = try self.getOrPutAdapted(allocator, key, ctx);
            if (!gop.found_existing) {
                gop.key_ptr.* = key;
                gop.value_ptr.* = value;
            }
            return Entry{ .key_ptr = gop.key_ptr, .value_ptr = gop.value_ptr };
        }

        pub fn getOrPutAssumeCapacity(self: *Self, key: K) GetOrPutResult {
            if (@sizeOf(Context) != 0) {
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call `get_or_put_assume_capacity_context` instead.");
            }
            return self.getOrPutAssumeCapacityContext(key, undefined);
        }
        pub fn getOrPutAssumeCapacityContext(self: *Self, key: K, ctx: Context) GetOrPutResult {
            var gop = self.getOrPutAssumeCapacityAdapted(key, ctx);
            if (!gop.found_existing) {
                gop.key_ptr.* = key;
            }
            return gop;
        }
        pub fn getOrPutAssumeCapacityAdapted(self: *Self, key: anytype, ctx: anytype) GetOrPutResult {
            comptime verifyContext(@TypeOf(ctx), @TypeOf(key), K, Hash, false);
            const hash = ctx.hash(key);
            if (@TypeOf(hash) != Hash) {
                @compileError("Context " ++ @typeName(@TypeOf(ctx)) ++ " has a generic hash function that returns the wrong type! " ++ @typeName(Hash) ++ " was expected, but found " ++ @typeName(@TypeOf(hash)));
            }

            const hashcode: Hashcode = getHashcode(hash);
            const fingerprint: Fingerprint = getFingerprint(hash);

            debug.assert(@popCount(self.capacity()) == 1);
            const cap_mask = self.capacity() - 1;
            var group_count = (self.capacity() / Group.lane_count) + 1;

            // TODO: this might not be correct for scalar
            // NOTE: `base_idx` does not align with `Group.Data`
            var base_idx: usize = hashcode & cap_mask;
            var opt_idx: ?usize = null;

            while (group_count != 0) : ({
                base_idx = (base_idx + Group.lane_count) % (self.capacity() + Group.lane_count);
                base_idx *= @intFromBool(base_idx < self.capacity());
                group_count -= 1;
            }) {
                if (base_idx >= self.capacity()) continue;
                const group = Group.fromMetadata(self.metadata.? + base_idx);

                var fp_matches = group.matchFingerprint(fingerprint);
                while (fp_matches != 0) : (fp_matches &= fp_matches - 1) {
                    const idx = base_idx + @ctz(fp_matches);
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
                    const unused_matches = group.matchUnused();
                    if (unused_matches == 0) break :blk;
                    opt_idx = base_idx + @ctz(unused_matches);
                }
                if (group.matchFree() != 0) {
                    break;
                }
            }
            const idx = opt_idx.?;

            debug.assert(self.available > 0);

            self.size += 1;
            self.available -= 1;
            self.metadata.?[idx] = fingerprint;

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

        pub fn remove(self: *Self, key: K) bool {
            if (@sizeOf(Context) != 0) {
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call `remove_context` instead.");
            }
            return self.removeContext(key, undefined);
        }
        pub fn removeContext(self: *Self, key: K, ctx: Context) bool {
            return self.removeAdapted(key, ctx);
        }
        pub fn removeAdapted(self: *Self, key: K, ctx: anytype) bool {
            const idx = self.getIndex(key, ctx) orelse return false;
            self.removeByIndex(idx);
            return true;
        }

        pub fn fetchRemove(self: *Self, key: K) ?KV {
            if (@sizeOf(Context) != 0) {
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call `fetch_remove_context` instead.");
            }
            return self.fetchRemoveContext(key, undefined);
        }
        pub fn fetchRemoveContext(self: *Self, key: K, ctx: Context) ?KV {
            return self.fetchRemoveAdapted(key, ctx);
        }
        pub fn fetchRemoveAdapted(self: *Self, key: K, ctx: anytype) ?KV {
            const idx = self.getIndex(key, ctx) orelse return null;
            defer self.removeByIndex(idx);
            return KV{
                .key = self.keys()[idx],
                .value = self.values()[idx],
            };
        }

        pub fn removeByPtr(self: *Self, key_ptr: *K) void {
            // TODO: Not quite sure what does `std.hash_map` wanted to achieve.
            // The comment says, if `@sizeOf(K) == 0`, we will always get zero index.
            // But I thought the hash value is depends on the hash function right? regardless of the size of the K.
            // And what is the practical usage of having hash map that it's key size is zero anyway.
            _ = self.remove(key_ptr.*);
        }

        fn _debug_print_metadata(self: *const Self) void {
            const metadata = self.metadata orelse return;
            const len = self.capacity() + Group.lane_count;
            var idx: usize = 0;
            var gidx: usize = 0;

            print("⎯⎯⎯⎯│", .{});
            for (0..Group.lane_count) |_| print("⎯⎯⎯│", .{});
            print("\n", .{});

            print("    │", .{});
            for (0..Group.lane_count) |i| {
                print("{:>3}│", .{i});
            }
            print("\n", .{});

            print("⎯⎯⎯⎯│", .{});
            for (0..Group.lane_count) |_| print("⎯⎯⎯│", .{});
            print("\n", .{});

            while (idx != len) : (idx += 1) {
                if (idx % Group.lane_count == 0) print("{:>4}│", .{gidx});
                const md = metadata[idx];
                switch (md) {
                    0...0b01111111 => print("{:>3}│", .{md}),
                    METADATA_SENTINEL => print("  X│", .{}),
                    METADATA_TOMBSTONE => print("  T│", .{}),
                    METADATA_FREE => print("   │", .{}),
                    else => unreachable,
                }
                if ((idx + 1) % Group.lane_count == 0) {
                    gidx += 1;
                    print("\n", .{});
                }
            }

            print("⎯⎯⎯⎯│", .{});
            for (0..Group.lane_count) |_| print("⎯⎯⎯│", .{});
            print("\n\n", .{});
        }
    };
}

const testing = std.testing;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;

test "std.hash_map basic usage" {
    var map = AutoHashMap(u32, u32).init(std.testing.allocator);
    defer map.deinit();

    const count = 5;
    var i: u32 = 0;
    var total: u32 = 0;
    while (i < count) : (i += 1) {
        try map.put(i, i);
        total += i;
    }

    var sum: u32 = 0;
    var it = map.iterator();
    while (it.next()) |kv| {
        sum += kv.key_ptr.*;
    }
    try expectEqual(total, sum);

    i = 0;
    sum = 0;
    while (i < count) : (i += 1) {
        try expectEqual(i, map.get(i).?);
        sum += map.get(i).?;
    }
    try expectEqual(total, sum);
}

test "std.hash_map ensureTotalCapacity" {
    var map = AutoHashMap(i32, i32).init(std.testing.allocator);
    defer map.deinit();

    try map.ensureTotalCapacity(20);
    const initial_capacity = map.capacity();
    try testing.expect(initial_capacity >= 20);
    var i: i32 = 0;
    while (i < 20) : (i += 1) {
        try testing.expect(map.fetchPutAssumeCapacity(i, i + 10) == null);
    }
    // shouldn't resize from putAssumeCapacity
    try testing.expect(initial_capacity == map.capacity());
}

test "std.hash_map ensureUnusedCapacity with tombstones" {
    // if (true) return error.SkipZigTest;
    var map = AutoHashMap(i32, i32).init(std.testing.allocator);
    defer map.deinit();

    var i: i32 = 0;
    while (i < 100) : (i += 1) {
        try map.ensureUnusedCapacity(1);
        map.putAssumeCapacity(i, i);
        _ = map.remove(i);
    }
}

test "std.hash_map clearRetainingCapacity" {
    // if (true) return error.SkipZigTest;
    var map = AutoHashMap(u32, u32).init(std.testing.allocator);
    defer map.deinit();

    map.clearRetainingCapacity();

    try map.put(1, 1);
    try expectEqual(map.get(1).?, 1);
    try expectEqual(map.count(), 1);

    map.clearRetainingCapacity();
    map.putAssumeCapacity(1, 1);
    try expectEqual(map.get(1).?, 1);
    try expectEqual(map.count(), 1);

    const cap = map.capacity();
    try expect(cap > 0);

    map.clearRetainingCapacity();
    map.clearRetainingCapacity();
    try expectEqual(map.count(), 0);
    try expectEqual(map.capacity(), cap);
    try expect(!map.contains(1));
}

test "std.hash_map grow" {
    // if (true) return error.SkipZigTest;
    var map = AutoHashMap(u32, u32).init(std.testing.allocator);
    defer map.deinit();

    const growTo = 12456;

    var i: u32 = 0;
    while (i < growTo) : (i += 1) {
        try map.put(i, i);
    }
    try expectEqual(map.count(), growTo);

    i = 0;
    var it = map.iterator();
    while (it.next()) |kv| {
        try expectEqual(kv.key_ptr.*, kv.value_ptr.*);
        i += 1;
    }
    try expectEqual(i, growTo);

    i = 0;
    while (i < growTo) : (i += 1) {
        try expectEqual(map.get(i).?, i);
    }
}

test "std.hash_map clone" {
    // if (true) return error.SkipZigTest;
    var map = AutoHashMap(u32, u32).init(std.testing.allocator);
    defer map.deinit();

    var a = try map.clone();
    defer a.deinit();

    try expectEqual(a.count(), 0);

    try a.put(1, 1);
    try a.put(2, 2);
    try a.put(3, 3);

    var b = try a.clone();
    defer b.deinit();

    try expectEqual(b.count(), 3);
    try expectEqual(b.get(1).?, 1);
    try expectEqual(b.get(2).?, 2);
    try expectEqual(b.get(3).?, 3);

    var original = AutoHashMap(i32, i32).init(std.testing.allocator);
    defer original.deinit();

    var i: u8 = 0;
    while (i < 10) : (i += 1) {
        try original.putNoClobber(i, i * 10);
    }

    var copy = try original.clone();
    defer copy.deinit();

    i = 0;
    while (i < 10) : (i += 1) {
        try testing.expect(copy.get(i).? == i * 10);
    }
}

test "std.hash_map ensureTotalCapacity with existing elements" {
    // if (true) return error.SkipZigTest;
    var map = AutoHashMap(u32, u32).init(std.testing.allocator);
    defer map.deinit();

    try map.put(0, 0);
    try expectEqual(map.count(), 1);
    try expectEqual(map.capacity(), @TypeOf(map).Unmanaged.minimal_capacity);

    try map.ensureTotalCapacity(65);
    try expectEqual(map.count(), 1);
    try expectEqual(map.capacity(), 128);
}

test "std.hash_map ensureTotalCapacity satisfies max load factor" {
    // if (true) return error.SkipZigTest;
    var map = AutoHashMap(u32, u32).init(std.testing.allocator);
    defer map.deinit();

    try map.ensureTotalCapacity(127);
    try expectEqual(map.capacity(), 256);
}

test "std.hash_map remove" {
    // if (true) return error.SkipZigTest;
    var map = AutoHashMap(u32, u32).init(std.testing.allocator);
    defer map.deinit();

    var i: u32 = 0;
    while (i < 16) : (i += 1) {
        try map.put(i, i);
    }

    i = 0;
    while (i < 16) : (i += 1) {
        if (i % 3 == 0) {
            _ = map.remove(i);
        }
    }
    try expectEqual(map.count(), 10);
    var it = map.iterator();
    while (it.next()) |kv| {
        try expectEqual(kv.key_ptr.*, kv.value_ptr.*);
        try expect(kv.key_ptr.* % 3 != 0);
    }

    i = 0;
    while (i < 16) : (i += 1) {
        if (i % 3 == 0) {
            try expect(!map.contains(i));
        } else {
            try expectEqual(map.get(i).?, i);
        }
    }
}

test "std.hash_map reverse removes" {
    // if (true) return error.SkipZigTest;
    var map = AutoHashMap(u32, u32).init(std.testing.allocator);
    defer map.deinit();

    var i: u32 = 0;
    while (i < 16) : (i += 1) {
        try map.putNoClobber(i, i);
    }

    i = 16;
    while (i > 0) : (i -= 1) {
        _ = map.remove(i - 1);
        try expect(!map.contains(i - 1));
        var j: u32 = 0;
        while (j < i - 1) : (j += 1) {
            try expectEqual(map.get(j).?, j);
        }
    }

    try expectEqual(map.count(), 0);
}

test "std.hash_map multiple removes on same metadata" {
    // if (true) return error.SkipZigTest;
    var map = AutoHashMap(u32, u32).init(std.testing.allocator);
    defer map.deinit();

    var i: u32 = 0;
    while (i < 16) : (i += 1) {
        try map.put(i, i);
    }

    _ = map.remove(7);
    _ = map.remove(15);
    _ = map.remove(14);
    _ = map.remove(13);
    try expect(!map.contains(7));
    try expect(!map.contains(15));
    try expect(!map.contains(14));
    try expect(!map.contains(13));

    i = 0;
    while (i < 13) : (i += 1) {
        if (i == 7) {
            try expect(!map.contains(i));
        } else {
            try expectEqual(map.get(i).?, i);
        }
    }

    try map.put(15, 15);
    try map.put(13, 13);
    try map.put(14, 14);
    try map.put(7, 7);
    i = 0;
    while (i < 16) : (i += 1) {
        try expectEqual(map.get(i).?, i);
    }
}

test "std.hash_map put and remove loop in random order" {
    // if (true) return error.SkipZigTest;
    var map = AutoHashMap(u32, u32).init(std.testing.allocator);
    defer map.deinit();

    var keys = std.ArrayList(u32).init(std.testing.allocator);
    defer keys.deinit();

    const size = 32;
    const iterations = 100;

    var i: u32 = 0;
    while (i < size) : (i += 1) {
        try keys.append(i);
    }
    var prng = std.rand.DefaultPrng.init(0);
    const random = prng.random();

    while (i < iterations) : (i += 1) {
        random.shuffle(u32, keys.items);

        for (keys.items) |key| {
            try map.put(key, key);
        }
        try expectEqual(map.count(), size);

        for (keys.items) |key| {
            _ = map.remove(key);
        }
        try expectEqual(map.count(), 0);
    }
}

test "std.hash_map remove one million elements in random order" {
    // if (true) return error.SkipZigTest;
    const Map = AutoHashMap(u32, u32);
    const n = 1000 * 1000;
    var map = Map.init(std.heap.page_allocator);
    defer map.deinit();

    var keys = std.ArrayList(u32).init(std.heap.page_allocator);
    defer keys.deinit();

    var i: u32 = 0;
    while (i < n) : (i += 1) {
        keys.append(i) catch unreachable;
    }

    var prng = std.rand.DefaultPrng.init(0);
    const random = prng.random();
    random.shuffle(u32, keys.items);

    for (keys.items) |key| {
        map.put(key, key) catch unreachable;
    }

    random.shuffle(u32, keys.items);
    i = 0;
    while (i < n) : (i += 1) {
        const key = keys.items[i];
        _ = map.remove(key);
    }
}

test "std.hash_map put" {
    // if (true) return error.SkipZigTest;
    var map = AutoHashMap(u32, u32).init(std.testing.allocator);
    defer map.deinit();

    var i: u32 = 0;
    while (i < 16) : (i += 1) {
        try map.put(i, i);
    }

    i = 0;
    while (i < 16) : (i += 1) {
        try expectEqual(map.get(i).?, i);
    }

    i = 0;
    while (i < 16) : (i += 1) {
        try map.put(i, i * 16 + 1);
    }

    i = 0;
    while (i < 16) : (i += 1) {
        try expectEqual(map.get(i).?, i * 16 + 1);
    }
}

test "std.hash_map putAssumeCapacity" {
    // if (true) return error.SkipZigTest;
    var map = AutoHashMap(u32, u32).init(std.testing.allocator);
    defer map.deinit();

    try map.ensureTotalCapacity(20);
    var i: u32 = 0;
    while (i < 20) : (i += 1) {
        map.putAssumeCapacityNoClobber(i, i);
    }

    i = 0;
    var sum = i;
    while (i < 20) : (i += 1) {
        sum += map.getPtr(i).?.*;
    }
    try expectEqual(sum, 190);

    i = 0;
    while (i < 20) : (i += 1) {
        map.putAssumeCapacity(i, 1);
    }

    i = 0;
    sum = i;
    while (i < 20) : (i += 1) {
        sum += map.get(i).?;
    }
    try expectEqual(sum, 20);
}

test "std.hash_map repeat putAssumeCapacity/remove" {
    // if (true) return error.SkipZigTest;
    var map = AutoHashMap(u32, u32).init(std.testing.allocator);
    defer map.deinit();

    try map.ensureTotalCapacity(20);
    const limit = map.unmanaged.available;

    var i: u32 = 0;
    while (i < limit) : (i += 1) {
        map.putAssumeCapacityNoClobber(i, i);
    }

    // Repeatedly delete/insert an entry without resizing the map.
    // Put to different keys so entries don't land in the just-freed slot.
    i = 0;
    while (i < 10 * limit) : (i += 1) {
        try testing.expect(map.remove(i));
        if (i % 2 == 0) {
            map.putAssumeCapacityNoClobber(limit + i, i);
        } else {
            map.putAssumeCapacity(limit + i, i);
        }
    }

    i = 9 * limit;
    while (i < 10 * limit) : (i += 1) {
        try expectEqual(map.get(limit + i), i);
    }
    try expectEqual(map.unmanaged.available, 0);
    try expectEqual(map.unmanaged.count(), limit);
}

test "std.hash_map getOrPut" {
    // if (true) return error.SkipZigTest;
    var map = AutoHashMap(u32, u32).init(std.testing.allocator);
    defer map.deinit();

    var i: u32 = 0;
    while (i < 10) : (i += 1) {
        try map.put(i * 2, 2);
    }

    i = 0;
    while (i < 20) : (i += 1) {
        _ = try map.getOrPutValue(i, 1);
    }

    i = 0;
    var sum = i;
    while (i < 20) : (i += 1) {
        sum += map.get(i).?;
    }

    try expectEqual(sum, 30);
}

test "std.hash_map basic hash map usage" {
    // if (true) return error.SkipZigTest;
    var map = AutoHashMap(i32, i32).init(std.testing.allocator);
    defer map.deinit();

    try testing.expect((try map.fetchPut(1, 11)) == null);
    try testing.expect((try map.fetchPut(2, 22)) == null);
    try testing.expect((try map.fetchPut(3, 33)) == null);
    try testing.expect((try map.fetchPut(4, 44)) == null);

    try map.putNoClobber(5, 55);
    try testing.expect((try map.fetchPut(5, 66)).?.value == 55);
    try testing.expect((try map.fetchPut(5, 55)).?.value == 66);

    const gop1 = try map.getOrPut(5);
    try testing.expect(gop1.found_existing == true);
    try testing.expect(gop1.value_ptr.* == 55);
    gop1.value_ptr.* = 77;
    try testing.expect(map.getEntry(5).?.value_ptr.* == 77);

    const gop2 = try map.getOrPut(99);
    try testing.expect(gop2.found_existing == false);
    gop2.value_ptr.* = 42;
    try testing.expect(map.getEntry(99).?.value_ptr.* == 42);

    const gop3 = try map.getOrPutValue(5, 5);
    try testing.expect(gop3.value_ptr.* == 77);

    const gop4 = try map.getOrPutValue(100, 41);
    try testing.expect(gop4.value_ptr.* == 41);

    try testing.expect(map.contains(2));
    try testing.expect(map.getEntry(2).?.value_ptr.* == 22);
    try testing.expect(map.get(2).? == 22);

    const rmv1 = map.fetchRemove(2);
    try testing.expect(rmv1.?.key == 2);
    try testing.expect(rmv1.?.value == 22);
    try testing.expect(map.fetchRemove(2) == null);
    try testing.expect(map.remove(2) == false);
    try testing.expect(map.getEntry(2) == null);
    try testing.expect(map.get(2) == null);

    try testing.expect(map.remove(3) == true);
}

test "std.hash_map getOrPutAdapted" {
    // if (true) return error.SkipZigTest;
    const AdaptedContext = struct {
        fn eql(self: @This(), adapted_key: []const u8, test_key: u64) bool {
            _ = self;
            return std.fmt.parseInt(u64, adapted_key, 10) catch unreachable == test_key;
        }
        fn hash(self: @This(), adapted_key: []const u8) u64 {
            _ = self;
            const key = std.fmt.parseInt(u64, adapted_key, 10) catch unreachable;
            return (AutoContext(u64){}).hash(key);
        }
    };
    var map = AutoHashMap(u64, u64).init(testing.allocator);
    defer map.deinit();

    const keys = [_][]const u8{
        "1231",
        "4564",
        "7894",
        "1132",
        "65235",
        "95462",
        "0112305",
        "00658",
        "0",
        "2",
    };

    var real_keys: [keys.len]u64 = undefined;

    inline for (keys, 0..) |key_str, i| {
        const result = try map.getOrPutAdapted(key_str, AdaptedContext{});
        try testing.expect(!result.found_existing);
        real_keys[i] = std.fmt.parseInt(u64, key_str, 10) catch unreachable;
        result.key_ptr.* = real_keys[i];
        result.value_ptr.* = i * 2;
    }

    try testing.expectEqual(map.count(), keys.len);

    inline for (keys, 0..) |key_str, i| {
        const result = map.getOrPutAssumeCapacityAdapted(key_str, AdaptedContext{});
        try testing.expect(result.found_existing);
        try testing.expectEqual(real_keys[i], result.key_ptr.*);
        try testing.expectEqual(@as(u64, i) * 2, result.value_ptr.*);
        try testing.expectEqual(real_keys[i], map.getKeyAdapted(key_str, AdaptedContext{}).?);
    }
}

test "std.hash_map ensureUnusedCapacity" {
    // if (true) return error.SkipZigTest;
    var map = AutoHashMap(u64, u64).init(testing.allocator);
    defer map.deinit();

    try map.ensureUnusedCapacity(32);
    const capacity = map.capacity();
    try map.ensureUnusedCapacity(32);

    // Repeated ensureUnusedCapacity() calls with no insertions between
    // should not change the capacity.
    try testing.expectEqual(capacity, map.capacity());
}

test "std.hash_map removeByPtr" {
    // if (true) return error.SkipZigTest;
    var map = AutoHashMap(i32, u64).init(testing.allocator);
    defer map.deinit();

    var i: i32 = undefined;

    i = 0;
    while (i < 10) : (i += 1) {
        try map.put(i, 0);
    }

    try testing.expect(map.count() == 10);

    i = 0;
    while (i < 10) : (i += 1) {
        const key_ptr = map.getKeyPtr(i);
        try testing.expect(key_ptr != null);

        if (key_ptr) |ptr| {
            map.removeByPtr(ptr);
        }
    }

    try testing.expect(map.count() == 0);
}

test "std.hash_map removeByPtr 0 sized key" {
    // if (true) return error.SkipZigTest;
    var map = AutoHashMap(u0, u64).init(testing.allocator);
    defer map.deinit();

    try map.put(0, 0);

    try testing.expect(map.count() == 1);

    const key_ptr = map.getKeyPtr(0);
    try testing.expect(key_ptr != null);

    if (key_ptr) |ptr| {
        map.removeByPtr(ptr);
    }

    try testing.expect(map.count() == 0);
}

test "std.hash_map repeat fetchRemove" {
    // if (true) return error.SkipZigTest;
    var map = AutoHashMapUnmanaged(u64, void){};
    defer map.deinit(testing.allocator);

    try map.ensureTotalCapacity(testing.allocator, 4);

    map.putAssumeCapacity(0, {});
    map.putAssumeCapacity(1, {});
    map.putAssumeCapacity(2, {});
    map.putAssumeCapacity(3, {});

    // fetchRemove() should make slots available.
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        try testing.expect(map.fetchRemove(3) != null);
        map.putAssumeCapacity(3, {});
    }

    try testing.expect(map.get(0) != null);
    try testing.expect(map.get(1) != null);
    try testing.expect(map.get(2) != null);
    try testing.expect(map.get(3) != null);
}
