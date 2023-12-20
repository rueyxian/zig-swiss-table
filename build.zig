const std = @import("std");
const fmt = std.fmt;

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    _ = b.addModule("swiss_table", .{ .source_file = .{ .path = "src/swiss_table.zig" } });

    const mod_swiss_table = b.createModule(.{ .source_file = .{ .path = "src/swiss_table.zig" } });

    {
        const step_test = b.step("test", "Run unit tests");
        const cmpl = b.addTest(.{
            .root_source_file = .{ .path = "src/swiss_table.zig" },
            .target = target,
            .optimize = optimize,
        });
        const run_atf = b.addRunArtifact(cmpl);
        step_test.dependOn(&run_atf.step);
    }

    const step_build_all = b.step("build_all_demo", "Build all demo");
    const step_run_all = b.step("run_all_demo", "Run all demo");
    for ([_]struct {
        name: []const u8,
        path: []const u8,
    }{
        .{ .name = "unmanaged", .path = "src/demo/unmanaged.zig" },
        .{ .name = "auto", .path = "src/demo/auto.zig" },
        .{ .name = "string", .path = "src/demo/string.zig" },
    }) |opt| {
        const step_run = blk: {
            const name = try fmt.allocPrint(b.allocator, "run_demo_{s}", .{opt.name});
            const desciption = try fmt.allocPrint(b.allocator, "Run `{s}`", .{opt.path});
            break :blk b.step(name, desciption);
        };
        const step_build = blk: {
            const name = try fmt.allocPrint(b.allocator, "build_demo_{s}", .{opt.name});
            const description = try fmt.allocPrint(b.allocator, "Build `{s}`", .{opt.path});
            break :blk b.step(name, description);
        };
        const cmpl = blk: {
            const cmpl = b.addExecutable(.{
                .name = opt.name,
                .root_source_file = .{ .path = opt.path },
                .target = target,
                .optimize = optimize,
            });
            cmpl.addModule("swiss_table", mod_swiss_table);
            break :blk cmpl;
        };
        const run_atf = b.addRunArtifact(cmpl);
        const build_atf = b.addInstallArtifact(cmpl, .{});
        step_run.dependOn(&run_atf.step);
        step_run_all.dependOn(&run_atf.step);
        step_build.dependOn(&build_atf.step);
        step_build_all.dependOn(&build_atf.step);
    }
}
