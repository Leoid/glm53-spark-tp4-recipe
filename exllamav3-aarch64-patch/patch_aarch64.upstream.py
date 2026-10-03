#!/usr/bin/env python3
"""Make exllamav3 1.4.5's C++/CUDA extension build on aarch64 (GB10 / DGX Spark).

x86-only pieces (AVX all-reduce, AVX CPU MoE offload for the mul1 codebook, x86 pause
intrinsics) are wrapped in architecture guards with throwing stubs. None of these paths are used
for GPU quantization or GPU inference.  Idempotent.
"""
import re, sys, os

root = sys.argv[1] if len(sys.argv) > 1 else "exllamav3/exllamav3_ext"
X86 = "defined(__x86_64__) || defined(__i386__)"
PAUSE_MACRO = """
#if defined(__x86_64__) || defined(__i386__)
#define EXL3_CPU_PAUSE() __builtin_ia32_pause()
#else
#define EXL3_CPU_PAUSE() __asm__ __volatile__("yield" ::: "memory")
#endif
"""

def wrap(path, stub_body):
    s = open(path).read()
    if "EXL3_AARCH64_STUB" in s:
        print("already:", path); return
    s = f"#if {X86}\n" + s + f"\n#else  /* EXL3_AARCH64_STUB */\n{stub_body}\n#endif\n"
    open(path, "w").write(s); print("wrapped:", path)

def decls_from_header(path):
    """Extract non-inline, non-template free-function declarations `ret name(args);`."""
    s = open(path).read()
    s = re.sub(r"//.*", "", s)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    out = []
    for m in re.finditer(r"^\s*((?:[A-Za-z_][\w:<>\*&\s]*?)\s+\**\s*([A-Za-z_]\w*)\s*\(([^;{}]*?)\)\s*;)", s, flags=re.M):
        full, name, args = m.group(1), m.group(2), m.group(3)
        head = full[: full.index(name)].strip()
        if head.startswith(("inline", "template", "static", "typedef", "return", "struct", "class")):
            continue
        if not head:
            continue
        out.append((head, name, " ".join(args.split())))
    return out

def stubs(header, exclude=()):
    lines = [f'#include "{os.path.basename(header)}"', "#include <stdexcept>"]
    for ret, name, args in decls_from_header(header):
        if name in exclude:
            continue
        body = 'throw std::runtime_error("' + name + ' unavailable on aarch64");'
        lines.append(f"{ret} {name}({args}) {{ {body} }}")
    return "\n".join(lines)

# 1. AVX all-reduce implementations
h512 = os.path.join(root, "parallel/all_reduce_cpu_avx512.h")
h2 = os.path.join(root, "parallel/all_reduce_cpu_avx2.h")
wrap(os.path.join(root, "parallel/all_reduce_cpu_avx512.cpp"),
     stubs(h512, exclude=("is_avx512_supported",)))
wrap(os.path.join(root, "parallel/all_reduce_cpu_avx2.cpp"),
     '#include "all_reduce_cpu_avx512.h"\n' + stubs(h2))

# 2. CPU MoE (mul1) offload kernels
hm = os.path.join(root, "cpu/moe_mul1.h")
wrap(os.path.join(root, "cpu/moe_mul1.cpp"), stubs(hm))

# 3. x86 pause intrinsic in host code of .cu files
for f in ("parallel/all_reduce_cpu.cu", "cpu/moe_handoff.cu"):
    p = os.path.join(root, f)
    s = open(p).read()
    if "EXL3_CPU_PAUSE" in s:
        print("already:", p); continue
    s = s.replace("__builtin_ia32_pause()", "EXL3_CPU_PAUSE()")
    # insert macro after the last #include at file top
    idx = 0
    for m in re.finditer(r"^#include.*$", s, flags=re.M):
        idx = m.end()
    s = s[:idx] + "\n" + PAUSE_MACRO + s[idx:]
    open(p, "w").write(s); print("pause-patched:", p)
print("stub decls:", [n for _, n, _ in decls_from_header(hm)])
