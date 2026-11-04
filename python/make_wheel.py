#!/usr/bin/env python3
"""make_wheel.py VERSION PLATFORM_TAG -- dist/ozaki_blackwell-VERSION-py3-none-PLATFORM_TAG.whl from ozaki_blackwell/ (with the
built libozaki_blackwell.so and LICENSE) and README.md: the PEP 427 layout, a RECORD with sha256 hashes, fixed timestamps
(the same inputs give the same bytes). No setuptools: the library is built by nvcc beforehand (build.sh)."""
import base64
import hashlib
import os
import sys
import zipfile

NAME = "ozaki_blackwell"


def main():
    ver, plat = sys.argv[1], sys.argv[2]
    here = os.path.dirname(os.path.abspath(__file__))
    pkg = os.path.join(here, NAME)
    files = []
    for f in sorted(os.listdir(pkg)):
        if f.endswith((".py", ".so")) or f == "LICENSE":
            with open(os.path.join(pkg, f), "rb") as fh:
                files.append((f"{NAME}/{f}", fh.read()))
    if not any(p.endswith(".so") for p, _ in files):
        sys.exit("libozaki_blackwell.so is missing: build it first (build.sh)")
    with open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
        readme = fh.read()
    with open(os.path.join(pkg, "LICENSE"), "rb") as fh:
        license_text = fh.read()
    dist = f"{NAME}-{ver}.dist-info"
    meta = "\n".join([
        "Metadata-Version: 2.1",
        "Name: ozaki-blackwell",
        f"Version: {ver}",
        "Summary: FP64, double-double and triple-double matrix products from the INT8 tensor cores of NVIDIA GPUs (Ozaki scheme II)",
        "Home-page: https://git.spacecode.tech/shaban/ozaki-blackwell",
        "Author: Shaban Ejupi",
        "License: Ozaki-Blackwell Restricted Source License 1.0 -- free for private study and open academic research; other use needs a commercial license",
        "Project-URL: Source, https://git.spacecode.tech/shaban/ozaki-blackwell",
        "Project-URL: Mirror, https://github.com/ShabanEjupi/ozaki-blackwell",
        "Keywords: cuda,gemm,matmul,fp64,double-double,triple-double,tensor-cores,int8,high-precision",
        "Classifier: Development Status :: 3 - Alpha",
        "Classifier: Environment :: GPU :: NVIDIA CUDA :: 12",
        "Classifier: Intended Audience :: Science/Research",
        "Classifier: License :: Other/Proprietary License",
        "Classifier: Operating System :: POSIX :: Linux",
        "Classifier: Programming Language :: Python :: 3",
        "Classifier: Topic :: Scientific/Engineering :: Mathematics",
        "Requires-Python: >=3.8",
        "Requires-Dist: numpy",
        "Description-Content-Type: text/markdown",
        "",
        readme,
    ])
    wheel = f"Wheel-Version: 1.0\nGenerator: ozaki-blackwell make_wheel.py\nRoot-Is-Purelib: false\nTag: py3-none-{plat}\n"
    files += [(f"{dist}/METADATA", meta.encode()), (f"{dist}/WHEEL", wheel.encode()), (f"{dist}/LICENSE", license_text)]
    rec = []
    for path, data in files:
        h = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        rec.append(f"{path},sha256={h},{len(data)}")
    rec.append(f"{dist}/RECORD,,")
    files.append((f"{dist}/RECORD", ("\n".join(rec) + "\n").encode()))
    os.makedirs(os.path.join(here, "dist"), exist_ok=True)
    out = os.path.join(here, "dist", f"{NAME}-{ver}-py3-none-{plat}.whl")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for path, data in files:
            zi = zipfile.ZipInfo(path, date_time=(2026, 11, 4, 0, 0, 0))
            zi.external_attr = (0o755 if path.endswith(".so") else 0o644) << 16
            zi.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(zi, data)
    print(out, os.path.getsize(out), "bytes")


if __name__ == "__main__":
    main()
