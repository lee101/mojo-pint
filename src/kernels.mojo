"""Numeric kernels for bulk unit conversion and dimensional-vector algebra."""

from std.sys import simd_width_of

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime BPtr = Pointer[UInt8, AnyOrigin[mut=True]]
comptime NDIM = 7


def convert_f64_range(
    src: FPtr,
    dst: FPtr,
    start: Int,
    end: Int,
    scale: Float64,
    shift: Float64,
):
    comptime W = simd_width_of[DType.float64]()
    var i = start
    var vs = SIMD[DType.float64, W](scale)
    if shift == 0.0:
        while i + W <= end:
            dst.unsafe_store(i, src.unsafe_load[width=W](i) * vs)
            i += W
        while i < end:
            dst.unsafe_store(i, src.unsafe_load(i) * scale)
            i += 1
    else:
        var vb = SIMD[DType.float64, W](shift)
        while i + W <= end:
            dst.unsafe_store(i, src.unsafe_load[width=W](i) * vs + vb)
            i += W
        while i < end:
            dst.unsafe_store(i, src.unsafe_load(i) * scale + shift)
            i += 1


def convert_f64(src: FPtr, dst: FPtr, n: Int, scale: Float64, shift: Float64):
    convert_f64_range(src, dst, 0, n, scale, shift)


def add_converted_f64(
    lhs: FPtr,
    rhs: FPtr,
    dst: FPtr,
    n: Int,
    scale: Float64,
    shift: Float64,
    sign: Float64,
):
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    var vs = SIMD[DType.float64, W](scale)
    var vb = SIMD[DType.float64, W](shift)
    var vo = SIMD[DType.float64, W](sign)
    while i + W <= n:
        dst.unsafe_store(
            i,
            lhs.unsafe_load[width=W](i)
            + vo * (rhs.unsafe_load[width=W](i) * vs + vb),
        )
        i += W
    while i < n:
        dst.unsafe_store(
            i,
            lhs.unsafe_load(i) + sign * (rhs.unsafe_load(i) * scale + shift),
        )
        i += 1


def combine_dimensions(
    lhs: FPtr, rhs: FPtr, dst: FPtr, rows: Int, rhs_sign: Float64
):
    for row in range(rows):
        var base = row * NDIM
        for dim in range(NDIM):
            dst.unsafe_store(
                base + dim,
                lhs.unsafe_load(base + dim)
                + rhs_sign * rhs.unsafe_load(base + dim),
            )


def dimensions_equal_many(
    lhs: FPtr, rhs: FPtr, dst: BPtr, rows: Int, tolerance: Float64
):
    for row in range(rows):
        var same = True
        var base = row * NDIM
        for dim in range(NDIM):
            if (
                abs(
                    lhs.unsafe_load(base + dim)
                    - rhs.unsafe_load(base + dim)
                )
                > tolerance
            ):
                same = False
        dst.unsafe_store(row, UInt8(1 if same else 0))


@export("mp_convert_f64")
def mp_convert_f64(
    src_addr: Int,
    dst_addr: Int,
    n: Int,
    scale: Float64,
    shift: Float64,
) abi("C"):
    if n <= 0 or src_addr == 0 or dst_addr == 0:
        return
    convert_f64(
        FPtr(unsafe_from_address=src_addr),
        FPtr(unsafe_from_address=dst_addr),
        n,
        scale,
        shift,
    )


@export("mp_convert_f64_serial")
def mp_convert_f64_serial(
    src_addr: Int,
    dst_addr: Int,
    n: Int,
    scale: Float64,
    shift: Float64,
) abi("C"):
    if n <= 0 or src_addr == 0 or dst_addr == 0:
        return
    convert_f64_range(
        FPtr(unsafe_from_address=src_addr),
        FPtr(unsafe_from_address=dst_addr),
        0,
        n,
        scale,
        shift,
    )


@export("mp_add_converted_f64")
def mp_add_converted_f64(
    lhs_addr: Int,
    rhs_addr: Int,
    dst_addr: Int,
    n: Int,
    scale: Float64,
    shift: Float64,
    sign: Float64,
) abi("C"):
    if n <= 0 or lhs_addr == 0 or rhs_addr == 0 or dst_addr == 0:
        return
    add_converted_f64(
        FPtr(unsafe_from_address=lhs_addr),
        FPtr(unsafe_from_address=rhs_addr),
        FPtr(unsafe_from_address=dst_addr),
        n,
        scale,
        shift,
        sign,
    )


@export("mp_combine_dimensions")
def mp_combine_dimensions(
    lhs_addr: Int,
    rhs_addr: Int,
    dst_addr: Int,
    rows: Int,
    rhs_sign: Float64,
) abi("C"):
    if rows <= 0 or lhs_addr == 0 or rhs_addr == 0 or dst_addr == 0:
        return
    combine_dimensions(
        FPtr(unsafe_from_address=lhs_addr),
        FPtr(unsafe_from_address=rhs_addr),
        FPtr(unsafe_from_address=dst_addr),
        rows,
        rhs_sign,
    )


@export("mp_dimensions_equal_many")
def mp_dimensions_equal_many(
    lhs_addr: Int,
    rhs_addr: Int,
    dst_addr: Int,
    rows: Int,
    tolerance: Float64,
) abi("C"):
    if rows <= 0 or lhs_addr == 0 or rhs_addr == 0 or dst_addr == 0:
        return
    dimensions_equal_many(
        FPtr(unsafe_from_address=lhs_addr),
        FPtr(unsafe_from_address=rhs_addr),
        BPtr(unsafe_from_address=dst_addr),
        rows,
        tolerance,
    )
