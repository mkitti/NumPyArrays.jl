# NumPyArrays.jl

NumPyArrays.jl is a Julia package that extends PyCall.jl in order to convert additional Julia arrays into NumPy arrays.

## Additional Features

NumPyArrays.jl also provides a [`AbstractArray` interface](https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-array)
and extends some functions of PyCall to apply to a `NumPyArray`. Much of this is redundant with the functionality of `PyCall.PyArray`, which this wraps.

For advanced usage with PyCall, it is recommended to convert the `NumPyArray` to a `PyObject` or `PyArray`.

## PyCall only converts some Julia arrays into a NumPy array

PyCall.jl already converts a Julia `Array` into a NumPy array.
However, PyCall converts a `SubArray`, `Base.ReinterpretArray`,
and `PermutedDimsArray` into a `list` even if their element
type is compatible with NumPy.

NumPyArrays.jl extends PyCall.jl to allow any array with a compatible
element type where the method `strides` is applicable and who has a
parent or ancestor that is mutable.

## Example and Demonstration

```julia
julia> using NumPyArrays, PyCall

julia> rA = reinterpret(UInt8, zeros(Int8, 4,4))
4×4 reinterpret(UInt8, ::Array{Int8,2}):
 0x00  0x00  0x00  0x00
 0x00  0x00  0x00  0x00
 0x00  0x00  0x00  0x00
 0x00  0x00  0x00  0x00

julia> pytypeof(PyObject(rA))
PyObject <class 'list'>

julia> PyObject(rA)
PyObject [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

julia> pytypeof(NumPyArray(rA))
PyObject <class 'numpy.ndarray'>

julia> NumPyArray(rA)
4×4 NumPyArray{UInt8,2}:
 0x00  0x00  0x00  0x00
 0x00  0x00  0x00  0x00
 0x00  0x00  0x00  0x00
 0x00  0x00  0x00  0x00

julia> PyObject(NumPyArray(rA))
PyObject array([[0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]], dtype=uint8)

julia> sA = @view collect(1:16)[5:9]
5-element view(::Array{Int64,1}, 5:9) with eltype Int64:
 5
 6
 7
 8
 9

julia> pytypeof(PyObject(sA))
PyObject <class 'list'>

julia> PyObject(sA)
PyObject [5, 6, 7, 8, 9]

julia> pytypeof(NumPyArray(sA))
PyObject <class 'numpy.ndarray'>

julia> npsA = NumPyArray(sA)
5-element NumPyArray{Int64,1} with indices 0:4:
 5
 6
 7
 8
 9

julia> sum(npsA)
35

julia> np = pyimport("numpy"); np.sum(npsA)
35
```

## Questions

### Why not add this functionality to PyCall.jl?

There is a pending pull request on PyCall.jl to integrate this functionality.
See [PyCall.jl#876: Convert AbstractArrays with strides to NumPy arrays](https://github.com/JuliaPy/PyCall.jl/pull/876).
As of the creation of this package on July 10th, 2021, the pull request was last reviewed six months ago on January 13th, 2021.

### Should I use NumPyArray or PyCall.PyArray to wrap arrays from Python?

You should `PyCall.PyArray`. This package is primarily useful for converting certain Julia arrays into a `PyCall.PyArray`.

### Why not just extend PyObject / PyArray by adding methods to those types?

Since boith `PyObject` or `PyArray` are defined in PyCall.jl and not this package, adding methods to those types would be
[type piracy](https://docs.julialang.org/en/v1/manual/style-guide/#Avoid-type-piracy). We avoid type piracy in this package
by creating a new type `NumPyArray` which wraps `PyArray`.

### Should NumPyArrays.jl moved under the PyJulia organization?

Sure. Feel free to contact me.
