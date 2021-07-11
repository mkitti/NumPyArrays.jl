"""
    NumPyArrays extends PyCall to provide for additional conversion of Julia
    arrays into NumPy arrays without copying.

```jldoctest
julia> using NumPyArrays, PyCall

julia> rA = reinterpret(UInt8, zeros(Int8, 4,4));

julia> pytypeof(PyObject(rA))
PyObject <class 'list'>

julia> pytypeof(NumPyArray(rA))
PyObject <class 'numpy.ndarray'>

julia> pytypeof(PyObject(NumPyArray(rA)))
PyObject <class 'numpy.ndarray'>

julia> sA = @view collect(1:16)[5:9];

julia> pytypeof(PyObject(sA))
PyObject <class 'list'>

julia> pytypeof(NumPyArray(sA))
PyObject <class 'numpy.ndarray'>
```
"""
module NumPyArrays

# Portions of this code are derived directly from PyCall.jl

@static if isdefined(Base, :Experimental) &&
           isdefined(Base.Experimental, Symbol("@optlevel"))
    Base.Experimental.@optlevel 1
end


export NumPyArray, pytypeof

# Imports for _NumPyArray
import PyCall: NpyArray, PYARR_TYPES, @npyinitialize, npy_api, npy_type
import PyCall: @pycheck, NPY_ARRAY_ALIGNED, NPY_ARRAY_WRITEABLE, pyembed
import PyCall: PyObject, PyPtr

# General imports
import PyCall: pyimport, pytype_query, pytypeof, PyArray

"""
    KnownImmutableArraysWithParent{T} where T <: PyCall.PYARR_TYPES

Immutable `AbstractArray`s in `Base` that have a non-immutable parent that can be embedded in the PyCall GC
"""
const KnownImmutableArraysWithParent{T} = Union{SubArray{T}, Base.ReinterpretArray{T}, Base.ReshapedArray{T}, Base.PermutedDimsArray{T}} where T

"""
    KnownStridedArrays{T} where T <: PyCall.PYARR_TYPES

`AbstractArray`s in `Base` where the method `strides` is applicable
"""
const KnownStridedArrays{T} = StridedArray{T} where T

"""
    NumPyArray{T,N}(po::PyObject)

NumPyArray is a wrapper around a PyCall.PyArray. It is an AbstractArray.
The main purpose of a NumPyArray is so to provide a constructor to generalize
the conversion of Julia arrays into NumPyArrays. `T` is the element type of the array.
`N` is the number of dimensions.

For other uses, such as wrapping an existing array from NumPy, use `PyCall.PyArray`.

Use `PyObject` and `PyArray` methods to convert `NumPyArray` into those types.
"""
mutable struct NumPyArray{T,N} <: AbstractArray{T,N}
    pa::PyArray{T,N}
end
NumPyArray{T,N}(po::PyObject) where {T,N} = NumPyArray{T,N}(PyArray(po))
NumPyArray(po::PyObject) = NumPyArray(PyArray(po))

"""
    NumPyArray(a::AbstractArray, [revdims::Bool])

Convert an AbstractArray where `isapplicable(strides, a)` is `true` to a NumPy array.
The AbstractArray must either be mutable or have a mutable parent. Optionally,
transpose the dimensions of the array if `revdims` is `true`.
"""
NumPyArray(a::AbstractArray{T}) where T <: PYARR_TYPES = NumPyArray(a, false)
function NumPyArray(a::KnownStridedArrays{T}, revdims::Bool) where T <: PYARR_TYPES
    _NumPyArray(a, revdims)
end
function NumPyArray(a::AbstractArray{T}, revdims::Bool) where T <: PYARR_TYPES
    # For a general AbstractArray, we do not know if strides applies
    if applicable(strides, a)
        _NumPyArray(a, revdims)
    else
        error("Only AbstractArrays where strides is applicable can be converted to NumPyArrays.")
    end
end
NumPyArray(o::PyPtr) = NumPyArray(PyObject(o))

# Modified PyCall.NpyArray to accept AbstractArray{T}, assumes strides is applicable
function _NumPyArray(a::AbstractArray{T}, revdims::Bool) where T <: PYARR_TYPES
    @npyinitialize
    size_a = revdims ? reverse(size(a)) : size(a)
    strides_a = revdims ? reverse(strides(a)) : strides(a)
    p = @pycheck ccall(npy_api[:PyArray_New], PyPtr,
        (PyPtr,Cint,Ptr{Int},Cint, Ptr{Int},Ptr{T}, Cint,Cint,PyPtr),
        npy_api[:PyArray_Type],
        ndims(a), Int[size_a...], npy_type(T),
        Int[strides_a...] * sizeof(eltype(a)), a, sizeof(eltype(a)),
        NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE,
        C_NULL)
    return NumPyArray{T,ndims(a)}(p, a)
end

# Make a NumPyArray that embeds a reference to keep, to prevent Julia
# from garbage-collecting keep until o is finalized.
# See also PyObject(o::PyPtr, keep::Any) from which this is derived
NumPyArray{T,N}(o::PyPtr, keep::Any) where {T,N} = numpyembed(NumPyArray{T,N}(PyObject(o)), keep)

# PyCall already has convert(::Type{PyObject}, o) = PyObject(o)
#Base.convert(::Type{PyObject}, a::NumPyArray) = a.po
PyObject(a::NumPyArray) = PyObject(PyArray(a))
Base.convert(::Type{PyArray}, a::NumPyArray) = PyArray(a)
PyArray(a::NumPyArray) = a.pa
Base.convert(::Type{Array}, a::NumPyArray{T}) where T = convert(Array{T}, PyArray(a))
Base.convert(T::Type{<:Array}, a::NumPyArray) = convert(T, PyArray(a))

# See PyCall.pyembed(po::PyObject, jo::Any)
function numpyembed(a::NumPyArray{T,N}, jo::Any) where {T,N}
    if isimmutable(jo)
        if applicable(parent, jo)
            return NumPyArray{T,N}(pyembed(PyObject(a), parent(jo)))
        else
            throw(ArgumentError("numpyembed: immutable argument without a parent is not allowed"))
        end
    else
        return NumPyArray{T,N}(pyembed(PyObject(a), jo))
    end
end
numpyembed(a::NumPyArray, jo::KnownImmutableArraysWithParent) = numpyembed(a, jo.parent)

# AbstractArray interface, provided as a convenience. Conversion to PyArray is recommended
Base.size(a::NumPyArray) = size(PyArray(a))
Base.length(a::NumPyArray) = length(PyArray(a))
Base.getindex(a::NumPyArray, args...) = getindex(PyArray(a), args...)
Base.setindex!(a::NumPyArray, args...) = setindex(PyArray(a), args...)
Base.axes(a::NumPyArray) = axes(PyArray(a))
# Strides should be added to PyArray
Base.strides(a::NumPyArray{T}) where T = PyArray(a).st
Base.stride(a::NumPyArray{T}) where T = stride(PyArray(a))

Base.pointer(a::NumPyArray, args...) = pointer(PyArray(a), args...)
Base.unsafe_convert(t::Type{Ptr{T}}, a::NumPyArray{T}) where T = Base.unsafe_convert(t, PyArray(a))

Base.similar(a::NumPyArray, ::Type{T}, dims::Dims) where {T} = similar(PyArray(a), T, dims)

# Aliasing some PyCall functions. Conversion to PyObject or PyArray is recommended
pytypeof(a::NumPyArray) = pytypeof(PyObject(PyArray(a)))

end # module NumPyArrays
