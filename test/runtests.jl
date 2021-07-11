using NumPyArrays
using PyCall
using Test

@testset "NumPyArrays.jl" begin
    let A = Float64[1 2; 3 4]
        # Normal array
        B = copy(A)
        C = NumPyArray(B)
        D = PyArray(C)
        @test collect(C) == B
        @test D == B
        B[1] = 3
        @test collect(C) == B && C[0] == B[1]
        @test D == B && D[1] == B[1]

        # SubArray
        B = view(A, 1:2, 2:2)
        C = NumPyArray(B)
        D = PyArray(C)
        @test collect(C) == B
        @test D == B
        A[3] = 5
        @test collect(C) == B && C[0] == A[3]
        @test D == B && D[1] == A[3]

        # ReshapedArray
        B = Base.ReshapedArray( A, (1,4), () )
        C = NumPyArray(B)
        D = PyArray(C)
        @test collect(C) == B
        @test D == B
        A[2] = 6
        @test collect(C) == B && C[1] == A[2]
        @test D == B && D[2] == A[2]

        # PermutedDimsArray
        B = PermutedDimsArray(A, (2,1) )
        C = NumPyArray(B)
        D = PyArray(C)
        @test collect(C) == B
        @test D == B
        A[1] == 7
        @test collect(C) == B && C[0] == A[1]
        @test D == B && D[1] == A[1]

        # ReinterpretArray
        B = reinterpret(UInt64, A)
        C = NumPyArray(B)
        D = PyArray(C)
        @test collect(C) == B
        @test D == B
        A[1] = 12
        @test collect(C) == B && C[0] == reinterpret(UInt64, A[1])
        @test D == B && D[1] == reinterpret(UInt64, A[1])
    end
end
