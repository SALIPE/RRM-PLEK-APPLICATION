include("dataIO.jl")

using DSP, Plots, AbstractFFTs, FASTX, LinearAlgebra, LoopVectorization, BenchmarkTools
using .DataIO

# seq1::String = "PALPEDGGSGAFPPGHFKDPKRLYCKNGGFFLRIHPDGRVDGVREKSDPHIKLQLQAEERGWSIKGVCANRYLAMKEDGRLLASKCVTDECFFFERLESNNYNTYRSRKYSSWYVALKRTGQYKLGPKTGPGQKAILFLPMSAKS"
# seq2::String = "FNLPLGNYKKPKLLYCSNGGYFLRILPDGTVDGTKDRSDQHIQLQLCAESIGEVYIKSTETGQFLAMDTDGLLYGSQTPNEECLFLERLEENHYNTYISKKHAEKHWFVGLKKNGRSKLGPRTHFGQKAILFLPLPVSSD"

# function crossSpectrumMinFFT(sequences, filePath)
#     sequenceLength::Int = DataIO.getShortestLength(filePath)
#     next = iterate(sequences)
#     crossEspectrum = nothing
#     while next !== nothing
#         (i, state) = next

#         numSerie::Vector{Float64} = DataIO.sequence2NumericalSerie(i, sequenceLength - 1)

#         dft = rfft(numSerie)
#         dft = deleteat!(abs.(dft), 1)
#         if isnothing(crossEspectrum)
#             crossEspectrum = dft
#         else
#             crossEspectrum = crossEspectrum .* dft
#         end
#         next = iterate(sequences, state)
#     end

#     dftfreq = rfftfreq(sequenceLength - 3)
#     plt = plot(dftfreq, crossEspectrum, xlabel="Frequency (Hz)", ylabel="Magnitude", title="FFT of the Signal")
#     savefig(plt, "myplotfft.png")
# end

# https://brianmcfee.net/dstbook-site/content/ch10-convtheorem/ConvolutionTheorem.html
function circConvt!(N::Vector{T}, K::Vector{T}, BS::Int) where {T<:Real}
    # Optimized for the case the kernel is in N (Shorter)
    lenN = length(N)
    lenN < length(K) && return circConvt!(K, N, BS)
    if lenN > BS
        error("BS must be >= the length of N")
    elseif length(K) > BS
        error("BS must be >= the length of K")
    end

    Y = Vector{T}(undef, BS)
    for ii in eachindex(Y)
        sumVal = zero(T)
        for kk ∈ eachindex(K)
            oa = (ii >= kk) ? N[ii-kk+1] : N[ii-kk+lenN]
            prev = sumVal + (K[kk] * oa)
            # sumVal = prev
            prev === Inf ? break : sumVal = prev
        end
        Y[ii] = sumVal
        N[ii] = sumVal
    end

    # for ii in eachindex(N)
    #     sumVal = zero(T)
    #     for kk ∈ eachindex(K)
    #         oa = (ii >= kk) ? N[ii-kk+1] : N[ii-kk+N]
    #         prev = sumVal + (K[kk] * oa)
    #         (prev === Nan || prev === Inf) ? break : sumVal = prev
    #     end
    #     N[ii] = sumVal
    # end
    return Y
end


function blockConv!(X::Vector{T}, H::Vector{T}) where {T<:Real}
    # Optimized for the case the kernel is in N (Shorter)
    Nx::Int = length(X)
    M::Int = length(H)
    Nx < M && return blockConv!(H, X)

    N = floor(Int, M + (M / 2))
    # number of zeros to pad
    M1::Int = M - 1
    # Block size
    L::Int = N - M1
    # Number of blocks
    K = floor(Int, (Nx + M1 - 1) / L)

    x::Vector{T} = reduce(vcat, [zeros(M1), X, zeros(N - 1)])

    Y = Matrix{T}(undef, 0, 0)

    @turbo for k in (0:K-1)
        xk = x[(k*L+1):(k*L+N)]
        Y[k, :] = circConvt!(xk, H, N)
    end
    Y = transpose(Y[:, M:N])
    return transpose(Y[:])

end


let
    filePath::String = "/home/salipe/Documents/GitHub/datasets/gen_dron_car.fasta"
    sequences = open(FASTAReader, filePath)

    convsig = nothing
    for seq in sequences
        numSeries = DataIO.sequence2NumericalSerie(sequence(seq))
        if isnothing(convsig)
            convsig = numSeries
        else
            println("block convolution")
            convsig = blockConv!(convsig, numSeries)
        end

    end

    println(length(convsig))
end


# seqLen = length(convsig)
# less::Int8 = seqLen % 2 == 0 ? 1 : 2

# dftfreq = rfftfreq(2000)
# plt = plot(dftfreq, abs.(dftv[1000:2000]), xlabel="Frequency (Hz)", ylabel="Magnitude", title="FFT of the Signal")
# savefig(plt, "myplotconv2.png")

