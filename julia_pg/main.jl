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
function sameSizeConv!(N::Vector{T}, K::Vector{T}) where {T<:Real}
    # Optimized for the case the kernel is in N (Shorter)
    length(N) < length(K) && return sameSizeConv!(K, N)
    # https://brianmcfee.net/dstbook-site/content/ch10-convtheorem/ConvolutionTheorem.html
    for ii in eachindex(N)
        sumVal = zero(T)
        for kk ∈ eachindex(K)
            oa = (ii >= kk) ? N[ii-kk+1] : zero(T) # N[ii-kk+N]
            prev = sumVal + (K[kk] * oa)
            (prev === Nan || prev === Inf) ? break : sumVal = prev
        end
        N[ii] = sumVal
    end
    return N
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
            convsig = sameSizeConv!(convsig, numSeries)
        end

    end

    println(length(convsig))
end


# seqLen = length(convsig)
# less::Int8 = seqLen % 2 == 0 ? 1 : 2

# dftfreq = rfftfreq(2000)
# plt = plot(dftfreq, abs.(dftv[1000:2000]), xlabel="Frequency (Hz)", ylabel="Magnitude", title="FFT of the Signal")
# savefig(plt, "myplotconv2.png")

