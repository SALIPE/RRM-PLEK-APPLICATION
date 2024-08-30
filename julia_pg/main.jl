include("dataIO.jl")
include("transformUtils.jl")

using DSP, AbstractFFTs, FASTX, Plots, LoopVectorization, Normalization
using .DataIO, .TransformUtils

# seq1::String = "PALPEDGGSGAFPPGHFKDPKRLYCKNGGFFLRIHPDGRVDGVREKSDPHIKLQLQAEERGWSIKGVCANRYLAMKEDGRLLASKCVTDECFFFERLESNNYNTYRSRKYSSWYVALKRTGQYKLGPKTGPGQKAILFLPMSAKS"
# seq2::String = "FNLPLGNYKKPKLLYCSNGGYFLRILPDGTVDGTKDRSDQHIQLQLCAESIGEVYIKSTETGQFLAMDTDGLLYGSQTPNEECLFLERLEENHYNTYISKKHAEKHWFVGLKKNGRSKLGPRTHFGQKAILFLPLPVSSD"


function _separateSequences!(
    series::Array{Vector{Float64}}
)::Vector{Float64}

    minLength::Int = length(series[argmin(series)])

    toCross::Array{Vector{Float64}} = []
    toReprocess::Array{Vector{Float64}} = []
    for numSeries in series
        seqLen::Int = length(seq)
        if seqLen == minLength
            push!(toCross, numSeries)
        else
            push!(crossParts, numSeries[1:minLength])
            push!(toReprocess, numSeries[minLength:length(numSeries)])
        end
    end

    if length(toReprocess) > 1
        return _separateSequences!(toReprocess)
    end

    cross = TransformUtils.elementWiseMult(toCross)

end




let
    filePath::String = "/home/salipe/Documents/GitHub/datasets/gen_dron_car.fasta"
    sequences = open(FASTAReader, filePath)

    minLength = DataIO.getShortestLength(filePath)
    @show minLength
    powder = 256

    toCross::Array{Vector{Float64}} = []
    toReprocess::Array{Vector{Float64}} = []
    for seq in sequences
        seqLen::Int = seqsize(seq)
        numSeries = DataIO.sequence2NumericalSerie(sequence(seq))
        push!(toCross, numSeries[25000000:25000000+powder])
        # if seqLen == minLength
        #     push!(toCross, numSeries)
        # else
        #     push!(toCross, numSeries[1:minLength])
        #     push!(toReprocess, numSeries[minLength:length(numSeries)])
        # end
    end
    cross = TransformUtils.elementWiseMult(toCross, powder)
    N = MinMax(cross)
    cross = N(cross)
    # dftfreq = rfftfreq(minLength)
    plt = plot(cross, xlabel="Frequency (Hz)", ylabel="Magnitude", title="FFT of the Signal")
    savefig(plt, "myplotconv2.png")

    # if length(toReprocess) > 1
    #     return _separateSequences!(toReprocess)
    # end


end


# seqLen = length(convsig)
# less::Int8 = seqLen % 2 == 0 ? 1 : 2

# dftfreq = rfftfreq(2000)
# plt = plot(dftfreq, abs.(dftv[1000:2000]), xlabel="Frequency (Hz)", ylabel="Magnitude", title="FFT of the Signal")
# savefig(plt, "myplotconv2.png")

