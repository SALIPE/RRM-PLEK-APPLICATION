include("dataIO.jl")

using DSP, Plots, AbstractFFTs
using .DataIO

seq1::String = "PALPEDGGSGAFPPGHFKDPKRLYCKNGGFFLRIHPDGRVDGVREKSDPHIKLQLQAEERGWSIKGVCANRYLAMKEDGRLLASKCVTDECFFFERLESNNYNTYRSRKYSSWYVALKRTGQYKLGPKTGPGQKAILFLPMSAKS"
seq2::String = "FNLPLGNYKKPKLLYCSNGGYFLRILPDGTVDGTKDRSDQHIQLQLCAESIGEVYIKSTETGQFLAMDTDGLLYGSQTPNEECLFLERLEENHYNTYISKKHAEKHWFVGLKKNGRSKLGPRTHFGQKAILFLPLPVSSD"

filePath::String = "/home/salipe/Documents/GitHub/datasets/dron_car_2737_selected_final.fa"


sequences = DataIO.getSequencesFromFastaFile(filePath)
numericalSeries = DataIO.sequenceList2NumericalSeries(sequences)


function longest_sequence_index(
    sequences::Array{Array{Float64}}
)::Int
    if isempty(sequences)
        error("Input array is empty.")
    end

    longest_index = 1
    longest_length = length(sequences[1])

    for i in 2:length(sequences)
        if length(sequences[i]) > longest_length
            longest_index = i
            longest_length = length(sequences[i])
        end
    end

    return longest_index
end

# longSeqIdx::Int = longest_sequence_index(numericalSeries)
longSeqIdx::Int = 1
global convres::AbstractArray = numericalSeries[longSeqIdx]

# for idx in 1:10
#     if (idx != longSeqIdx)
#         global convres
#         convres = conv(numericalSeries[idx], convres)
#     end

# end

dftv = rfft(convres)
# deleteat!(dftv, 1:3)
# less::Int8 = length(convres) % 2 == 0 ? 4 : 5
# dftfreq = rfftfreq(length(convres) - 6)
plt = plot(abs.(dftv[200:500]), xlabel="Frequency (Hz)", ylabel="Magnitude", title="FFT of the Signal")
savefig(plt, "myplot.png")