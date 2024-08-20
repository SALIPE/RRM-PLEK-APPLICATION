include("dataIO.jl")

using DSP, Plots, AbstractFFTs, FASTX
using .DataIO

seq1::String = "PALPEDGGSGAFPPGHFKDPKRLYCKNGGFFLRIHPDGRVDGVREKSDPHIKLQLQAEERGWSIKGVCANRYLAMKEDGRLLASKCVTDECFFFERLESNNYNTYRSRKYSSWYVALKRTGQYKLGPKTGPGQKAILFLPMSAKS"
seq2::String = "FNLPLGNYKKPKLLYCSNGGYFLRILPDGTVDGTKDRSDQHIQLQLCAESIGEVYIKSTETGQFLAMDTDGLLYGSQTPNEECLFLERLEENHYNTYISKKHAEKHWFVGLKKNGRSKLGPRTHFGQKAILFLPLPVSSD"

filePath::String = ""


genomes = DataIO.getSequencesFromFastaFile(filePath)

let
    next = iterate(genomes)
    convsig = nothing
    while next !== nothing
        (i, state) = next
        # body
        if isnothing(convsig)
            convsig = DataIO.sequence2NumericalSerie(i)
        else
            convsig = conv(convsig, DataIO.sequence2NumericalSerie(i))
        end
        next = nothing
        # next = iterate(genomes, state)
    end
    dftv = rfft(convsig)

    # seqLen = length(convsig)
    # less::Int8 = seqLen % 2 == 0 ? 1 : 2

    dftfreq = rfftfreq(2000)
    plt = plot(dftfreq, abs.(dftv[2:1002]), xlabel="Frequency (Hz)", ylabel="Magnitude", title="FFT of the Signal")
    savefig(plt, "myplotconv.png")
end
