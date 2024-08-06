import Pkg
Pkg.activate(".")
using DSP
using Plots
using AbstractFFTs


EIIP_NUCLEOTIDE = Dict{String,Float64}([
    ("A", 0.1260),
    ("G", 0.0806),
    ("T", 0.1335),
    ("C", 0.1340)])

EIIP_AMINOACID = Dict{Char,Float64}([
    ('L', 0.0000),
    ('I', 0.0000),
    ('N', 0.0036),
    ('G', 0.0050),
    ('V', 0.0057),
    ('E', 0.0058),
    ('P', 0.0198),
    ('H', 0.0242),
    ('K', 0.0371),
    ('A', 0.0373),
    ('Y', 0.0516),
    ('W', 0.0548),
    ('Q', 0.0761),
    ('M', 0.0823),
    ('S', 0.0829),
    ('C', 0.0829),
    ('T', 0.0941),
    ('F', 0.0946),
    ('R', 0.0959),
    ('D', 0.1263)])

seq1::String = "PALPEDGGSGAFPPGHFKDPKRLYCKNGGFFLRIHPDGRVDGVREKSDPHIKLQLQAEERGWSIKGVCANRYLAMKEDGRLLASKCVTDECFFFERLESNNYNTYRSRKYSSWYVALKRTGQYKLGPKTGPGQKAILFLPMSAKS"
seq2::String = "FNLPLGNYKKPKLLYCSNGGYFLRILPDGTVDGTKDRSDQHIQLQLCAESIGEVYIKSTETGQFLAMDTDGLLYGSQTPNEECLFLERLEENHYNTYISKKHAEKHWFVGLKKNGRSKLGPRTHFGQKAILFLPLPVSSD"


function readSequence(seqpar::AbstractString)::Array{Float64}
    arrSeq = Float64[]
    for key in seqpar
        push!(arrSeq, EIIP_AMINOACID[key])
    end
    return arrSeq
end

eiipArray1::Array{Float64} = readSequence(seq1)
eiipArray2::Array{Float64} = readSequence(seq2)

convres = conv(eiipArray1, eiipArray2)
dftv = rfft(convres)
deleteat!(dftv, 1)
dftfreq = rfftfreq(length(convres) - 1)



plt = plot(dftfreq, abs.(dftv), xlabel="Frequency (Hz)", ylabel="Magnitude", title="FFT of the Signal")
savefig(plt, "myplot.pdf")