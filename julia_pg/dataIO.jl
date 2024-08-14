module DataIO
using FASTX

export DataIO

EIIP_NUCLEOTIDE = Dict{Char,Float64}([
    ('A', 0.1260),
    ('G', 0.0806),
    ('T', 0.1335),
    ('C', 0.1340)])

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

function getSequencesFromFastaFile(
    filePath::String
)::Array{FASTX.FASTA.Record}
    sequences::Array{FASTX.FASTA.Record} = []
    FASTAReader(open(filePath)) do reader
        for record in reader
            push!(sequences, record)
            # println(identifier(record))
            # println(sequence(record))
            # println(description(record))
        end
    end

    return sequences
end

function sequence2NumericalSerie(
    seqpar::FASTX.FASTA.Record
)::Array{Float64}
    arrSeq = Float64[]
    for key in sequence(seqpar)
        if (key in keys(EIIP_NUCLEOTIDE))
            push!(arrSeq, EIIP_NUCLEOTIDE[key])
        end
    end
    return arrSeq
end

function sequenceList2NumericalSeries(
    seqlist::Array{FASTX.FASTA.Record}
)::Array{Array{Float64}}
    series::Array{Array{Float64}} = []
    for record in seqlist
        seq = sequence2NumericalSerie(record)
        if (!isempty(seq))
            push!(series, seq)
        end
    end
    return series
end

end