from Bio import SeqIO


def extract_regions(input_fasta, output_prefix, regions):
    """
    Extracts specific regions from a FASTA file and writes them to separate files.

    Parameters:
    - input_fasta: Path to the input FASTA file.
    - output_prefix: Prefix for the output files.
    - regions: List of tuples (header, start, end) specifying the regions to extract.
               `start` and `end` are 1-based inclusive coordinates.
    """

    # Parse the input FASTA file
    sequences = SeqIO.to_dict(SeqIO.parse(input_fasta, "fasta"))

    # Extract sequences
    for header in regions:
        if header in sequences:
            seq = sequences[header].seq
            # Create a new sequence record
            new_record = SeqIO.SeqRecord(seq, id=f"{header}_2724", description="")
            # Write to a new FASTA file
            output_file = f"{output_prefix}.fasta"
            with open(output_file, "w") as output_handle:
                SeqIO.write(new_record, output_handle, "fasta")

            print(f"Extracted region {header}")
        else:
            print(f"Header {header} not found in the input FASTA file.")
    # Extract regions
    # for header, start, end in regions:
    #     # Adjust for 0-based indexing
    #     start -= 1

    #     # Extract the sequence
    #     if header in sequences:
    #         seq = sequences[header].seq[start:end]
    #         # Create a new sequence record
    #         new_record = SeqIO.SeqRecord(seq, id=f"{header}_{start+1}_{end}", description="")
    #         # Write to a new FASTA file
    #         output_file = f"{output_prefix}.fasta"
    #         with open(output_file, "w") as output_handle:
    #             SeqIO.write(new_record, output_handle, "fasta")

    #         print(f"Extracted region {header}:{start+1}-{end}")
    #     else:
    #         print(f"Header {header} not found in the input FASTA file.")

    

# Example usage
input_fasta = ""
output_prefix = "gen_extract"
regions = [
"Group1"
]

extract_regions(input_fasta, output_prefix, regions)


