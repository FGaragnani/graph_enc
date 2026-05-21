#!/bin/bash

if [ "$#" -ne 1 ]; then
	echo "Usage: $0 <chr | start-end>"
	echo "Example: $0 1"
	echo "Example: $0 1-10"
	echo "Example: $0 X"
	exit 1
fi

INPUT=$1

# Mouse genome settings
SPECIES="mus_musculus"
ASSEMBLY="GRCm39"
GTF_VERSION="115"

FASTA_BASE="https://ftp.ensembl.org/pub/current_fasta/${SPECIES}/dna"
GTF_URL="https://ftp.ensembl.org/pub/current_gtf/${SPECIES}/${SPECIES^}.${ASSEMBLY}.${GTF_VERSION}.gtf.gz"

process_chr () {
	chr=$1

	echo "Processing chromosome $chr..."

	mkdir -p mouse_chr_$chr
	cd mouse_chr_$chr || exit

	# FASTA
	FASTA_FILE="Mus_musculus.${ASSEMBLY}.dna.chromosome.${chr}.fa.gz"
	wget ${FASTA_BASE}/${FASTA_FILE}

	echo "Extracting FASTA..."
	gunzip ${FASTA_FILE}

	# GTF (download once per chr folder for simplicity)
	wget ${GTF_URL}

	echo "Extracting GTF..."
	GTF_FILE="${SPECIES^}.${ASSEMBLY}.${GTF_VERSION}.gtf.gz"
	gunzip ${GTF_FILE}

	echo "Filtering CDS for chr $chr..."
	grep -P "^${chr}\t" ${SPECIES^}.${ASSEMBLY}.${GTF_VERSION}.gtf | grep -w "CDS" > mouse_chr${chr}_CDS.gtf

	rm ${SPECIES^}.${ASSEMBLY}.${GTF_VERSION}.gtf

	cd ..
	echo "Done chr $chr"
}

# Parse input
if [[ $INPUT =~ ^[0-9]+-[0-9]+$ ]]; then
	START=${INPUT%-*}
	END=${INPUT#*-}

	for ((i=START; i<=END; i++)); do
		process_chr $i
	done

elif [[ $INPUT =~ ^[0-9]+$ ]] || [[ $INPUT == "X" ]] || [[ $INPUT == "Y" ]] || [[ $INPUT == "MT" ]]; then
	process_chr $INPUT

else
	echo "Invalid input format"
	exit 1
fi