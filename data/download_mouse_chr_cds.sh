#!/bin/bash

if [ "$#" -ne 1 ]; then
	echo "Pass the chromosome number (e.g. 1, 2, X, Y)"
	exit 1
fi

num=$1

mkdir mouse_chr_$num
cd mouse_chr_$num

echo "Downloading chromosome $num FASTA for Mus musculus..."

wget https://ftp.ensembl.org/pub/current_fasta/mus_musculus/dna/Mus_musculus.GRCm39.dna.chromosome.$num.fa.gz

echo Extracting the FASTA file...
gunzip Mus_musculus.GRCm39.dna.chromosome.$num.fa.gz

echo "Downloading GTF annotation..."
wget https://ftp.ensembl.org/pub/current_gtf/mus_musculus/Mus_musculus.GRCm39.115.gtf.gz

echo Extracting the GTF file...
gunzip Mus_musculus.GRCm39.115.gtf.gz

echo Generating the CDS annotation for chromosome $num...

grep -P "^$num\t" Mus_musculus.GRCm39.115.gtf | grep -w "CDS" > chr${num}_CDS.gtf

rm Mus_musculus.GRCm39.115.gtf

echo "Done!"