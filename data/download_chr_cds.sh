#!/bin/bash

if [ "$#" -ne 1 ]; then
	echo "Pass the chromosome number"
	exit
fi

num=$1

mkdir human_chr_$num
cd human_chr_$num

wget https://ftp.ensembl.org/pub/current_fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.chromosome.$num.fa.gz
echo Extracting the FASTA file...
gunzip Homo_sapiens.GRCh38.dna.chromosome.$num.fa.gz

wget https://ftp.ensembl.org/pub/current_gtf/homo_sapiens/Homo_sapiens.GRCh38.115.gtf.gz
echo Extracting the CDS file...
gunzip Homo_sapiens.GRCh38.115.gtf.gz

echo Generating the CDS sequence for chromosome $num...
grep -P "^$num\t" Homo_sapiens.GRCh38.115.gtf | grep -w "CDS" > chr${num}_CDS.gtf

rm Homo_sapiens.GRCh38.115.gtf
echo Done!
