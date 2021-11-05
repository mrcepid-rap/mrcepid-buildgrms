# CollectHsMetrics (DNAnexus Platform App)

This is the source code for an app that runs on the DNAnexus Platform.
For more information about how to run or modify it, see
https://documentation.dnanexus.com/.

### Table of Contents

- [Introduction](#introduction)
    * [Background](#background)
    * [Dependencies](#dependencies)
        + [Docker](#docker)
        + [Resource Files](#resource-files)
- [Methodology](#methodology)
    * [1. Selecting Individuals](#1-selecting-individuals)
    * [2. Perform Genotyping QC](#2-perform-genotyping-qc)
    * [3. Perform Filtering](#3-perform-filtering)
    * [4. Make a GRM Compatible with SAIGE](#4-make-a-grm-compatible-with-saige)
- [Running on DNANexus](#running-on-dnanexus)
    * [Inputs](#inputs)
    * [Outputs](#outputs)
    * [Command line example](#command-line-example)
        + [Batch Running](#batch-running)

## Introduction

This applet performs several tasks related to the genotyping data provided by UKBiobank for the purposes of generating
genetic relatedness matrices (GRMs) during rare variant burden testing. This applet should only ever need to be run once.
However, I am documenting here how it is run and the filtering approaches applied for the purposes of transparency.

This README makes use of DNANexus file and project naming conventions. Where applicable, an object available on the DNANexus
platform has a hash ID like:

* file – `file-1234567890ABCDEFGHIJKLMN`
* project – `project-1234567890ABCDEFGHIJKLMN`

Information about files and projects can be queried using the `dx describe` tool native to the DNANexus SDK:

```commandline
dx describe file-1234567890ABCDEFGHIJKLMN
```

**Note:** This README pertains to data included as part of the DNANexus project "MRC - Variant Filtering" (project-G2XK5zjJXk83yZ598Z7BpGPk)

### Background

Most rare variant burden tools require genotyping data to generate GRMs to control for cryptic population structure. Here,
I have developed an applet that:

1. Gets genetic data into the required formats for various rare variant burden tools
2. Sets exclusion/inclusion lists for individuals based on genetic ancestry and/or relatedness 
3. Computes a GRM for the tool [SAIGE-GEN](https://github.com/weizhouUMICH/SAIGE/wiki/Genetic-association-tests-using-SAIGE)

### Dependencies

#### Docker

This applet uses [Docker](https://www.docker.com/) to supply dependencies to the underlying AWS instance
launched by DNANexus. The Dockerfile used to build dependencies is available as part of the MRCEpid organisation at:

https://github.com/mrcepid-rap/dockerimages/blob/main/associationtesting.Dockerfile

This Docker image is built off of a 20.04 Ubuntu distribution with miniconda3 pre-installed available via [dockerhub](https://hub.docker.com/r/continuumio/miniconda3).
This was done to remove issues with installing miniconda3 via Dockerfile which is a dependancy of CADD. This image is
very light-weight and only provides basic OS installation and miniconda3. Other basic software (e.g. wget, make, and gcc)
need to be installed manually. For more details on how to build a Docker image for use on the UKBiobank RAP, please see:

https://github.com/mrcepid-rap#docker-images

The only external tool that this applet requires from this Docker image is [plink2](https://www.cog-genomics.org/plink/2.0/).

This statement is not exhaustive and does not include dependencies of dependencies and software needed
to acquire other resources (e.g. wget). See the referenced Dockerfile for more information.

#### Resource Files

This app also makes use of several genetics resource files generated by UKBiobank with specific directories/files listed here:

* UKBiobank genotypes - `/Bulk/Genotype Results/Genotype calls/ukb22418_c*`
* UKBiobank pre-computed relatedness – `/Bulk/Genotype Results/Genotype calls/ukb_rel.dat`
* UKBiobank SNP QC – `/Bulk/Genotype Results/Genotype calls/ukb_snp_qc.txt`
* UKBiobank Sample QC - `/Bulk/Genotype Results/Genotype calls/ukb_sqc_v2.txt`
* UKBiobank ChrY WES VCF – `/Bulk/Exome sequences/Population level exome OQFE variants, pVCF format/ukb23156_c24_b0_v1.vcf.gz`
   * This is to get a quick list of individuals with WES data
* WBA Phenotype – `/project_resources/genetics/wba.txt`
   * This file was generated by Felix Day to represent a "better" collection of white European ancestry individuals

## Methodology

This applet is a supplementary applet (mrc-buildgrm) to the rare variant testing pipeline developed by Eugene Gardner for the UKBiobank
RAP at the MRC Epidemiology Unit:

![](https://github.com/mrcepid-rap/.github/blob/main/images/RAPPipeline.png)

This applet has several steps that are presented/annotated in more detail in the source code: `src/mrcepid-buildgrms.py`
Here I present crucial steps that could effect downstream analysis:

### 1. Selecting Individuals

The first step of this applet is to select individuals we want to retain for testing based on:

1. Presence in the whole exome sequencing (WES) data

We use the header of a WES VCF file and get a list of individuals to retain using:

```commandline
bcftools query -l ukb23156_c24_b0_v1.vcf.gz > wes_samples.txt
```

2. Relatedness to other participants
   
We use the relatedness file pre-computed by [Bycroft et al.](https://www.nature.com/articles/s41586-018-0579-z) provided
on the RAP as `ukb_rel.dat`. In brief, we:

+ remove individuals who do not have WES from this file
+ calculate the number of times a given individual occurs within the file 
+ Remove the individual with the most relatedness pairs.
+ Repeat until no relatedness pairs are left

3. White, European genetic ancestry

Felix Day from the MRC-Epidemiology Unit provided me with a list of individuals with white, European ancestry. This list
is more inclusive than that provided by UK Biobank as part of [field 22006](https://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=22006)
on the UK Biobank data showcase.

These files are then combined in various ways to make three exclusion files that are returned as part of this applet's 
outputs:

1. EXCLUDEFOR_White_Euro_Relateds.txt – non-Euro, related individuals
2. KEEPFOR_White_Euro.txt – white Euro genetic ancestry individuals
3. EXCLUDEFOR_Relateds.txt - related individuals

### 2. Perform Genotyping QC

We next perform quality control of SNPs and Samples in the main genotyping files provided by UK Biobank 
(e.g. `/Bulk/Genotype Results/Genotype calls/ukb22418_c*`).

We include SNPs based on the following parameters where the SNP is:

1. Included on both the standard and BiLEVE array
2. Autosomal
3. Missingness < 5%. Missingness is calculated using plink2 like:
```commandline
plink2 --missing 'variant-only' --pfile UKBB_500K_Autosomes --out UKBB_500K_Autosomes
```
4. Passes QC on all individual standard and BiLEVE arrays
5. Are not monomorphic

We include samples that:

1. Are included in the WES sample list
2. Is not in the heterozygous missingness outliers list
3. Is part of samples included in autosomal phasing

### 3. Perform Filtering

We then generate a filtered .bed file for rare variant association testing using plink:

```commandline
plink2 --mac 1 --pfile /test/UKBB_500K_Autosomes --make-bed \
        --extract /test/pass_snps.txt \ 
        --keep-fam /test/samp_pass_gt_qc.txt \
        --out /test/UKBB_200K_Autosomes_QCd
```

We also generate a list of low minor-allele count sites (MAC ≤ 100) to exclude when running BOLT here. 

### 4. Make a GRM Compatible with SAIGE

SAIGE allows for pre-computing a compatible GRM. We do that here using the following command line:

```commandline
createSparseGRM.R \
          --plinkFile=UKBB_200K_Autosomes_QCd \
          --nThreads=16 \
          --outputPrefix=sparseGRM_200K_Autosomes_QCd \
          --numRandomMarkerforSparseKin=2000 \
          --relatednessCutoff=0.125"
```

## Running on DNANexus

### Inputs

This applet has no inputs and all are required files are hardcoded by default.

### Outputs

All outputs have pre-determined names that cannot be changed on the command line. The file names are documented in the 
output table here:

|output                 | description       | file name |
|-----------------------|-------------------| --------- |
|output_pgen            | SNP/Sample-filtered autosomal plink format file from [step 3](#3-perform-filtering) above | `UKBB_200K_Autosomes_QCd.bed` |
|output_psam            | Associated sample file                                                                    | `UKBB_200K_Autosomes_QCd.fam` |
|output_pvar            | Associated variant file                                                                   | `UKBB_200K_Autosomes_QCd.bim` |
|wba_related_filter     | non-Euro, related list for exclusion from [step 1](#1-selecting-individuals) above        | `EXCLUDEFOR_White_Euro_Relateds.txt` |
|wba_filter             | Euro list for inclusion from [step 1](#1-selecting-individuals) above                     | `KEEPFOR_White_Euro.txt` |
|related_filter         | related list for exclusion from [step 1](#1-selecting-individuals) above                  | `EXCLUDEFOR_Relateds.txt` |
|grm                    | SAIGE-compatible GRM from [step4](#4-make-a-grm-compatible-with-saige) above              | `sparseGRM_200K_Autosomes_QCd_relatednessCutoff_0.125_2000_randomMarkersUsed.sparseGRM.mtx` |
|grm_samp               | Associated .sample file                                                                   | `sparseGRM_200K_Autosomes_QCd_relatednessCutoff_0.125_2000_randomMarkersUsed.sparseGRM.mtx.sampleIDs.txt` |
|snp_list               | List of low MAC SNPs to exclude from BOLT from [step3](#3-perform-filtering) above        | `UKBB_200K_Autosomes_QCd.low_MAC.snplist` |

### Command line example

If this is your first time running this applet within a project other than "MRC - Variant Filtering", please see our
organisational documentation on how to download and build this app on the DNANexus Research Access Platform:

https://github.com/mrcepid-rap

Running this command is straightforward using the DNANexus SDK toolkit as no inputs have to be provided on the command line:

```commandline
dx run mrcepid-buildgrms --priority low --destination project_resources/genetics/
```

Brief I/O information can also be retrieved on the command line:

```commandline
dx run mrcepid-buildgrms --help
```

Some notes here regarding execution:
1. Outputs are deposited into the folder named by `destination`. If this is left off of the command line, the tool will 
   deposit the resulting data into the top level directory of your project

2. I have set a sensible (and tested) default for compute resources on DNANexus that is baked into the json used for building the app (at `dxapp.json`)
   so setting an instance type is unnecessary. This current default is for a mem2_ssd1_v2_x16 instance (16 CPUs, 32 Gb RAM, 400Gb storage).
   If necessary to adjust compute resources, one can provide a flag like `--instance-type mem1_ssd1_v2_x36`.
   
#### Batch Running

This applet is not compatible with batch running.