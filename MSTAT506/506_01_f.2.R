rm(list = ls())
getwd()
setwd("C:\\Users\\LENOVO\\Desktop\\assignment 506")
############################### 1 b data preparation

raw_counts <- read.delim("GSE58135_raw_counts_GRCh38.p13_NCBI.tsv.gz",
                         header = TRUE, row.names = 1, check.names = FALSE)
dim(raw_counts)
colnames(raw_counts)[1:5]
rownames(raw_counts)[1:5]

annot <- read.delim("Human.GRCh38.p13.annot.tsv.gz", header = TRUE)
dim(annot)
View(annot)
colnames(annot)

####### Replace gene id with symbol
id2symbol <- setNames(annot$Symbol, annot$GeneID)
gene_symbols <- id2symbol[rownames(raw_counts)]
gene_symbols_unique <- make.unique(gene_symbols)
rownames(raw_counts) <- gene_symbols_unique
head(rownames(raw_counts))

## Get additional data
library(GEOquery)

gse = getGEO("GSE58135",GSEMatrix = T)
pdata_1 = pData(gse[[1]])
dim(pdata_1)
View(pdata_1)
colnames(pdata_1)
#unique(pdata$source_name_ch1)
# label case control
pdata = pdata_1[pdata_1$source_name_ch1 %in% c("ER+ Breast Cancer Primary Tumor",
                                               "Uninvolved Breast Tissue Adjacent to ER+ Primary Tumor"),]


dim(pdata)
pdata$condition <- ifelse(grepl("Cancer", pdata$source_name_ch1),
                          "case", "control")
table(pdata$condition)

raw_counts   <- raw_counts[, rownames(pdata)]
dim(raw_counts)
############################################################
#library(DESeq2)
#library(edgeR)
#library(limma)

# 1(b)
# Filter low-count genes
keep_gene <- rowSums(cpm(raw_counts) >=1) >= 30
counts_filt <- raw_counts[keep_gene, ]
dim(counts_filt)

########################################################

# 1(c)
# Deseq2

pdata$condition <- factor(pdata$condition, levels = c("control", "case"))
dds <- DESeqDataSetFromMatrix(countData = counts_filt,
                              colData = pdata,
                              design = ~condition)
dds <- DESeq(dds)
result_deseq2 <- results(dds, contrast = c("condition", "case", "control"), alpha = 0.05)
sig_deseq2 <- subset(as.data.frame(result_deseq2), padj < 0.05 & abs(log2FoldChange) > 2)
nrow(sig_deseq2)
write.csv(sig_deseq2,"report_01_desecq2.csv")

#  edgeR
y <- DGEList(counts = counts_filt, group = pdata$condition)
y <- calcNormFactors(y, method = "TMM") # Normalization
design <- model.matrix(~condition, data = pdata)
y <- estimateDisp(y, design)
fit <- glmQLFit(y, design)
qlf <- glmQLFTest(fit, coef = 2)
res_edger <- topTags(qlf, n = Inf)$table
sig_edger <- subset(res_edger, FDR < 0.05 & abs(logFC) > 2)
nrow(sig_edger)
write.csv(sig_edger,"report_01_edger.csv")

# 4. limma-voom
v <- voom(y, design)
fit_lim <- lmFit(v, design)
fit_lim <- eBayes(fit_lim)
res_limma <- topTable(fit_lim, coef = 2, number = Inf)
sig_limma <- subset(res_limma, adj.P.Val < 0.05 & abs(logFC) > 2)
nrow(sig_limma)
write.csv(sig_limma,"report_01_limma.csv")
##### Summary
summarize_degs <- function(df, lfc_col = "logFC", label) {
  data.frame(Method = label,
             Up_Regulated = sum(df[[lfc_col]] > 0),
             Down_Regulated = sum(df[[lfc_col]] < 0),
             Total = length(df[[lfc_col]]))
}

deg_summary <- do.call(rbind, list(
  summarize_degs(sig_deseq2, "log2FoldChange", "DESeq2 ER+"),
  summarize_degs(sig_edger, label = "edgeR ER+"),
  summarize_degs(sig_limma, label = "Limma ER+")
))

deg_summary
write.csv(deg_summary,"report_01_deg_summary.csv")

######### Volcano plot
#library(EnhancedVolcano)

# DESeq2
png("report_01_deseq2.png",height = 8,width = 12,units = "in",res = 300)
EnhancedVolcano(result_deseq2,
                lab = rownames(result_deseq2),
                x = 'log2FoldChange', y = 'padj',
                pCutoff = 0.05, FCcutoff = 2,
                title = 'DESeq2: ER+ Breast Cancer vs Normal')
dev.off()
# edgeR
png("report_01_edger.png",height = 8,width = 12,units = "in",res = 300)
EnhancedVolcano(res_edger,
                lab = rownames(res_edger),
                x = 'logFC', y = 'FDR',
                pCutoff = 0.05, FCcutoff = 2,
                title = 'edgeR: ER+ Breast cancer vs Normal')
dev.off()
# Limma-voom
png("report_01_limma.png",height = 8,width = 12,units = "in",res = 300)
EnhancedVolcano(res_limma,
                lab = rownames(res_limma),
                x = 'logFC', y = 'adj.P.Val',
                pCutoff = 0.05, FCcutoff = 2,
                title = 'Limma-voom: ER+ Breast cancer Cancer vs Normal')
dev.off()


## Comparision among methods
all_degs <- unique(c(rownames(sig_deseq2), rownames(sig_edger), rownames(sig_limma)))
dim(all_degs)
# requires binary
venn_matrix <- data.frame(
  DESeq2 = all_degs %in% rownames(sig_deseq2),
  edgeR  = all_degs %in% rownames(sig_edger),
  Limma  = all_degs %in% rownames(sig_limma)
)

# calculate counts
v_counts <- vennCounts(venn_matrix)
png("report_01_Venn_Diagram_limma.png", width = 8, height = 6, units = "in", res = 300)

vennDiagram(v_counts, 
            include = "both",
            names = c("DESeq2", "edgeR", "Limma-voom"),
            circle.col = c("red", "blue", "green"),
            main = "Common DEGs Across Three Methods")

dev.off()

################################ 1 (d) get common deg  
# get common
common_DEG = intersect(rownames(sig_deseq2),rownames(sig_edger))
common_DEG = intersect(common_DEG,rownames(sig_limma))
length(common_DEG)
write.csv(common_DEG,"report_01_common_deg.csv")


##################################################################################################
##   I used two data sets under one GSE data set. Just of different types of Breast Cancer.
############# Question 02

rm(list = ls())
getwd()
setwd("C:\\Users\\LENOVO\\Desktop\\assignment 506")
############################### 1 b data preparation
raw_counts <- read.delim("GSE58135_raw_counts_GRCh38.p13_NCBI.tsv.gz",
                         header = TRUE, row.names = 1, check.names = FALSE)
dim(raw_counts)
colnames(raw_counts)[1:5]
rownames(raw_counts)[1:5]

annot <- read.delim("Human.GRCh38.p13.annot.tsv.gz", header = TRUE)
dim(annot)
View(annot)
colnames(annot)

####### Replace gene id with symbol
id2symbol <- setNames(annot$Symbol, annot$GeneID)
gene_symbols <- id2symbol[rownames(raw_counts)]
gene_symbols_unique <- make.unique(gene_symbols)
rownames(raw_counts) <- gene_symbols_unique
head(rownames(raw_counts))

## Get additional data

gse = getGEO("GSE58135",GSEMatrix = T)
pdata_1 = pData(gse[[1]])
dim(pdata_1)
#View(pdata_1)
colnames(pdata_1)
unique(pdata_1$source_name_ch1)
# label case control
pdata = pdata_1[pdata_1$source_name_ch1 %in% c("Triple Negative Breast Cancer Primary Tumor",
                                               "Uninvolved Breast Tissue Adjacent to TNBC Primary Tumor"),]


dim(pdata)
pdata$condition <- ifelse(grepl("Cancer", pdata$source_name_ch1),
                          "case", "control")
table(pdata$condition)

raw_counts   <- raw_counts[, rownames(pdata)]
dim(raw_counts)
############################################################
# 1(b)
# Filter low-count genes
keep_gene <- rowSums(cpm(raw_counts) >=1) >= 21
counts_filt <- raw_counts[keep_gene, ]
dim(counts_filt)

########################################################
# 1(c)
# Deseq2

pdata$condition <- factor(pdata$condition, levels = c("control", "case"))
dds <- DESeqDataSetFromMatrix(countData = counts_filt,
                              colData = pdata,
                              design = ~condition)
dds <- DESeq(dds)
result_deseq2 <- results(dds, contrast = c("condition", "case", "control"), alpha = 0.05)
sig_deseq2 <- subset(as.data.frame(result_deseq2), padj < 0.05 & abs(log2FoldChange) > 2)
nrow(sig_deseq2)
write.csv(sig_deseq2,"report_02_sig_deseq2.csv")

#  edgeR
y <- DGEList(counts = counts_filt, group = pdata$condition)
y <- calcNormFactors(y, method = "TMM") # Normalization
design <- model.matrix(~condition, data = pdata)
y <- estimateDisp(y, design)
fit <- glmQLFit(y, design)
qlf <- glmQLFTest(fit, coef = 2)
res_edger <- topTags(qlf, n = Inf)$table
sig_edger <- subset(res_edger, FDR < 0.05 & abs(logFC) > 2)
nrow(sig_edger)
write.csv(sig_edger,"report_02_edger.csv")

# 4. limma-voom
v <- voom(y, design)
fit_lim <- lmFit(v, design)
fit_lim <- eBayes(fit_lim)
res_limma <- topTable(fit_lim, coef = 2, number = Inf)
sig_limma <- subset(res_limma, adj.P.Val < 0.05 & abs(logFC) > 2)
nrow(sig_limma)
write.csv(sig_limma,"report_02_limma.csv")

summarize_degs <- function(df, lfc_col = "logFC", label) {
  data.frame(Method = label,
             Up_Regulated = sum(df[[lfc_col]] > 0),
             Down_Regulated = sum(df[[lfc_col]] < 0),
             Total = length(df[[lfc_col]]))
}

deg_summary <- do.call(rbind, list(
  summarize_degs(sig_deseq2, "log2FoldChange", "DESeq2 ER+"),
  summarize_degs(sig_edger, label = "edgeR ER+"),
  summarize_degs(sig_limma, label = "Limma ER+")
))

deg_summary
write.csv(deg_summary,"report_02_deg_summary.csv")


#library(EnhancedVolcano)

# DESeq2
png("Report 2 deseq2.png",height = 8,width = 12,units = "in",res = 300)
EnhancedVolcano(result_deseq2,
                lab = rownames(result_deseq2),
                x = 'log2FoldChange', y = 'padj',
                pCutoff = 0.05, FCcutoff = 2,
                title = 'DESeq2: TN Breast Cancer vs Normal')
dev.off()
# edgeR
png("Report 2 edger.png",height = 8,width = 12,units = "in",res = 300)
EnhancedVolcano(res_edger,
                lab = rownames(res_edger),
                x = 'logFC', y = 'FDR',
                pCutoff = 0.05, FCcutoff = 2,
                title = 'edgeR: TN Breast cancer vs Normal')
dev.off()
# Limma-voom
png("Report 2 limma.png",height = 8,width = 12,units = "in",res = 300)
EnhancedVolcano(res_limma,
                lab = rownames(res_limma),
                x = 'logFC', y = 'adj.P.Val',
                pCutoff = 0.05, FCcutoff = 2,
                title = 'Limma-voom: TN Breast cancer Cancer vs Normal')
dev.off()

## Comparision among methods
all_degs <- unique(c(rownames(sig_deseq2), rownames(sig_edger), rownames(sig_limma)))
length(all_degs)
# requires binary
venn_matrix <- data.frame(
  DESeq2 = all_degs %in% rownames(sig_deseq2),
  edgeR  = all_degs %in% rownames(sig_edger),
  Limma  = all_degs %in% rownames(sig_limma)
)

# calculate counts
v_counts <- vennCounts(venn_matrix)
png("Report 2 Venn_Diagram_limma.png", width = 8, height = 6, units = "in", res = 300)

vennDiagram(v_counts, 
            include = "both",
            names = c("DESeq2", "edgeR", "Limma-voom"),
            circle.col = c("red", "blue", "green"),
            main = "Common DEGs Across Three Methods")

dev.off()

################################ 2 (d) Select top 10  

# get common for q-2
common_DEG_2 = intersect(rownames(sig_deseq2),rownames(sig_edger))
common_DEG_2 = intersect(common_DEG_2,rownames(sig_limma))
length(common_DEG_2)
write.csv(common_DEG_2,"report_02_common_deg.csv")


############# Get overall common deg across two datasets

common_DEG_1 = read.csv("report_01_common_deg.csv", header = T,row.names = 1)
head(common_DEG_1)
overall_common = intersect(common_DEG_1$x,common_DEG_2)
length(overall_common)
write.csv(overall_common,"Overall common deg.csv")



########### To fetch data using code

library(GEOquery)
## Download all GSE62944 supplementary files
getGEOSuppFiles("GSE337023", makeDirectory = TRUE)
list.files("GSE337023")
list.files()
untar("GSE303509_RAW.tar")



