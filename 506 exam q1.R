rm( list = ls())
# load raw counts
data = read.delim("GSE50760_raw_counts_GRCh38.p13_NCBI.tsv.gz",header = T,row.names = 1,check.names = F)
dim(data)
head(colnames(data))
head(rownames(data))
# load annotation to get gene symbol
annot = read.delim("Human.GRCh38.p13.annot.tsv.gz",header = T)
colnames(annot)
# convert gene id to symbol
id2symbol = setNames(annot$Symbol,annot$GeneID)
gs = id2symbol[rownames(data)]
gsu = make.unique(gs)
rownames(data) = gsu
# download metadata 
library(GEOquery)
gsedata = getGEO("GSE50760") # internet required for this line
pdata1 = pData(gsedata[[1]])
dim(pdata1)
#View(pdata1)               
# source_name_ch1 (this varieble name can change datasetwise)
unique(pdata1$source_name_ch1)
#This line is not generally required pdata1 = pdata1[pdata1$source_name_ch1 %in% c("primary colorectal cancer","normal colon","metastasized cancer" ),]
pd = pdata1
pd$condition = ifelse(grepl("cancer",pd$source_name_ch1),"Case","Control") # keyword "cancer" can change dataset wise
table(pd$condition)

# arrange metadata and raw counts (fixes dimention miss match)
com.samp = intersect(colnames(data),rownames(pd))
length(com.samp)
pd = pd[com.samp,]
data = data[,com.samp]

#filter
library(edgeR) # edgeR package is always required
library(EnhancedVolcano)
kpg = rowSums(cpm(data)>=1)>=18 # change 18 to the lowest group size dataset wise
data.f = data[kpg,]
dim(data.f)

##edgeR (use which ever is asked)
y = DGEList(counts = data.f,group = pd$condition)
y = calcNormFactors(y,method = "TMM")
design = model.matrix(~condition,data = pd)
y = estimateDisp(y,design)
fit = glmQLFit(y,design)
qlf = glmQLFTest(fit,2)
res_edger = topTags(qlf,n = Inf)$table
sig_edg = subset(res_edger,FDR <.05 & abs(logFC)>2 ) ## Change logFC if needed
sig_edg$Regulation = ifelse(sig_edg$logFC > 0, "Up", "Down")
dim(sig_edg)
table(sig_edg$Regulation)
write.csv(sig_edg,"EdgeR result.csv")
EnhancedVolcano(res_edger,
                lab = rownames(res_edger),
                x = 'logFC', y = 'FDR',
                pCutoff = 0.05, FCcutoff = 2,
                title = 'Case VS Control')

##Deseq2
library(DESeq2)
pd$condition = factor(pd$condition, levels = c("Control", "Case"))
dds = DESeqDataSetFromMatrix(countData = data.f,
                              colData = pd,
                              design = ~condition)
dds = DESeq(dds)
result_deseq2 = results(dds, contrast = c("condition", "Case", "Control"), alpha = 0.05)
result_deseq2 = as.data.frame(result_deseq2)
sig_deseq2 <- subset(result_deseq2, padj < 0.05 & abs(log2FoldChange) > 2)
sig_deseq2$Regulation = ifelse(sig_deseq2$log2FoldChange > 0 ,"Up","Down")
nrow(sig_deseq2)
write.csv(sig_deseq2,"report_01_desecq2.csv")
EnhancedVolcano(result_deseq2,
                lab = rownames(result_deseq2),
                x = 'log2FoldChange', y = 'padj',
                pCutoff = 0.05, FCcutoff = 2,
                title = 'Case VS Control')


#limma-voom
library(limma)
pd$condition = factor(pd$condition, levels = c("Control", "Case"))
y = DGEList(counts = data.f, group = pd$condition)
y = calcNormFactors(y, method = "TMM")
design = model.matrix(~condition, data = pd)

v <- voom(y, design)
fit_lim <- lmFit(v, design)
fit_lim <- eBayes(fit_lim)
res_limma <- topTable(fit_lim, coef = 2, number = Inf)
sig_limma <- subset(res_limma, adj.P.Val < 0.05 & abs(logFC) > 2)
sig_limma$Regulation = ifelse(sig_limma$logFC > 0 ,"Up","Down")
nrow(sig_limma)
write.csv(sig_limma,"report_01_limma.csv")
EnhancedVolcano(res_limma,
                lab = rownames(res_limma),
                x = 'logFC', y = 'adj.P.Val',
                pCutoff = 0.05, FCcutoff = 2,
                title = 'Case VS Control')
