
setwd("C:\Users\LENOVO\Desktop\assignment 506\Report 3,4")
# CSV
data <- read.csv("metadata_PRJNA871997.csv")
# TSV 
write.table(data, file = "metadata_PRJNA871997.tsv",sep = "\t",row.names = FALSE,quote = FALSE)


setwd("D:/New folder (3)/New folder")
library("phyloseq")
library("qiime2R")
library("readr")

physeq = qza_to_phyloseq(features = "PRJANA871997-table.qza",taxonomy = "PRJANA871997-taxonomy.qza", metadata = "metadata_PRJNA871997.tsv")
physeq

#3(B)
prevalence <- apply(otu_table(physeq),1,function(x) sum(x > 0))
physeq_filt <- prune_taxa(prevalence >= 0.05 * nsamples(physeq),physeq)

## Normalize
rarefy = rarefy_even_depth(physeq_filt, sample.size= min(sample_sums(physeq_filt)),
                           rngseed=FALSE, replace=TRUE, verbose=T)
rarefy


#3(C)

######### Load liberies
library("ggplot2")
library("ggpubr")
#### Alpha diversity plot
## Chao1

tiff("Chao1_1_PRJANA871997.tiff", width = 3.5, height = 4, res = 300,units = "in", compression = c("lzw"))
plot_richness(rarefy, x = "Group", 
              measures = c("Chao1"), color = "Group")+
  geom_boxplot( notch = T, outlier.shape = NA,lwd=.6)+ 
  scale_colour_manual(values = c('red','blue')) + stat_boxplot(geom = "errorbar", width=0.1,lwd = .6)+
  theme_test()+guides(fill=F, color=FALSE)+xlab("")+ylab("") + 
  stat_compare_means(method = "wilcox.test")
dev.off()

###### Beta Diversity ###########
library("phyloseq")
library("ggplot2")
library("dplyr")
library("ggpubr")
library("Matrix")
library("reshape2")
library("vegan")

# First of all, we need to transform the data to relative abundance and create a new phyloseq object:
relab_genera <- transform_sample_counts(physeq_filt,function(x)x/sum(x)*100)

### Calculate Bray-Curtis distance among samples and convert the result to a matrix:
abrel_bray <- phyloseq::distance(relab_genera, method = "bray") 
abrel_bray <- as.matrix(abrel_bray)

########## PERMANOVA Test ################
library("vegan")
samples <- data.frame(sample_data(relab_genera))
adonis2(abrel_bray ~ Group, data = samples)  # bray curties distances

##To generate the PcoA plot, first get the ordination result with the "ordinate" command 
#from phyloseq. It requires a phyloseq object, and it accepts diverse methods and distances. 
#Next, generate the plot using the "plot_ordination()" function.

ord_bray = ordinate(relab_genera, method="PCoA", distance = "bray") #pcoa

######## PCoA plot with bray distance ########
tiff("PCoA_bray_PRJANA871997.tiff", height=4.7, width=6.2, units="in", res=300)
plot_ordination(relab_genera, ord_bray, color = "Group", shape="Group") + 
  geom_point(size=4) + scale_colour_manual(values = c('red','green')) + 
  stat_ellipse(aes(group=Group))+theme_test()+xlab("")+ylab("")+font("x.text", face=2,size=14)+
  font("y.text", face=2,size=14)+font("legend.text", size=15, face=2)+font("legend.title", size=16,face=2)+
  theme(legend.text = element_text(family = "Times New Roman"))
dev.off()




#3(D)


############  Relative Abundances #########

phyla = tax_glom(rarefy, taxrank = "Genus");phyla
ps_rel_abund = phyloseq::transform_sample_counts(phyla, function(x){x / sum(x)})
taxonomy <- as.data.frame(phyloseq::tax_table(ps_rel_abund))
write.csv(taxonomy, "taxa_Genus.csv", row.names = TRUE)

############ Taxa bar plot at Genus Level #########
Genus = tax_glom(rarefy, taxrank = "Genus");Genus
mergedPS = merge_samples(Genus, "Group");mergedPS # output is only two group
write.csv(otu_table(mergedPS), "otu_genus_1.csv")

df <- psmelt(Genus)

# Mean relative abundance of each genus
top15 <- df %>%group_by(Genus) %>% summarise(Perct = mean(Abundance), .groups = "drop") %>%arrange(desc(Perct)) %>% slice(1:15)

# Keep only Top 15 genera
df_top15 <- df %>% filter(Genus %in% top15$Genus)
plot_data <- df_top15 %>% group_by(Group, Genus) %>%summarise(Perct = mean(Abundance), .groups = "drop")

write.csv(plot_data, "genus_top15.csv", row.names = FALSE)

################### Taxa bar plot Manually ####################
######### Plot ########
#https://rpkgs.datanovia.com/ggpubr/reference/font.html
library(tidyverse)
theme_set(theme_bw(16))
library(extrafont)
library("ggplot2")
library(showtext)
### phylum label
library("extrafont")

######## Top 15 genus barplot #######
tiff("genus_top15_PRJANA871997.tiff",height = 4,width = 10, res = 300, units = "in", bg = "white")

ggplot(plot_data,
       aes(x = Group,
           y = Perct,
           fill = Genus)) +
  geom_bar(stat = "identity",
           position = "fill",
           width = 0.7) +
  scale_fill_manual(values = rainbow(15)) +
  coord_flip() +
  theme_classic() +
  labs(x = "",
       y = "Relative abundance") +
  theme(
    axis.text.y = element_text(size = 12, face = "bold"),   # Healthy/T2D দেখাবে
    axis.text.x = element_text(size = 12),
    legend.title = element_blank(),
    legend.position = "bottom",
    legend.text = element_text(size = 12),
    axis.line = element_line(linewidth = 0.3))

dev.off()


#3(e)

######### MetagenomeSeq ###############

library(metagenomeSeq)
library(phyloseq)
library(edgeR)

set.seed(100)

# Convert phyloseq object to metagenomeSeq object
metas <- phyloseq_to_metagenomeSeq(physeq_filt)

# CSS normalization
metas <- cumNorm(metas, p = 0.5)

# Sample metadata
pd <- pData(metas)

# Design matrix
mod <- model.matrix(~ Group, data = pd)

# Differential abundance analysis
res <- fitFeatureModel(metas, mod)

# Extract statistics
logFC <- res@fitZeroLogNormal$logFC
pvalue <- res@pvalues
adj.p <- p.adjust(pvalue, method = "BH")
Taxa <- res@taxa

# Create result table
result <- data.frame(Taxa = Taxa, LogFC = logFC, PValue = pvalue,Adj.PValue = adj.p, stringsAsFactors = FALSE)
# Extract taxonomy from phyloseq object
taxonomy <- as.data.frame(tax_table(physeq_filt))
taxonomy$Taxa <- rownames(taxonomy)

# Merge taxonomy with statistical results
result <- merge(result, taxonomy, by = "Taxa", all.x = TRUE)

# Reorder columns (optional)
result <- result[, c("Taxa","Kingdom","Phylum","Class", "Order", "Family","Genus","LogFC", "PValue", "Adj.PValue")]
# Significant taxa (FDR < 0.05)
sig_result <- subset(result, Adj.PValue < 0.05)
enriched <- subset(sig_result, LogFC > 0)
depleted <- subset(sig_result, LogFC < 0)
# Save results
write.csv(result, "MetaSeq_All_Results_With_TaxonomyPRJANA871997.csv", row.names = FALSE)
write.csv(sig_result,"MetaSeq_Significant_GenusPRJANA871997.csv",row.names = FALSE)

library(ggplot2)
tiff("MetaSeqPRJANA871997.tiff",height = 4,width = 6, res = 300, units = "in")
result$Status <- "Not Significant"

result$Status[result$Adj.PValue < 0.05 &
                result$LogFC > 0] <- "Enriched"

result$Status[result$Adj.PValue < 0.05 &
                result$LogFC < 0] <- "Depleted"
ggplot(result,
       aes(x = LogFC,
           y = -log10(Adj.PValue),
           color = Status)) +
  geom_point(size = 2) +
  scale_color_manual(values = c(
    "Enriched" = "red",
    "Depleted" = "blue",
    "Not Significant" = "grey"
  )) +
  geom_vline(xintercept = c(-1, 1),
             linetype = 2) +
  geom_hline(yintercept = -log10(0.05),
             linetype = 2) +
  theme_classic()


dev.off()

######### DESeq2 #########

library(DESeq2)
library(phyloseq)

# Convert phyloseq object to DESeq2 object
dds <- phyloseq_to_deseq2(physeq_filt, ~ Group)

dds <- estimateSizeFactors(dds, type = "poscounts")

dds <- DESeq(dds, sfType = "poscounts")
# Extract results
res <- results(dds, alpha = 0.05)

# Convert to data frame
res_df <- as.data.frame(res)
res_df$Taxa <- rownames(res_df)

# Extract taxonomy
taxonomy <- as.data.frame(tax_table(physeq_filt))
taxonomy$Taxa <- rownames(taxonomy)

# Merge taxonomy with DESeq2 results
result <- merge(res_df, taxonomy, by = "Taxa", all.x = TRUE)

# Reorder columns
result <- result[, c("Taxa","Kingdom","Phylum","Class","Order","Family","Genus","baseMean", "log2FoldChange","lfcSE","stat","pvalue", "padj")]
# Remove NA adjusted p-values
result <- result[!is.na(result$padj), ]

# Significant taxa (FDR < 0.05)
sig_result <- subset(result, padj < 0.05 )
enriched <- subset(sig_result, log2FoldChange > 0)
depleted <- subset(sig_result, log2FoldChange < 0)
# Save results
write.csv(result,"DESeq2_All_Results_With_TaxonomyPRJANA871997.csv",row.names = FALSE)
write.csv(sig_result, "DESeq2_Significant_GenusPRJANA871997.csv", row.names = FALSE)

library(ggplot2)
tiff("DESeq2_PRJANA871997.tiff",height = 4,width = 6, res = 300, units = "in")
result$Status <- "Not Significant"

result$Status[result$padj < 0.05 &
                result$log2FoldChange > 0] <- "Enriched"

result$Status[result$padj < 0.05 &
                result$log2FoldChange < 0] <- "Depleted"

ggplot(result,
       aes(x = log2FoldChange,
           y = -log10(padj),
           color = Status)) +
  geom_point(size = 2) +
  scale_color_manual(values = c(
    "Enriched" = "red",
    "Depleted" = "blue",
    "Not Significant" = "grey"
  )) +
  geom_vline(xintercept = c(-1, 1),
             linetype = 2) +
  geom_hline(yintercept = -log10(0.05),
             linetype = 2) 
dev.off()

######### edgeR #########

library(edgeR)
library(phyloseq)

# Extract count matrix
counts <- as.matrix(otu_table(physeq_filt))

# Sample metadata
meta <- data.frame(sample_data(physeq_filt))

# Create group factor
group <- factor(meta$Group)

# Create DGEList object
dge <- DGEList(counts = counts, group = group)

# Design matrix
design <- model.matrix(~ group)

# Estimate dispersion
dge <- estimateDisp(dge, design)

# Fit GLM
fit <- glmFit(dge, design)

# Likelihood ratio test
lrt <- glmLRT(fit)

# Extract all results
result <- topTags(lrt, n = Inf)$table
result$Taxa <- rownames(result)

# Extract taxonomy
taxonomy <- as.data.frame(tax_table(physeq_filt))
taxonomy$Taxa <- rownames(taxonomy)

# Merge taxonomy
result <- merge(result, taxonomy, by = "Taxa", all.x = TRUE)

# Reorder columns
result <- result[, c("Taxa", "Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "logFC", "logCPM", "LR","PValue", "FDR")]

# Significant taxa
sig_result <- subset(result, FDR < 0.05)
enriched <- subset(sig_result, logFC > 0)
depleted <- subset(sig_result, logFC < 0)
# Save results
write.csv(result,"edgeR_All_Results_With_TaxonomyPRJANA871997.csv", row.names = FALSE)
write.csv(sig_result,"edgeR_Significant_GenusPRJANA871997.csv",row.names = FALSE)

result$Status <- "Not Significant"

result$Status[result$FDR < 0.05 &
                result$logFC > 0] <- "Enriched"

result$Status[result$FDR < 0.05 &
                result$logFC < 0] <- "Depleted"
tiff("edgR_PRJANA871997.tiff",height = 4,width = 6, res = 300, units = "in")
ggplot(result,
       aes(x = logFC,
           y = -log10(FDR),
           color = Status)) +
  geom_point(size = 2) +
  scale_color_manual(values = c(
    "Enriched" = "red",
    "Depleted" = "blue",
    "Not Significant" = "grey"
  )) +
  geom_vline(xintercept = c(-1,1), linetype = 2) +
  geom_hline(yintercept = -log10(0.05), linetype = 2) +
  theme_classic()

dev.off()




#4
#(F)


edger1 <- read.csv("edgeR_All_Results_With_TaxonomyPRJANA871997.csv")
edger2 <- read.csv("edgeR_All_Results_With_Taxonomy_PRJNA820272.csv")

common1 <- intersect(edger1$Genus, edger2$Genus)
write.csv(common1,"common1.csv", row.names = FALSE)
length(common1)


meta1 <- read.csv("MetaSeq_All_Results_With_TaxonomyPRJANA871997.csv")
meta2 <- read.csv("MetaSeq_All_Results_With_Taxonomy_PRJNA820272.csv")

common2 <- intersect(meta1$Genus, meta2$Genus)
write.csv(common2,"common.csv", row.names = FALSE)
length(common2)


