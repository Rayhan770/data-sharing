# CSV
data <- read_tsv("metadata_PRJNA871997.tsv")
# TSV 
write.table(data, file = "metadata_PRJNA871997.tsv",sep = "\t",row.names = FALSE,quote = FALSE)

setwd("C:/Users/LENOVO/Desktop/exam 506/New folder (3)")
library("phyloseq")
library("qiime2R")
library("readr")

physeq = qza_to_phyloseq(features = "PRJANA871997-table.qza",taxonomy = "PRJANA871997-taxonomy.qza", metadata = "metadata_PRJNA871997.tsv")
physeq

#3(B) filtering
prevalence <- apply(otu_table(physeq),1,function(x) sum(x > 0))
physeq_filt <- prune_taxa(prevalence >= 0.05 * nsamples(physeq),physeq)

## Normalize
rarefy = rarefy_even_depth(physeq_filt, sample.size= min(sample_sums(physeq_filt)),
                           rngseed=FALSE, replace=TRUE, verbose=T)
rarefy


#3(C)

library("ggplot2")
library("ggpubr")
#### Alpha diversity plot
## Chao1
p.c1 = plot_richness(rarefy, x = "Group", measures = c("Chao1"), color = "Group") +
  geom_boxplot() +
  stat_compare_means(method = "wilcox.test")
p.c1  

###### Beta Diversity ###########
library("phyloseq")
library("ggplot2")
library("dplyr")
library("ggpubr")
library("Matrix")
library("reshape2")
library("vegan")

relab_genera <- transform_sample_counts(physeq_filt,function(x)x/sum(x)*100)

abrel_bray <- phyloseq::distance(relab_genera, method = "bray") 
abrel_bray <- as.matrix(abrel_bray)

########## PERMANOVA Test ################
library("vegan")
samples <- data.frame(sample_data(relab_genera))
adonis2(abrel_bray ~ Group, data = samples)  
ord_bray = ordinate(relab_genera, method="PCoA", distance = "bray") #pcoa

######## PCoA plot with bray distance 
p = plot_ordination(relab_genera, ord_bray, color = "Group") +
  geom_point(size = 3) +
  stat_ellipse() +
  theme_classic()
p

#3(D)
############  Relative Abundances

Genus = tax_glom(rarefy, taxrank = "Genus");Genus
ps_rel_abund = phyloseq::transform_sample_counts(Genus, function(x){x / sum(x)})
taxonomy <- as.data.frame(phyloseq::tax_table(ps_rel_abund))
write.csv(taxonomy, "taxa_Genus.csv", row.names = TRUE)

############ Taxa bar plot at Genus Level #########
mergedPS = merge_samples(Genus, "Group");mergedPS # output is only two group
write.csv(otu_table(mergedPS), "otu_genus_1.csv")

df <- psmelt(Genus)

# Mean relative abundance of each genus
m = aggregate(Abundance ~ Genus, data = df, FUN = mean)   # 1. mean of each genus
m = m[order(m$Abundance, decreasing = TRUE), ]            # 2. sort, biggest first
top = head(m, 15)                                       # 3. take the first 15
df_top = df[df$Genus %in% top$Genus, ]                                        # 4. keep only the top 15
plot_data = aggregate(Abundance ~ Group + Genus, data = df_top, FUN = mean)   # 5. mean per group                                                   # look at it
p = ggplot(plot_data, aes(x = Group, y = Abundance, fill = Genus)) +          # 6. plot
  geom_bar(stat = "identity", position = "fill", width = 0.7)
p
write.csv(plot_data, "genus_top15.csv", row.names = FALSE)

library(tidyverse)
library(extrafont)
library("ggplot2")
library(showtext)
### phylum label
library("extrafont")
#3(e)
######### MetagenomeSeq 

library(metagenomeSeq)
library(phyloseq)
library(edgeR)

set.seed(100)

metas <- phyloseq_to_metagenomeSeq(physeq_filt)
metas <- cumNorm(metas, p = 0.5)
pd <- pData(metas)
mod <- model.matrix(~ Group, data = pd)
res <- fitFeatureModel(metas, mod)

# Extract statistics
logFC <- res@fitZeroLogNormal$logFC
pvalue <- res@pvalues
adj.p <- p.adjust(pvalue, method = "BH")
Taxa <- res@taxa

# Create result table
result = data.frame(Taxa = Taxa, LogFC = logFC, PValue = pvalue,Adj.PValue = adj.p)
taxonomy = as.data.frame(tax_table(physeq_filt))
taxonomy$Taxa = rownames(taxonomy)
result = merge(result, taxonomy, by = "Taxa", all.x = TRUE)

# Significant taxa
sig_result = subset(result, Adj.PValue < 0.05)
sig_result$Status = ifelse(sig_result$LogFC > 0, "Enriched", "Depleted")

write.csv(sig_result,"MetaSeq_Significant_GenusPRJANA871997.csv",row.names = FALSE)

library(EnhancedVolcano)

p = EnhancedVolcano(result,
                    lab = result$Taxa,
                    x = "LogFC",
                    y = "Adj.PValue",
                    pCutoff = 0.05,
                    FCcutoff = 1,
                    title = "Case VS Control")
p
######### DESeq2 #########

library(DESeq2)
library(phyloseq)

dds <- phyloseq_to_deseq2(physeq_filt, ~ Group)
dds <- estimateSizeFactors(dds, type = "poscounts")
dds <- DESeq(dds, sfType = "poscounts")
res <- results(dds, alpha = 0.05)

res_df <- as.data.frame(res)
res_df$Taxa <- rownames(res_df)

taxonomy <- as.data.frame(tax_table(physeq_filt))
taxonomy$Taxa <- rownames(taxonomy)

result <- merge(res_df, taxonomy, by = "Taxa", all.x = TRUE)

result <- result[!is.na(result$padj), ]
sig_result <- subset(result, padj < 0.05 )

write.csv(sig_result, "DESeq2_Significant_GenusPRJANA871997.csv", row.names = FALSE)
sig_result = subset(result, padj < 0.05)
sig_result$Status = ifelse(sig_result$log2FoldChange > 0, "Enriched", "Depleted")

write.csv(sig_result,"MetaSeq_Significant_GenusPRJANA871997.csv",row.names = FALSE)

library(EnhancedVolcano)
p = EnhancedVolcano(result,
                    lab = result$Taxa,
                    x = "log2FoldChange",
                    y = "padj",
                    pCutoff = 0.05,
                    FCcutoff = 1,
                    title = "Case VS Control")
p
######### edgeR 

library(edgeR)
library(phyloseq)

counts <- as.matrix(otu_table(physeq_filt))
meta <- data.frame(sample_data(physeq_filt))
group <- factor(meta$Group)
dge <- DGEList(counts = counts, group = group)
design <- model.matrix(~ group)
dge <- estimateDisp(dge, design)

fit <- glmFit(dge, design)
lrt <- glmLRT(fit)
result <- topTags(lrt, n = Inf)$table
result$Taxa <- rownames(result)

# Extract taxonomy
taxonomy <- as.data.frame(tax_table(physeq_filt))
taxonomy$Taxa <- rownames(taxonomy)

# Merge taxonomy
result <- merge(result, taxonomy, by = "Taxa", all.x = TRUE)

# Significant taxa
sig_result <- subset(result, FDR < 0.05)

write.csv(sig_result,"edgeR_Significant_GenusPRJANA871997.csv",row.names = FALSE)
sig_result = subset(result, PValue < 0.05)
sig_result$Status = ifelse(sig_result$logFC > 0, "Enriched", "Depleted")

write.csv(sig_result,"MetaSeq_Significant_GenusPRJANA871997.csv",row.names = FALSE)

library(EnhancedVolcano)


p = EnhancedVolcano(result,
                    lab = result$Taxa,
                    x = "logFC",
                    y = "PValue",
                    pCutoff = 0.05,
                    FCcutoff = 1,
                    title = "Case VS Control")
p



