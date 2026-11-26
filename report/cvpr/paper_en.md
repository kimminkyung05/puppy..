TITLE: Estimating Mixed Visual Category Proportions from Breed-Labeled Images: A Case Study on Mixed-Breed Dogs
AUTHORS: Solmin Kim | Minkyung Kim | Chaerim Shin | Suyeon Cho
IDS: 2491045 | 2491046 | 2491051 | 2491053
AFFIL: Ewha Womans University

ABSTRACT
Estimating the composition proportions of categories mixed within one image is hard to evaluate because objective ground truth is rare. We study this problem on mixed-breed dogs, for which a DNA test reports the breed proportions of each individual. All methods are trained only on breed-labeled photos (Tsinghua Dogs, 127 breeds) and evaluated on photos and DNA results of 97 mixed-breed dogs. For RQ1, we compare three estimators on four frozen feature extractors (SIFT+BoVW, CLIP, DINOv2, DINOv3): a classifier whose breed probabilities are read as proportions, prototype unmixing that solves for a convex combination of breed-mean features, and mixture training on synthetic feature mixtures. For RQ2, we remove region, texture, or color cues and measure the excess drop beyond a matched control. The primary metric is the captured ancestry share, the true DNA proportion covered by the three top-ranked predicted breeds, reported with a prior baseline that ignores the image and an upper bound set by the training breeds. On the 77 test dogs, the nine deep-feature settings captured 0.052–0.083 of each dog's ancestry, against 0.058 for the prior baseline and 0.548 for the upper bound; none was distinguishable from the baseline, and differences between estimators were small and extractor-dependent. For RQ2, scores on the unperturbed test photos were already near the baseline, so cue reliance could be read only as tendencies: strong blur produced excess drops for the deep features, keeping only the head did not hurt more than keeping random pixels, and grayscale conversion had no detectable effect. Because 0.431 of the test dogs' ancestry is unreported by the DNA test or lies outside the training breeds, and the evaluation covers 97 dogs from one source, these results describe this setting rather than mixed-breed dogs in general.

# Introduction
When several categories are mixed in one image, objective ground truth for their composition proportions is rare. Mixed labels are usually assigned by human annotators, so they vary with the annotator. As a result, the question of how much mixture information image features contain is often studied without a reliable answer key.

Mixed-breed dogs offer a test case with objective ground truth. A DNA test reports the breed proportions of each dog, so proportions estimated from a photo can be scored against an answer that does not depend on human judgment. Several studies have reported that people struggle to identify the breeds of mixed-breed dogs from appearance [@Voith2009; @Olson2015; @Gunter2018]; when participants in the citizen-science survey MuttMix guessed the top three breeds of 30 mixed-breed dogs, they identified on average only 20.9% (SD 20.4%) of each dog's breed ancestry [@Morrill2022]. Visible traits such as coat color, coat length, ear shape, and body shape are nonetheless heritable and differ across breeds. Whether image features can recover at least part of the proportions is therefore an empirical question.

The task comes with a data constraint. Photos paired with DNA proportions are too scarce to train on, so we train only on breed-labeled photos and use mixed-breed photos only for evaluation. Each method must estimate composition proportions without ever seeing a mixed-breed dog. This setting isolates a single question: what transfers from single-breed supervision to mixtures.

We address two research questions. RQ1 asks how well image features recover the breed composition of mixed-breed dogs when trained only on breed-labeled photos, and how this depends on the estimator. RQ2 asks which visual cues (region, texture, or color) the estimates rely on. The three estimators encode different assumptions about how single-breed supervision extends to mixtures: reading classifier probabilities as proportions, solving for a convex combination of breed-mean features, and learning from synthetic feature mixtures. We repeat every experiment on four feature extractors with different training signals. We do not rank the extractors; they are measurement instruments, and we ask whether the direction of each conclusion holds across them.

Our contributions are threefold. First, we set up an evaluation protocol that scores proportion estimates for mixed visual categories against DNA, reporting the primary metric together with a prior baseline, an upper bound, and coverage. Second, we compare three estimators on shared frozen features and decompose where ancestry is lost along the pipeline. Third, we analyze cue reliance with perturbations and matched controls. On the test dogs, no estimator on frozen features recovered more of the breed composition than a baseline that ignores the image, whereas the estimators had looked useful on a small dev split. In the cue analysis, the deep features lost more under strong blur than under noise of equal PSNR and showed no reliance on color, although near-baseline scores limit what these drops can show. This work is not intended to replace DNA testing, and its scope is limited to dogs.

# Related Work
**Appearance and ancestry.** Breed labels assigned from appearance by adoption agencies, shelter staff, and other dog professionals often disagree with DNA results, and agreement between observers is low to moderate [@Voith2009; @Voith2013; @Olson2015]. Gunter et al. [@Gunter2018] surveyed the genetic breed heritage of shelter dogs and reported discrepancies between visual labels and DNA, and Morrill et al. [@Morrill2022] compared human breed guesses for mixed-breed dogs with DNA results in a citizen-science survey. Other work estimates genetic information directly from images. Ma et al. [@Ma2026] predicted genomic admixture from photos of 52 gayal paired with genomic data, and E and Wang [@E2026] used deep image embeddings of dogs as phenotypes in breed-level genome-wide association studies. In humans, genomic ancestry proportions have been predicted from 3D facial images as one component of a genome-based re-identification study [@Lippert2017]. Relatedly, Rando et al. [@Rando2024] tested whether consumer canine DNA tests are swayed by the photo submitted with the sample. These studies support the view that appearance carries some genetic information. To our knowledge, however, no prior study trains only on breed-labeled photos, without photo–DNA pairs, and scores the estimated breed composition of mixed-breed dogs against DNA.

**Estimating mixture proportions.** Each estimator relates to an existing line of work. Label distribution learning maps an input to a distribution over labels [@Geng2016; @Gao2017]. Mixup mixes two inputs and their labels with the same ratio [@Zhang2018], also in hidden representations [@Verma2019]; mixture training uses such mixing as the prediction target rather than as a regularizer. Prototype unmixing borrows fully constrained least squares (FCLS) from hyperspectral unmixing, which explains a pixel spectrum as a nonnegative, sum-to-one combination of pure-material spectra [@Heinz2001]; the finding that text embeddings of composite concepts in vision–language models are well approximated by linear combinations of a few vectors [@Trager2023] gives indirect support for applying it in feature space. Reading class probabilities as proportions also relates to calibration [@Guo2017].

**Visual cues and feature extractors.** Geirhos et al. [@Geirhos2019] used cue-conflict stimuli to show that ImageNet-trained CNNs rely more on texture than on shape, and frequency analyses have examined reliance on high-frequency components [@Yin2019; @Wang2020]. RQ2 applies this approach to proportion estimation and adds a control for each perturbation so that the drop caused by image degradation itself is subtracted. As feature extractors we use CLIP, which is trained with language supervision [@Radford2021]; DINOv2 and DINOv3, which are self-supervised [@Oquab2024; @Simeoni2025]; and SIFT [@Lowe2004] with a bag of visual words [@Csurka2004] as a classical baseline.

# Method
::: figure {#fig:pipe} src=figures/en/fig1_pipeline.png width=3.25
**Pipeline.** Breed-labeled photos (training) and mixed-breed photos (evaluation) pass once through the same frozen feature extractor. The three estimators (RQ1) turn the shared features into proportions over the Tsinghua classes; per-dog proportions are mapped to the DNA breed space with a breed map and scored. RQ2 perturbs the evaluation photos.
:::

[[fig:pipe]] summarizes the pipeline. Feature extraction and scoring are shared by all methods, so differences between methods come only from the estimator.

## Problem Setup and Data
**Task.** Let $\mathcal{B}$ be the set of training breeds with $|\mathcal{B}|=K$. The DNA result of dog $d$ is a vector of breed proportions $y_d$, which may include breeds outside $\mathcal{B}$. Each method estimates composition proportions $\hat{p}_d\in\Delta^{K-1}$ from the photos of the dog. Only breed-labeled photos are used for training; mixed-breed photos and DNA proportions are never used for training, and the dev dogs serve only to choose settings. The coverage $c_d=\sum_{b\in\mathcal{B}}y_{d,b}$ measures how much of a dog's ancestry the training breeds cover. When $c_d<1$, the remainder cannot be recovered by any method, so we use both the original proportions $y_d$ and the proportions renormalized within the training breeds, $\tilde{y}_{d,b}=y_{d,b}/c_d$. The former measures absolute performance including the unrecoverable share; the latter measures performance within the recoverable range.

**Training data.** We use the low-resolution release of Tsinghua Dogs [@Zou2020] (130 classes; 65,228 train and 5,200 val images). We remove chinese_rural_dog, which is not a single breed, and two wild canids (dhole, African_hunting_dog), leaving 127 classes (61,592 train and 5,080 val images). Inputs are crops of the body bounding boxes provided with the dataset. Because the labels were assigned from photos rather than pedigrees or DNA, we call these images breed-labeled photos.

**Mixed-breed evaluation set.** We use the public dog-breed identification photo chart of the University of Florida Shelter Medicine Program [@UF2026], which provides photos and DNA results for 100 dogs; 97 are mixed breed and 3 are single breed, which we exclude. Each dog has one image (about 448×195 pixels) with the face and full body side by side, which we split into a face crop and a body crop. The page does not name the DNA testing company or its breed panel. For each dog it also lists the breeds most often named by more than 5,000 dog experts in a survey, which we use only as a human reference row. We randomly assign 20 dogs to a dev split (seed 1) and the remaining 77 to a test split. In accordance with the terms of use, we report only aggregate results and do not release individual photos or per-dog DNA results; we have requested permission from the provider [TBD: response].

**Breed map.** We map the 138 breed names in the DNA table to Tsinghua classes as exact (89), similar (6), or none (43), and score only exact matches. When several classes correspond to one breed, their predicted proportions are summed; for example, the four poodle classes are merged into Poodle. Dingo stays in training but matches no DNA breed. The mean coverage of the 20 dev dogs is 0.567.

**Purebred auxiliary evaluation.** To check behavior on single-breed inputs, we split the 5,080 Tsinghua val images in half per class (seed 1) and measure Top-1 accuracy on the 2,540-image dev half. We report these results separately from the mixed-breed results.

## Feature Extractors
::: table {#tab:ext} cols=0.68,0.82,0.6,0.57,0.58 size=8
**Feature extractors.** All public weights are used frozen, without fine-tuning.
| Extractor | Training signal | Data | Backbone | Feature (dim.) |
|---|
| SIFT+BoVW | None (hand-designed) | — | — | Word<br>histogram (512) |
| CLIP | Image–text contrastive | WIT, 400M pairs | ViT-B/16 | Projected embedding (512) |
| DINOv2 | Self-distillation + masked patches | LVD-142M | ViT-B/14 | CLS token (768) |
| DINOv3 | DINOv2 objective + Gram anchoring | LVD-1689M | ViT-B/16 | CLS token (768) |
:::

The four extractors differ in their training signal ([[tab:ext]]). The deep models share a ViT-Base encoder [@Dosovitskiy2021] with 224×224 inputs. CLIP [@Radford2021] aligns images with captions through a contrastive loss on 400M image–text pairs, so its embedding carries concepts described in language. DINOv2 [@Oquab2024] is trained without labels by self-distillation at the image level and at masked patches on 142M curated images; our ViT-B/14 is distilled from ViT-g/14. DINOv3 [@Simeoni2025] scales this recipe to 1,689M images and a 7B teacher and adds Gram anchoring, which keeps patch features locally consistent; our ViT-B/16 is distilled from that teacher. SIFT+BoVW counts the assignments of 128-dimensional SIFT descriptors [@Lowe2004] to 512 visual words [@Csurka2004]; it uses no labels and is computed on grayscale images, so it carries local texture and edges but no color. Our main feature is the global feature: the projected image embedding for CLIP and the final-layer-normalized CLS token for DINOv2 and DINOv3. The patch-token mean (excluding the four register tokens of DINOv3) is used only in a diagnostic check.

## Proportion Estimators (RQ1)
All three estimators take a frozen feature $f\in\mathbb{R}^D$ and output proportions $\hat{p}\in\Delta^{K-1}$ over the Tsinghua classes.

**Classifier.** We train a single linear layer on the L2-normalized feature $\bar{f}=f/\|f\|_2$ and read its softmax probabilities directly as proportions:
$$ \hat{p}=\mathrm{softmax}(W\bar{f}+b),\quad W\in\mathbb{R}^{K\times D} $$ {#eq:clf}
Training minimizes a class-balanced cross-entropy with an L2 penalty,
$$ \mathcal{L}_{\mathrm{clf}}=-\frac{1}{N}\sum_{i=1}^{N}\omega_{y_i}\log\hat{p}_{i,y_i}+\alpha\|W\|_2^2,\quad \omega_c=\frac{N}{KN_c} $$ {#eq:clfloss}
where $N_c$ is the number of training images in class $c$ and $\alpha=10^{-4}$. We apply no temperature scaling. This estimator tests whether probabilities learned only for single-breed classification can also be read as mixing proportions.

**Prototype unmixing.** Without any training, we stack the mean training feature $\mu_k$ of each breed into $M\in\mathbb{R}^{K\times D}$ and find the convex weights that best approximate an evaluation feature:
$$ \hat{w}=\arg\min_{w\ge 0,\ \mathbf{1}^{\top}w=1}\|f-M^{\top}w\|_2^2 $$ {#eq:fcls}
This is the FCLS problem [@Heinz2001]. Following Heinz and Chang, we append the sum-to-one constraint as an extra row, here weighted by $\delta=10\max_{k,d}|M_{kd}|$, and solve the augmented system by nonnegative least squares; unlike the original method, we additionally divide the solution by its sum so that $\hat{p}$ sums exactly to one. Geometrically, $\hat{p}$ gives the barycentric coordinates of the point on the prototype simplex closest to $f$. We record the relative reconstruction error $\|f-M^\top\hat{w}\|_2/\|f\|_2$ only as a candidate confidence signal for the demo, not for scoring. Because the method has no training seed, we measure prototype variability by rebuilding the prototypes three times from random 80% subsets of each breed (seeds 1–3). We do not L2-normalize the features; doing so changed every dev metric by at most 0.003.

**Mixture training.** We synthesize mixtures in feature space and train the model to predict their proportions. For each mixture, we draw the number of breeds $k\in\{2,3,4\}$ uniformly, draw $k$ distinct breeds, draw one training feature $f_j$ per breed, and mix them with symmetric Dirichlet weights:
$$ \tilde{f}=\sum_{j=1}^{k}\lambda_j f_j,\quad t_c=\sum_{j=1}^{k}\lambda_j\mathbb{1}[c=c_j],\quad \lambda\sim\mathrm{Dir}(\mathbf{1}_k) $$ {#eq:mix}
The model is the same linear layer as the classifier and minimizes the soft cross-entropy $\mathcal{L}_{\mathrm{mix}}=-\sum_{c}t_c\log\hat{p}_c$ toward the proportions $t$. Features are standardized with training-set statistics before mixing, and the same statistics are applied to validation and evaluation photos. Each epoch draws 20,000 new mixtures and trains on them together with all single-breed photos with one-hot targets, so that the model keeps separating single breeds while learning to decompose mixtures. To isolate the effect of the training target, we also train a single-label control with the same settings but without synthetic mixtures.

**Per-dog prediction and repeats.** For all estimators, we average the predictions of a dog's face and body crops and renormalize them to obtain $\hat{p}_d$. The classifier and mixture training are repeated with seeds 1–3 and prototype unmixing with three subsamples; the representative score is computed from the per-dog proportions averaged over the three repeats.

## Cue Perturbation (RQ2)
We remove one visual cue from an evaluation photo, rerun the same estimator, and measure the score drop. Unlike RQ1, which averages the face and body crops, RQ2 perturbs each dog's whole UF image (face and body side by side), so that a region can be removed in context; after perturbation, the image is cropped to the bounding box of the dog mask with a 10% margin. Masks come from SAM 3 [@Carion2025] prompted with "dog" for the body and "dog head" for the head; for one test dog the head box was drawn by hand on the face panel.

**Perturbations.** The **region** perturbations keep only the head, only the body without the head, or only the background (outside the dog mask) and fill the rest with the median image color. The **texture** perturbation is a Gaussian blur with $\sigma=\max(0.5,\ r\cdot s)$ and kernel size $2\lceil 3\sigma\rceil+1$ for $r\in\{0.01,0.03,0.06,0.12\}$, where $s$ is the short side of the whole image; a second set uses the short side of the dog's bounding box, so that the blur scales with the dog rather than the photo. The **color** perturbation converts the image to grayscale. Each dog thus has 12 perturbed images.

**Matched controls.** A perturbation removes a cue but also degrades the image, so each perturbation has a control that degrades the image equally while keeping the cue. For blur and grayscale, the control adds Gaussian noise scaled to the same mean squared error as the perturbation, so both have equal PSNR. For a region perturbation, the control keeps a random set of pixels of the same count and fills the rest in the same way. Random draws are seeded per image and condition. With the original, each dog has 25 images.

**Estimators and scoring.** The deep extractors use the classifier of RQ1 with its three seeds. SIFT+BoVW uses mixture training on features re-extracted with the stored codebook, because perturbed images must be encoded with the same visual words as the training features; this differs from the official SIFT+BoVW features of RQ1 (Sec. 4.1). Scores are captured ancestry shares on the original proportions, computed after averaging the three seeds. We measure cue reliance by the excess drop,
$$ \Delta_{\mathrm{ex}}=(S_{\mathrm{o}}-S_{\mathrm{p}})-(S_{\mathrm{o}}-S_{\mathrm{c}})=S_{\mathrm{c}}-S_{\mathrm{p}} $$ {#eq:excess}
where $S_{\mathrm{o}}$, $S_{\mathrm{p}}$, and $S_{\mathrm{c}}$ are the scores on the original, perturbed, and control photos. We compute $\Delta_{\mathrm{ex}}$ per dog and report its mean ± 2SE over dogs; a positive value indicates a drop that image degradation alone does not explain, and an interval that includes zero is reported as inconclusive. With 12 perturbations on four extractors we make 48 comparisons without correction, so about two intervals would exclude zero by chance; we read only patterns that recur across extractors or blur levels. Two checks guard the pipeline. On dev, our implementation reproduces the perturbed images of the team's first RQ2 pipeline (equal PSNR for all 280 images). SIFT+BoVW is computed on grayscale images, so its score under the grayscale perturbation must equal that of the original; it did for every dev and test dog.

## Evaluation
**Primary metric.** Let $T_d$ be the three DNA breeds with the largest predicted proportions for dog $d$. The captured ancestry share is
$$ s_d=\sum_{b\in T_d}y_{d,b},\quad S=\frac{1}{|\mathcal{D}|}\sum_{d\in\mathcal{D}}s_d $$ {#eq:cap}
We select the top three after summing classes that map to the same DNA breed. The primary metric uses the original proportions, so ancestry outside the training breeds can never be captured; we therefore always report the mean coverage next to it. The metric first asks which breeds are present rather than their exact proportions, so methods whose predicted distributions differ in sharpness are compared on the same scale.

**Auxiliary metrics and baselines.** The auxiliary metrics are the captured ancestry share on the renormalized proportions $\tilde{y}_d$; the distribution overlap $O_d=\sum_b\min(\hat{p}_{d,b},\tilde{y}_{d,b})$, which also reflects proportion magnitudes; and Top-1 agreement, which checks whether the top predicted class belongs to the largest DNA breed. Dogs whose largest breed lies outside the training breeds are excluded from the Top-1 denominator, and we report the number of scorable dogs. Every results table includes a prior baseline, which predicts the training-set breed frequencies for every dog without looking at the image, and an upper bound, the score obtained by choosing the three training breeds with the largest true proportions. As a human reference, we score the three breeds most often named for each dog in the UF expert survey with the same metric; the survey gives no proportions, so overlap is not computed for it.

**Protocol.** We fix all settings on dev only and score the test split once with frozen settings; we do not change settings after seeing test scores, and we report all twelve extractor–estimator combinations on test. The representative setting used for the deliverable is chosen by the team after test scoring, so it is a technical choice rather than a confirmatory result. Before this official scoring, two team members had scored the test split once under an earlier breed map and pipeline; those runs did not change any setting reported here. Comparisons with the baseline or between methods use the mean of per-dog paired differences ±2 standard errors, and an interval that includes zero is reported as inconclusive. Top-1 agreement carries a 95% Wilson interval [@Wilson1927]. We apply no multiple-comparison correction, so we do not claim a ranking from a single comparison whose interval barely excludes zero. We repeat each comparison on every extractor and call a result consistent when the signs agree and extractor-dependent when they do not.

# Experiments
**Implementation.** Each deep model uses its default Hugging Face processor and normalization: CLIP and DINOv2 inputs are resized to a short side of 224 and 256 pixels, respectively, and center-cropped to 224×224, whereas DINOv3 inputs are resized directly to 224×224 without preserving the aspect ratio. Inference runs in fp16. For SIFT+BoVW, we extract OpenCV SIFT descriptors from 336×336 grayscale crops, cluster the descriptors of 20,000 training images into 512 visual words with mini-batch k-means, and apply a Hellinger transform to the histograms. The classifier is trained with Adam (learning rate $10^{-3}$, batch 1,024, at most 30 epochs, early stopping on a 10% held-out training split). Mixture training uses AdamW (learning rate $3\times10^{-3}$, weight decay $10^{-4}$, batch 512, 20 epochs, 3,200 steps). We selected its learning rate from $\{10^{-3}, 3\times10^{-3}, 10^{-2}\}$ by purebred val Top-1 on SIFT+BoVW features and reused it for the deep features; no mixed-breed photo was used for this choice. The range of breed counts, the Dirichlet parameter, and the number of mixtures per epoch were fixed without search. All experiments ran on a single Colab T4 GPU; extracting DINOv3 features for the whole training set took about 9 minutes.

**Reproducibility.** Experiment logic lives in Python scripts, and notebooks only call them. We use public checkpoints (openai/clip-vit-base-patch16, facebook/dinov2-base, facebook/dinov3-vitb16-pretrain-lvd1689m). Features are extracted once into atomically written shards of 5,000 images shared by all estimators; every run, including failed ones, is logged with its settings, seed, data version, split, and metrics; and one script scores the per-dog predictions of all estimators. The code is available at [TBD: public repository link]. We do not redistribute the UF photos, DNA results, or features derived from them.

We first report the single scoring of the 77 test dogs, which carries our conclusions, and then the dev analyses (20 dogs) that fixed the settings; dev numbers do not support conclusions.

## Proportion Recovery (RQ1)
::: table {#tab:test} cols=0.6,0.55,0.4,0.55,0.45,0.35,0.35 size=8
**Mixed-breed proportion recovery** on test (77 dogs, mean coverage 0.569; scored once with frozen settings) and, in the last column, on dev (20 dogs, coverage 0.567; used only for setting selection). Capt.: captured ancestry share on original (orig.) or renormalized (renorm.) proportions; overlap: distribution overlap on renormalized proportions. Top-1 is over the 66 scorable test dogs (UF survey: 69); Wilson 95% intervals are [0.024, 0.146] for 4/66, [0.063, 0.221] for 8/66, and [0.208, 0.421] for 21/69. †SIFT+BoVW; not interpreted because of a codebook mismatch in the evaluation features (Sec. 4.1).
| Method | Extractor | Capt.<br>(orig.) | Capt.<br>(renorm.) | Overlap | Top-1 | Dev<br>capt. |
|---|
| Prior baseline | — | 0.058 | 0.097 | 0.034 | — | 0.088 |
|---|
| Classifier | SIFT† | 0.015 | 0.022 | 0.025 | 1/66 | 0.017 |
|  | CLIP | 0.078 | 0.144 | 0.047 | 8/66 | 0.205 |
|  | DINOv2 | 0.078 | 0.139 | 0.068 | 4/66 | 0.255 |
|  | DINOv3 | 0.072 | 0.128 | 0.071 | 5/66 | 0.155 |
|---|
| Prototype unmixing | SIFT† | 0.013 | 0.018 | 0.005 | 0/66 | 0.000 |
|  | CLIP | 0.057 | 0.100 | 0.051 | 4/66 | 0.217 |
|  | DINOv2 | 0.083 | 0.144 | 0.076 | 4/66 | 0.205 |
|  | DINOv3 | 0.078 | 0.130 | 0.073 | 4/66 | 0.130 |
|---|
| Mixture training | SIFT† | 0.023 | 0.037 | 0.033 | 1/66 | 0.038 |
|  | CLIP | 0.052 | 0.092 | 0.064 | 7/66 | 0.167 |
|  | DINOv2 | 0.076 | 0.139 | 0.068 | 5/66 | 0.205 |
|  | DINOv3 | 0.080 | 0.133 | 0.088 | 7/66 | 0.155 |
|---|
| UF survey responses | — | 0.144 | 0.240 | — | 21/69 | 0.180 |
| Upper bound | — | 0.548 | 0.974 | 1.000 | — | 0.541 |
:::

**Test results.** On the 77 test dogs ([[tab:test]]), the prior baseline scored 0.058 and the upper bound 0.548. The nine deep settings scored 0.052–0.083, and none differed from the prior baseline per dog; the largest differences were +0.024 ± 0.046 (prototype unmixing, DINOv2) and +0.022 ± 0.040 (mixture training, DINOv3). Top-1 agreement was 4–8 of the 66 scorable dogs, and 15 dogs received a correct Top-1 from at least one deep setting. The three breeds most often named by the UF experts scored 0.144, higher than the prior baseline (+0.086 ± 0.052). Across repeats, the deep settings varied by at most 0.030 (mixture training, DINOv2: 0.071–0.101).

::: table {#tab:pairs} cols=0.83,0.81,0.81,0.8 size=8
**Paired differences between estimators on test** in captured ancestry share (77 dogs, original proportions): mean per-dog difference (first − second) ± 2SE. *Interval excludes zero (before multiple-comparison correction).
| Pair | CLIP | DINOv2 | DINOv3 |
|---|
| Classifier −<br>prototype | +0.021 ± 0.016* | −0.005 ± 0.022 | −0.005 ± 0.021 |
| Classifier −<br>mixture | +0.026 ± 0.022* | +0.002 ± 0.029 | −0.008 ± 0.031 |
| Prototype −<br>mixture | +0.005 ± 0.019 | +0.006 ± 0.029 | −0.003 ± 0.030 |
:::

**Differences between estimators on test.** Seven of the nine comparisons were inconclusive ([[tab:pairs]]). The two exceptions were both on CLIP, where the classifier scored higher than prototype unmixing (+0.021 ± 0.016) and mixture training (+0.026 ± 0.022). For all three pairs the sign of the difference varied across extractors, so every comparison is extractor-dependent. Per-dog scores were again correlated across estimators (0.62–0.89). We therefore find no evidence that prototype unmixing or mixture training, which build in an explicit mixture model, recover composition better than reading classifier probabilities.

**Why dev did not carry over.** The dev score of each deep setting (0.130–0.255) was 1.7–3.8 times its test score ([[tab:test]], [[fig:pure]]). On dev, much of the score came from dogs with bully-breed ancestry (see the per-dog analysis below), and such dogs made up 9 of the 20 dev dogs but only 15 of the 77 test dogs. Coverage was similar (0.567 vs. 0.569), so we read the gap as reflecting which breeds the dogs carry rather than how much of their ancestry the training breeds cover (not proven). The dev analyses below explain how the settings were fixed; we read their numbers only in that role.

::: figure {#fig:pure} src=figures/en/fig2_purebred_vs_mixed.png width=3.25
**Purebred Top-1 accuracy vs. mixed-breed captured ancestry share** for the nine deep settings, on dev (open, 20 dogs) and test (filled, 77 dogs). Color: estimator; marker: extractor; horizontal lines: prior baselines.
:::

**Dev results (setting selection).** On dev (last column of [[tab:test]]), the three estimators on deep features scored 0.130–0.255 against a prior baseline of 0.088, and 3 of 12 settings scored higher than the baseline per dog: the classifier on DINOv2 (+0.167 ± 0.122) and prototype unmixing on CLIP (+0.129 ± 0.125) and DINOv2 (+0.117 ± 0.112). Mixture training and the UF survey responses (0.180) were inconclusive. Eight of the nine pairwise comparisons were inconclusive; the ninth, classifier minus prototype unmixing on DINOv2 (+0.050 ± 0.046), did not hold on test (−0.005 ± 0.022). Per-dog scores were highly correlated across estimators (0.58–0.97), and 6 of the 20 dogs scored zero under all nine deep settings, so the estimators succeeded and failed on the same dogs.

**Ensembles.** Of three equal-weight combinations of the nine deep settings, only the arithmetic mean exceeded the best single setting (0.267 vs. 0.255, +0.013 ± 0.025, from a single dog), and a per-dog oracle also reached only 0.255, so we do not use ensembles.

**Extractors and feature types.** On dev, the signs of classifier − prototype unmixing and prototype unmixing − mixture training varied with the extractor, and classifier − mixture training was positive on CLIP and DINOv2 and 0.000 ± 0.073 on DINOv3; all of these differences were inconclusive. Replacing the CLS token with the patch-token mean or a concatenation of both never scored higher in 18 combinations, and the patch mean scored lower for the classifier (−0.105 ± 0.066) and prototype unmixing (−0.142 ± 0.084) on DINOv2. On synthetic mixtures of validation features, prototype unmixing and mixture training matched or exceeded the classifier (0.792 and 0.777 vs. 0.753 on DINOv2), but not on real dogs, which suggests (not proves) that real mixed-breed features are poorly described as linear combinations of breed features. Dev scores also moved with conditions outside the settings (prototype unmixing on DINOv2: 0.255 with 30 images per breed, 0.205 with all), and on 20 dogs a difference of 0.05 equals a score of 1.0 on a single dog.

::: table {#tab:pure} cols=0.97,0.7,0.5,0.54,0.54 size=8
**Purebred auxiliary evaluation:** Top-1 accuracy on the val dev half (2,540 images), mean of three repeats.
| Method | SIFT+BoVW | CLIP | DINOv2 | DINOv3 |
|---|
| Classifier | 0.091 | 0.723 | 0.860 | 0.879 |
| Prototype unmixing | 0.071 | 0.619 | 0.808 | 0.822 |
| Mixture training | 0.077 | 0.753 | 0.839 | 0.856 |
| Single-label control | 0.074 | 0.771 | 0.842 | 0.870 |
:::

**Purebred auxiliary evaluation.** On deep features, all three estimators exceeded 0.6 Top-1, while SIFT+BoVW stayed below 0.1 ([[tab:pure]]). Prototype unmixing was 0.052–0.104 below the classifier, and mixture training only 0.003–0.018 below the single-label control despite more steps (3,200 vs. 2,420), so its purebred cost is small; both rows may be slightly optimistic because their learning rate was chosen on the full purebred val set. The purebred ordering did not carry over to mixed-breed dogs ([[fig:pure]]): within each estimator, DINOv3 had the highest purebred Top-1 but the lowest dev score, whereas prototype unmixing on CLIP, with the lowest purebred Top-1 (0.619), reached 0.217 on dev; on test, every deep setting stayed at the baseline despite purebred Top-1 of 0.62–0.88. Purebred accuracy thus does not guarantee recovery of mixed-breed proportions.

**Where ancestry is lost.** An ensemble helps only if settings fail on different parts of the ancestry, so we decomposed where each dev dog's ancestry is lost ([[fig:fail]]). Before any photo is seen, 0.433 is lost: 0.182 is not reported by the DNA test and 0.251 lies outside the training breeds. Of the remaining 0.567, the largest share in every setting was pushed to rank 31 or lower (0.220–0.283), whereas near misses at ranks 4–10 were small (0.025–0.113), so most failures lie far from the top-three boundary. The union captured by any of the nine settings is at most 0.267, only 0.013 above the best single setting, and of the 0.299 missed by all nine, 0.225 was ranked 11th or lower by every setting. Ranking breeds by cosine similarity to DINOv2 breed prototypes, without any estimator, already gives 0.242 (face crop alone 0.242, body crop alone 0.175). On dev, ancestry is thus lost in the order mapping, features, and estimator, and settings differ little at the estimator stage.

::: figure {#fig:fail} src=figures/en/fig3_failure_stages.png width=3.25
**Where ancestry is lost** (dev, 20 dogs, original proportions, aggregates only). (a) For each deep setting, each dog's ancestry is split into not reported, outside training breeds, and exact-mapped breeds by predicted rank; the share at ranks 1–3 is the primary metric. (b) Exact-mapped ancestry (mean coverage 0.567) split by how many of the nine settings captured it.
:::

**Per-dog analysis.** On dev, the nine deep settings captured on average 0.35 of the upper bound, and the 8 dogs with coverage below 0.5 had an upper bound of only 0.340. The 9 dogs with bully-breed ancestry (American Staffordshire terrier, Staffordshire bull terrier) averaged 0.312, or 0.56 of their upper bound, whereas the other 11 averaged 0.087, or 0.17, and 48 of the 49 bully-breed Top-1 choices were for dogs carrying that ancestry. Much of the dev score thus came from one visually distinctive breed group.

**SIFT+BoVW.** SIFT+BoVW carried weak breed signal even on purebred photos: on 800 training images (8 breeds × 100), breed explained a fraction $\eta^2$ of the feature variance of 0.027 for SIFT+BoVW, 0.245 for CLIP, and 0.408 for DINOv2. A grayscale word histogram discards spatial layout and coat color, and even a dense color opponent-SIFT bag of words with a spatial pyramid reached only 0.216 on the 120-breed ILSVRC 2012 dog challenge [@Goring2013]. Because the deep features' purebred Top-1 (0.723–0.879) falls near the Tsinghua Dogs benchmark range (0.777–0.864, from fine-tuned networks on larger inputs [@Zou2020]), we attribute the low SIFT+BoVW score to the representation rather than to the shared pipeline. Separately, the mixed-breed evaluation features contained an implementation error: the stored codebook was re-clustered without a fixed seed when the training features were built, so word indices disagreed between training and evaluation features (rank correlation of word usage: 0.904 between training and validation photos, 0.234 between training and mixed-breed photos). With a consistent codebook, mixture training raised its dev overlap from 0.017 to 0.071 but its captured share (0.033) stayed below the baseline (0.088). The official results keep the original features so that the three estimators share identical inputs, and we do not interpret SIFT+BoVW mixed-breed results.

## Visual Cue Dependence (RQ2)
::: table {#tab:cue} cols=0.5,0.83,0.48,0.48,0.48,0.48 size=8
**Excess drop on test** (77 dogs; captured ancestry share, original proportions). Positive: a larger drop than the matched control. Blur is relative to the short side of the image or the dog box. *Interval excludes zero (2SE 0.012–0.045, uncorrected). SIFT: mixture training; others: classifier.
| Cue | Perturbation | CLIP | DINOv2 | DINOv3 | SIFT |
|---|
| — | Original score | 0.073 | 0.063 | 0.066 | 0.031 |
|---|
| Region | Head only | −0.020 | −0.042* | −0.060* | −0.003 |
|  | Body only | −0.003 | +0.010 | −0.007 | −0.006 |
|  | Background only | −0.007 | +0.024 | +0.022 | −0.013 |
|---|
| Texture<br>(image) | Blur 1% | −0.021 | −0.007 | −0.025 | −0.012 |
|  | Blur 3% | +0.019 | +0.007 | −0.007 | −0.013 |
|  | Blur 6% | +0.035 | +0.018 | +0.049* | +0.001 |
|  | Blur 12% | +0.036* | +0.039* | +0.030 | −0.022 |
|---|
| Texture<br>(dog) | Blur 1% | −0.018 | −0.014 | −0.019 | −0.003 |
|  | Blur 3% | −0.003 | −0.016 | −0.021 | −0.017 |
|  | Blur 6% | +0.037* | −0.003 | +0.038* | 0.000 |
|  | Blur 12% | +0.023 | +0.043* | +0.027 | −0.016 |
|---|
| Color | Grayscale | −0.013 | −0.003 | −0.009 | +0.003 |
:::

**Cue reliance on test.** On the unperturbed whole photos, the four extractors scored 0.031–0.073 against the prior baseline of 0.058 ([[tab:cue]]; the deep scores differ slightly from [[tab:test]] because RQ2 uses the whole photo instead of the two crops). Every RQ2 drop is therefore measured from near the baseline, and we read the results as tendencies. Eight of the 48 intervals excluded zero, against about two expected by chance, and they formed two patterns. **Texture:** strong blur (6% and 12%) gave a positive excess drop in 11 of the 12 deep-feature cells, 6 of them beyond 2SE; these photos scored 0.014–0.055, below the baseline, whereas their noise controls scored 0.047–0.070. **Region:** keeping only the head lowered the score less than keeping an equal number of random pixels on DINOv2 (−0.042 ± 0.042) and DINOv3 (−0.060 ± 0.040), and the head-only photo scored at least as high as the original (DINOv2 0.073, DINOv3 0.096); body-only and background-only photos were inconclusive. **Color:** grayscale conversion was inconclusive on all extractors, with excess drops of −0.013 to +0.003. SIFT+BoVW was inconclusive throughout. Taken together, the deep features appear to rely on fine texture and on the head region rather than on color, but because the original estimates carry little composition signal, this shows what the features respond to rather than which cues support a correct composition.

# Discussion
**Deliverable: family-tree-style view.** Our deliverable is a Gradio page that shows the three estimators' breed proportions for one photo side by side as family trees, on DINOv2 global features chosen on dev, rounded to 12.5% steps in eight great-grandparent slots. When the reconstruction error of prototype unmixing exceeds the 95th percentile of the dev crops, the page abstains and shows only a table; we did not test whether this heuristic tracks accuracy. The page states that the tree depicts proportions rather than a pedigree (50% may mean one parent or two grandparents) and is a research estimate that does not replace DNA testing. The demo does not use UF photos.

**Limitations.** The evaluation set has 97 dogs and the dev split 20, so small differences remain inconclusive, and the RQ2 drops were measured from near-baseline scores. The photos are low resolution, show one scene per dog, and come from a single source that does not describe how the dogs were recruited, so we cannot separate task difficulty from imaging conditions or know whether the results hold for other populations. Ancestry outside the training breeds or not visible in appearance cannot be recovered; the training labels were assigned from appearance, DNA results depend on the testing company's reference panel, and the breed map involves judgment. Mixture training was tuned only for its learning rate, and SIFT+BoVW results describe only our simple configuration (grayscale, 512 words).

# Conclusion
We scored photo-based breed compositions of mixed-breed dogs against DNA while training only on breed-labeled photos, reporting the captured ancestry share with a prior baseline, an upper bound, and coverage. On the 77 test dogs, none of the nine deep-feature settings beat a baseline that ignores the image, despite purebred Top-1 of 0.62–0.88, whereas the UF experts' guesses did; the estimators differed little, and only on CLIP did the classifier score higher. Dev scores of 0.13–0.26 fell to 0.05–0.08 on test, showing how far a small dev split dominated by a few distinctive breeds can mislead. The deep features responded to strong blur and to the head region but not to color, as tendencies measured from near-baseline scores. Because our dev decomposition located most of the loss in the breed map and the features, wider breed coverage and features that separate breeds within mixtures appear to matter more than the choice of estimator.
