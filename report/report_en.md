# Estimating Composition Proportions of Mixed Visual Categories: A Case Study of Mixed-Breed Dog Ancestry

Draft report v0 (2026-11-22). Writing rules follow `WRITING.md`.

**Status of this draft**

- Sections 1–6, the references, and the appendices are drafted. Section 7 contains only table templates, and the cells are left empty. Sections 8 and 9 are partly written. The abstract and Section 10 will be written after the results are available.
- `[TBD: …]` marks a value that does not exist yet, `[VERIFY: …]` marks content that we have not verified, and ※ marks a reference whose bibliographic details must be rechecked.
- The structure with 2 research questions, the purebred auxiliary evaluation, and the evaluation set built mainly on the UF material are proposals that the team has not yet confirmed `[TBD: team decision]`.

---

## Abstract

`[TBD: to be written after the results are available. Problem, method, key result, and limitations in one paragraph]`

---

## 1. Introduction

When several categories are mixed in one image, a ground truth that can score their composition proportions objectively is rare. Mixed labels are usually assigned subjectively by people. As a result, the question of how much information about mixture proportions image features carry is often addressed with an uncertain ground truth.

Mixed-breed dogs offer a case in which this question can be tested. A DNA test gives the breed composition proportions of each dog as numbers, so proportions estimated from photos can be compared with a ground truth that does not depend on human judgment. Several studies have reported that people have difficulty identifying the breeds of mixed-breed dogs from appearance alone (Voith et al., 2009; Olson et al., 2015; Gunter et al., 2018). In MuttMix, a citizen-science survey, the breeds that participants chose were reported to capture 20.9% of the dogs' actual ancestry on average (Lord et al., 2025 ※). However, appearance traits such as coat color, coat length, ear shape, and body shape are heritable and differ across breeds. Whether image features can recover at least part of the proportions is therefore a question worth testing experimentally.

The task has one constraint. Data that pair photos with DNA proportions are scarce and cannot be used for training. We train only on breed-labeled photos and use photos of mixed-breed dogs only for evaluation. The model must therefore estimate mixture proportions without ever having seen a mixed-breed dog.

This report asks two questions.

- **RQ1.** When training uses only breed-labeled photos, how well can image features recover the breed composition proportions of mixed-breed dogs? How does the answer differ across methods (classifier, prototype unmixing, mixture training)?
- **RQ2.** Which visual cues (region, texture, color) do these estimates rely on?

For both questions, we repeat the same experiments with 4 feature extractors (SIFT+BoVW, CLIP, DINOv2, DINOv3). We do not rank the extractors. An extractor is a measurement instrument, and we examine whether the direction of each conclusion is the same regardless of the extractor.

The contributions of this report are as follows.

1. We set out an evaluation procedure that scores proportion estimates for mixed visual categories against DNA, an objective ground truth. We report the primary metric together with the prior baseline, the upper bound, and coverage.
2. We compare three methods under the same conditions on frozen features. `[TBD: one sentence on the RQ1 result]`
3. We analyze which visual cues the estimates rely on, using cue transformations and controls. `[TBD: one sentence on the RQ2 result]`
4. We build a deliverable that presents the estimated proportions and their confidence in a family-tree-style view.

This study does not aim to replace DNA testing. Its scope is limited to dogs.

---

## 2. Related Work

### 2.1 Limits of Breed Identification from Appearance

Breed labels assigned from appearance at shelters and adoption agencies have been reported to disagree frequently with DNA test results (Voith et al., 2009; Voith et al., 2013; Olson et al., 2015). Gunter et al. (2018) examined the genetic breed composition of shelter dogs, and Morrill et al. (2022) analyzed the relationship between breed and individual traits using large-scale genomic data. These studies concern identification by people. We pose the same task for image features, and we estimate a proportion vector instead of naming a single breed.

Recent studies that treat images together with genetic information include E & Wang (2026 ※) and Ma et al. (2026 ※). `[VERIFY: check in the original papers how their task definitions differ from ours, then add one or two sentences]`

### 2.2 Learning with Mixed Labels

Label distribution learning maps an input to a distribution over labels instead of a single label (Geng, 2016; Gao et al., 2017). The mixup method creates training samples by mixing two inputs and their two labels in the same ratio (Zhang et al., 2018), and Manifold Mixup performs this mixing on hidden representations (Verma et al., 2019). Our mixture training (Section 5.4) follows this line of work. It differs in one respect: we do not use mixing as a regularization technique, and we use the mixture proportions themselves as the prediction target.

### 2.3 Unmixing and the Linear Structure of Representations

In hyperspectral imaging, linear spectral unmixing treats the spectrum of a pixel as a nonnegative weighted sum of pure-material spectra and solves for the weights. Fully constrained least squares (FCLS) is a solution that imposes the nonnegativity constraint and the sum-to-one constraint together (Heinz & Chang, 2001). Our prototype unmixing (Section 5.3) applies this solution directly in the image feature space. An analysis has also reported that compositions of concepts appear close to linear in learned representations (Trager et al., 2023). We test experimentally whether this linearity assumption is usable for breed mixtures.

### 2.4 Cue Bias

Geirhos et al. (2019) showed with cue-conflict experiments that CNNs trained on ImageNet rely on texture more than on shape. Analyses from a frequency perspective have examined how much models rely on high-frequency components (Yin et al., 2019; Wang et al., 2020). Our RQ2 applies this experimental approach to the proportion estimation task. The transformations target region, texture (blur), and color (grayscale), and each transformation has a control.

### 2.5 Self-Supervised and Classical Features

CLIP is a representation learned jointly from images and text (Radford et al., 2021), and DINOv2 and DINOv3 are self-supervised representations learned without labels (Oquab et al., 2024; Siméoni et al., 2025). These representations are widely used in frozen form, with only a lightweight layer added on top. SIFT (Lowe, 2004) and the bag of visual words (BoVW; Csurka et al., 2004) are classical features designed without learning. We include all four in the same experiments to see whether the conclusions change with the type of feature. We treat prediction confidence from the perspectives of calibration (Guo et al., 2017) and selective prediction (Geifman & El-Yaniv, 2017).

---

## 3. Problem Definition

### 3.1 Task and Notation

Let $\mathcal{B}$ denote the set of training breeds and $K$ its size. The ground truth for dog $d$ is the vector of per-breed proportions $y_d$ given by a DNA test, and $y_d$ may include breeds outside $\mathcal{B}$. The model takes the photos $x_{d,1}, \dots, x_{d,n_d}$ of dog $d$ and outputs a proportion vector $\hat{p}_d \in \Delta^{K-1}$ over $\mathcal{B}$ ($\hat{p}_{d,b} \ge 0$, $\sum_b \hat{p}_{d,b} = 1$).

Predictions are made for each photo and then averaged per dog. Scoring is also done per dog.

Training uses only photos that carry one breed label each. Photos of mixed-breed dogs and DNA proportions are not used for training.

### 3.2 Breed Space and Coverage

The model predicts in the class space of Tsinghua Dogs, and at scoring time a breed mapping table converts the predictions to the breed space of the DNA test. When several classes map to one breed, we sum the predicted proportions of those classes (Appendix B).

The coverage of dog $d$ is the sum of its DNA proportions that belong to training breeds.

$$c_d = \sum_{b \in \mathcal{B}} y_{d,b}$$

We use the ground-truth labels in two forms.

- Original proportions: we use $y_d$ as is. The model cannot, in principle, recover the share of breeds outside $\mathcal{B}$.
- Renormalized proportions: $\tilde{y}_{d,b} = y_{d,b} / c_d$ ($b \in \mathcal{B}$). These are the relative proportions within the training breeds.

Dogs with coverage below 0.5 are reported separately from the overall results.

### 3.3 Primary Metric and Reference Values

**Primary metric: captured ancestry share.** Let $T_d$ be the set of the 3 breeds with the largest predicted proportions. Then

$$s_d = \sum_{b \in T_d} y_{d,b}$$

and the score on the evaluation set is the mean of $s_d$ over dogs. We compute the score with the original proportions and with the renormalized proportions and report both. `[TBD: team decision, the label type used for confirmatory decisions. The project proposal uses renormalized proportions]`

We chose this metric for two reasons. First, it asks which breeds are present before it asks for the exact values of the predicted proportions, so methods can be compared even when their proportion scales differ. Second, it has the same form as surveys in which people choose a few breeds. However, we have not verified that the aggregation used in MuttMix matches our definition, so we do not compare its figure (20.9%) directly with our results `[VERIFY: the aggregation method in the original MuttMix source]`.

**Two reference values.** Every results table reports the following two values.

- Prior baseline: the score of a prediction that outputs the breed proportions of the training set without looking at the photo. It is the floor of the metric.
- Upper bound: the score obtained by choosing the 3 training breeds with the largest true proportions. It is the maximum that a model can attain.

**Secondary metrics.** We report distribution overlap $\sum_b \min(\hat{p}_{d,b}, \tilde{y}_{d,b})$, Top-1 agreement (the top predicted breed against the top true breed), recall by ancestry-share bin, KL divergence, and calibration. Confirmatory decisions rest on the primary metric only.

---

## 4. Data

### 4.1 Training: Tsinghua Dogs

We train on the low-resolution version of Tsinghua Dogs (Zou et al., 2020). The original dataset has 130 classes, and the public split has 65,228 images in train and 5,200 images in val (40 per class). We remove 3 classes and use 127 classes (61,592 images in train and 5,080 images in val; Table 1). We counted the number of images per class directly from the public split files (Appendix A).

**Table 1. Class handling in Tsinghua Dogs**

| Class | Train images | Handling | Reason |
|---|---|---|---|
| chinese_rural_dog | 3,336 | Excluded | The category groups dogs of several types and is not a single breed |
| dhole, African_hunting_dog | 209, 211 | Excluded | They are wild canids, not domestic dogs |
| dingo | 217 | Included `[TBD: team decision]` | Depends on whether it is in the breed list of the DNA test |
| teddy, toy / miniature / standard_poodle | 7,449, `[VERIFY: train image counts of the 3 poodle classes]` | Trained as is; merged into Poodle at scoring | teddy appears to be a grooming style of the poodle `[VERIFY: check the photos]` |

The breed labels in Tsinghua Dogs were assigned by looking at photos. The dogs are not confirmed as purebred by pedigree certificates or DNA, so this report calls the photos "breed-labeled photos." For training and feature extraction, we use photos cropped with the whole-body bounding box that the dataset provides.

### 4.2 Mixed-Breed Evaluation Set

Within the scope of our search, we found only one public source that provides photos, per-dog DNA breed proportions, and terms that permit our use. It is the material from a dog breed identification study published by the Shelter Medicine Program at the University of Florida (UF Shelter Medicine ※). The material contains photos and DNA test results for 100 shelter dogs. By our classification, 97 are mixed-breed dogs and 3 are single-breed dogs.

**Terms of use.** The material permits storage for educational purposes but requires written permission for publication. We therefore keep the photos and the per-dog DNA table out of the repository, the report, the presentation materials, and the demo, and we report only aggregate numbers. The email requesting permission to publish: `[TBD: whether it was sent, and the reply]`.

**Photo processing.** Each dog has one original photo (about 448×195 pixels), which places the face on the left and the full body on the right in a single image. We split it into a face panel (about 157×183) and a full-body panel (about 259×183), giving 2 images per dog (200 images in total). We average the predictions for the two panels to obtain the prediction for the dog.

**Ground-truth processing.** We map the breed names in the DNA table to Tsinghua classes with the breed mapping table. The mapping table is currently a draft, and some entries are not settled (Appendix B). We will fill in the coverage distribution of the 97 mixed-breed dogs in Table 2 after the mapping is settled.

**Table 2. Coverage distribution of the 97 mixed-breed dogs** (to be filled in after the mapping is settled)

| Coverage | Without similar-breed mappings | With similar-breed mappings |
|---|---|---|
| Below 0.5 | | |
| 0.5 to below 0.8 | | |
| 0.8 or above | | |

**Characteristics that limit the evaluation.** The photos have low resolution, and each dog appears in only one scene. The company that ran the DNA tests and the breed panel at the time of testing: `[VERIFY: the UF page and the related paper]`. Section 9 discusses these points.

**Additional collection.** Other sources, such as posts in online communities, are not included in the current evaluation set because the right to use them has not been confirmed `[TBD: team decision, whether to collect more]`.

### 4.3 Purebred Auxiliary Evaluation Set

The mixed-breed evaluation set is small, and many of its dogs have low coverage. We therefore add an auxiliary evaluation of how each method behaves on single-breed inputs. We split the 5,080 Tsinghua val images in half within each class, giving 2,540 dev images and 2,540 test images (seed 1).

In this evaluation, the ground truth assigns 1 to a single breed. The metrics are the proportion assigned to the true breed, Top-1 accuracy, and the effective number of breeds (the exponential of the entropy of the predicted distribution; it equals 1 when the entire prediction goes to one breed). We report these results in a table separate from the mixed-breed results and do not combine the two into one number.

### 4.4 Splits and Overlap Prevention

**Table 3. Data splits**

| Data | Use | Size |
|---|---|---|
| Tsinghua train | Training, prototype computation | 61,592 images, 127 classes |
| Tsinghua val (dev half) | Purebred auxiliary evaluation, configuration selection | 2,540 images |
| Tsinghua val (test half) | Purebred auxiliary evaluation, final | 2,540 images |
| UF mixed-breed dev | Configuration selection | 20 dogs (40 images) |
| UF mixed-breed test | Final scoring, once | |
| UF single-breed | Reference | 3 dogs |

The 20 mixed-breed dev dogs were drawn at random from the 97 mixed-breed dogs (seed 1). The split is made per dog: the two panels of a dog always fall in the same split. The formal evaluation-set index and the test split: `[TBD: owner A, eval_index.csv]`.

---

## 5. Methods

### 5.1 Overview and Feature Extractors

All methods follow the same sequence. We freeze the feature extractor, extract a feature vector from each photo once, and store it. We then run the three methods on these features. Differences between methods therefore arise only from the method and not from the features.

**Table 4. The 4 feature extractors (all frozen)**

| Extractor | Type | Feature vector |
|---|---|---|
| SIFT + BoVW | Local features designed without learning | SIFT descriptors pooled into a visual-word histogram. Codebook size and keypoint scheme: `[TBD: owner C]` |
| CLIP ViT-B/16 | Trained jointly on images and text | Image embedding after projection |
| DINOv2 ViT-B/14 | Self-supervised learning | CLS token |
| DINOv3 ViT-B/16 | Self-supervised learning | CLS token |

For the 3 deep features, we use public weights, and input preprocessing follows the default preprocessor of each model `[VERIFY: input resolution values]`. We also store the mean of the patch tokens, but the main experiments use the CLS feature. Whether to L2-normalize the features is decided on dev `[TBD: team decision]`.

### 5.2 Classifier

The classifier trains a single linear layer on the features to classify the 127 breeds. At inference, we read the softmax probabilities as breed proportions. A temperature $\tau$ controls how strongly the probabilities concentrate on one breed, and $\tau$ is set on dev.

$$\hat{p} = \mathrm{softmax}(Wf + b \,/\, \tau)$$

Here $f$ is the feature vector. The training configuration (optimization, epochs, 3 seeds): `[TBD: owner A]`.

This method tests whether the probabilities of a model trained only for single-breed classification can also be read as mixture proportions.

### 5.3 Prototype Unmixing

Prototype unmixing involves no training. We first average the features of the training photos of breed $k$ to obtain the breed prototype $\mu_k$. We then solve for the weights that approximate the feature $f$ of a mixed-breed photo as a weighted sum of the prototypes.

$$\hat{w} = \arg\min_{w} \Big\lVert f - \sum_{k=1}^{K} w_k \mu_k \Big\rVert_2^2 \quad \text{s.t.}\quad w_k \ge 0,\ \sum_k w_k = 1$$

This problem is FCLS (Heinz & Chang, 2001). Our implementation solves nonnegative least squares (NNLS) with the sum-to-one constraint appended as one extra row weighted by $\delta$. We set $\delta$ to 10 times the largest absolute value among the entries of the prototype matrix. After solving, we divide the weights once more by their sum so that they sum exactly to 1. The vector $\hat{w}$ is the breed proportion vector $\hat{p}$.

This method has no training seed. Instead, we build the prototypes from a random subsample of 80% of the training photos and repeat this 3 times to measure the variation that comes from estimating the prototypes.

We also record the reconstruction error $\lVert f - \sum_k \hat{w}_k \mu_k \rVert_2 / \lVert f \rVert_2$ for each photo. It is a relative error, divided by the norm of the feature. This value indicates how well a convex combination of the prototypes explains the input, and it is a candidate for the confidence display in the deliverable (Section 8).

This method tests whether the assumption that the feature of a mixed-breed dog lies close to a convex combination of the mean features of its parent breeds is usable with a fixed linear solution alone.

### 5.4 Mixture Training

Mixture training learns from synthetic mixtures created in the feature space. We draw one training-photo feature from each of 2–4 different breeds and mix the features with proportions $\lambda$. The mixed feature $\sum_j \lambda_j f_j$ is the input, and the vector that places $\lambda$ at the positions of those breeds is the ground truth. The model has the same linear structure as the classifier, and the loss is the KL divergence between the predicted distribution and the ground-truth distribution.

The distribution of mixture proportions, the distribution of the number of breeds, and the training configuration (3 seeds): `[TBD: owner C]`.

Because the structure is the same as that of the classifier, the difference between the two methods arises only from the training objective (single labels versus mixture proportions).

### 5.5 Cue Transformations and Controls (RQ2)

RQ2 transforms the evaluation photos, predicts again with the same method, and measures how much the score drops. Only one transformation is applied at a time.

**Table 5. Cue transformations and controls**

| Cue | Transformation | Control |
|---|---|---|
| Region | Head only / body only / background only | Random mask of the same area |
| Texture | Gaussian blur at 4 levels | PSNR-matched noise |
| Color | Grayscale conversion | PSNR-matched noise |

We measure cue reliance with the excess drop.

$$\text{excess drop} = (\text{original score} - \text{transformed score}) - (\text{original score} - \text{control score})$$

The control removes the drop caused by the mere fact that the photo has been degraded. We interpret the excess drop as a difference between breed groups, not as an absolute value. The breed groups are 2–3 groups defined by appearance attributes `[TBD: owner D, definition of the breed groups]`.

SIFT is computed on the intensity image, so its score should barely change under the grayscale transformation. We use this as a check that the transformation experiments run correctly.

How to divide the regions: `[TBD: owner D. The candidates are SAM 3 (Carion et al., 2025 ※) masks, or using the face panel and the full-body panel of the UF photos as they are]`. The values of the 4 blur levels and the methods to which the transformations apply (the classifier only, or all three methods) are also `[TBD: owner D]`.

### 5.6 Combined Model

After obtaining the individual results of the three methods, we build one model that combines their predictions. The combination scheme (for example, a weighted average of the predictions of the three methods) and the weights are set on dev `[TBD: team decision]`. The combined model is not used for confirmatory decisions on the research questions. Its purpose is to choose the configuration for the deliverable (Section 8), and its results are reported separately in Section 7.4.

---

## 6. Experimental Setup

### 6.1 Procedure

1. Extract and store the features (4 extractors × training set, val, evaluation set, transformed evaluation sets).
2. Run each method on dev and set its configuration (temperature, whether to apply L2 normalization, mixture distribution, and so on).
3. Fix the configuration and record it in the decision log.
4. Score test only once per configuration.

Only the configuration of each method is set on dev. We do not use dev scores to select an extractor or a method for our conclusions. After we see test scores, we do not change the configuration.

### 6.2 Repeats and Statistics

- **Repeats.** We run the classifier and mixture training with 3 seeds and prototype unmixing with 3 subsamples, and we report the mean and standard deviation.
- **Confidence intervals.** We compute 95% confidence intervals with a dog-level bootstrap `[TBD: owner A, number of resamples]`.
- **Chance-level check.** We randomly shuffle the pairing between dogs and DNA ground truth 1,000 times to build a chance distribution of the score, and we check whether the actual score is higher than this distribution. The check also serves to detect misaligned pairs or mapping errors in the pipeline. If the check fails, we first look for errors, and if we find none, we make no claim about differences between methods.
- **Confirmatory comparisons.** We test the between-method comparisons of RQ1 and the excess-drop comparisons of RQ2 on DINOv3 and apply the Holm correction (Holm, 1979) `[TBD: owner A, test method. The candidates are a dog-level paired bootstrap or a paired permutation test]`.

### 6.3 Extractor Consistency Rule

Confirmatory decisions are made on DINOv3 only. For the other three extractors, we look only at the sign of the difference in the same comparison. If the signs for all four extractors agree, we report "consistent"; otherwise, we report "extractor-dependent." The extractor used for confirmatory decisions was fixed before the experiments and is not changed on the basis of dev results.

### 6.4 Compute Environment and Reproducibility

- The runtime environment is a Colab T4 GPU. Deep feature extraction ran at about 130–156 images per second on the T4 (measured with DINOv3) and took about 9 minutes per extractor for the training set.
- Features are stored in shards of 5,000 images. If a run is interrupted, it skips the finished shards and resumes.
- Experiment logic lives only in the `.py` files of the repository, and the notebook only runs them.
- Each run writes one line to `logs/runs.jsonl` with its run name, date, owner, commit, extractor, configuration, seed, data version, split, and metrics. Failed runs are not deleted (Appendix C).
- The reproduction procedure is given in Appendix D.

---

## 7. Results

This section contains only table templates. The empty cells will be filled in after the formal scoring.

### 7.1 RQ1: Method × Feature Extractor

**Table 6. Captured ancestry share on the mixed-breed test split** `[TBD: number of dogs, label type]`

| Method | SIFT+BoVW | CLIP | DINOv2 | DINOv3 |
|---|---|---|---|---|
| Prior baseline | | | | |
| Classifier | | | | |
| Prototype unmixing | | | | |
| Mixture training | | | | |
| Upper bound | | | | |

The prior baseline and the upper bound do not depend on the extractor, so their values are the same in all four cells. Each cell is the mean over 3 seeds (3 subsamples for prototype unmixing).

**Table 7. Differences between methods (DINOv3, Holm-corrected) and extractor consistency**

| Comparison | Difference [95% CI] | Corrected p | Signs for the four extractors | Decision |
|---|---|---|---|---|
| Prototype unmixing − classifier | | | | |
| Mixture training − classifier | | | | |
| Mixture training − prototype unmixing | | | | |

**Table 8. Captured ancestry share by coverage bin** `[TBD]`

Result of the chance-level check: `[TBD: owner A]`. Table of secondary metrics: `[TBD]`.

### 7.2 Purebred Auxiliary Evaluation

**Table 9. Test half of Tsinghua val (2,540 images)** `[TBD]`

| Method | Extractor | Proportion assigned to the true breed | Top-1 | Effective number of breeds |
|---|---|---|---|---|
| | | | | |

### 7.3 RQ2: Cue Reliance

**Table 10. Excess drop by transformation** `[TBD: owner D]`

| Cue | Transformation | Breed group 1 | Breed group 2 | Difference between breed groups [95% CI] |
|---|---|---|---|---|
| | | | | |

Result of the SIFT grayscale check: `[TBD: owner D]`.

### 7.4 Combined Model

`[TBD: after the team decision]`

---

## 8. Deliverable: A Family-Tree-Style View of Estimated Ancestry

This section describes the design only. The implementation and screen examples: `[TBD: owner B]`.

The deliverable is a view that takes one photo and shows the estimated breed proportions in the shape of a family tree (Gradio). The view makes the research results visible, and each element of the view is grounded in the experimental results (Table 11).

**Table 11. View elements and their basis**

| View element | Basis | Content |
|---|---|---|
| Breed proportions → 8 great-grandparent slots | RQ1 | Compute the proportions with the configuration chosen on dev, round them to units of 12.5%, and place them in the slots |
| Confidence display, abstention | RQ1, coverage analysis | When the prediction is uncertain, show the abstention screen instead of the proportions. The criteria (the spread of the predicted distribution, the reconstruction error of prototype unmixing): `[TBD: owner B, decided on dev]` |
| "How this was estimated" explanation | RQ2 | Shows which cues the estimate relies on `[TBD: depending on the RQ2 results]` |
| "A depiction of proportions, not a pedigree" notice | Limits of the task | Proportions alone do not determine a single actual pedigree |

The last item must appear in the view. For example, an estimate that a breed accounts for 50% may mean that one parent is of that breed, or it may mean that two grandparents, across both sides of the family, are of that breed. The family-tree layout in the view is a way of depicting proportions and is not the result of estimating a pedigree.

The view carries the statement "A research estimate that does not replace DNA testing." The demo does not use UF photos. The configuration used in the demo is chosen from dev results, and this choice is separate from the confirmatory decisions on the research questions.

---

## 9. Limitations

The following limitations can be stated now, regardless of the results. Limitations revealed by the results: `[TBD: to be added after the results]`.

- **Size of the evaluation set.** There are 97 mixed-breed dogs, and test has fewer. If the differences between methods are small, they may remain inconclusive. Per-breed analyses are reported descriptively only.
- **Quality of the evaluation photos.** The evaluation photos have low resolution, and each dog appears in only one scene. Because the shooting conditions differ from those of the training photos, it is difficult to separate whether a low score comes from the difficulty of the task or from the difference in photo conditions.
- **A single source.** All evaluation dogs are shelter dogs from one region. We cannot tell whether other populations would give the same results.
- **Coverage.** Some dogs include breeds that are not among the training breeds (Table 2). For dogs with low coverage, the score under the original proportions is low in principle.
- **Training labels.** The breed labels in Tsinghua Dogs were assigned from appearance, so the classes may contain mixed-breed dogs. Some pairs of classes are difficult to distinguish by appearance.
- **Uncertainty in the ground truth.** DNA test results also depend on the reference panel of the testing company. We use them as the ground truth, but they are not free of error. Some dogs have notation problems, such as proportions that sum to more than 100% (Appendix B).
- **Breed mapping.** Mapping the breed names of the DNA test to Tsinghua classes involves judgment. Appendix B lists the mapping rules and the unsettled items.
- **Scope of the task.** Ancestry that does not show in appearance cannot be recovered from photos. This study does not replace DNA testing.

---

## 10. Conclusion

`[TBD: to be written after the results are available]`

---

## References

※ marks an entry whose authors, title, and venue must be rechecked against the original source. The final version will keep only the works cited in the text.

**Dog breeds and identification from appearance**

- Gunter L. M., Barber R. T., Wynne C. D. L. (2018). A canine identity crisis: Genetic breed heritage testing of shelter dogs. *PLoS ONE*.
- Lord K. et al. (2025). [title to be verified; MuttMix study of human breed guesses] ※
- Morrill K. et al. (2022). Ancestry-inclusive dog genomics challenges popular breed stereotypes. *Science*.
- Olson K. R. et al. (2015). Inconsistent identification of pit bull-type dogs by shelter staff. *The Veterinary Journal*.
- UF Shelter Medicine Program. Dog Breed Identification: Dog Breed DNA and Survey Results. University of Florida web page ※ (publication year and related paper to be verified)
- Voith V. L. et al. (2009). Comparison of adoption agency breed identification and DNA breed identification of dogs. *Journal of Applied Animal Welfare Science*.
- Voith V. L. et al. (2013). Comparison of visual and DNA breed identification of dogs and inter-observer reliability. *American Journal of Sociological Research*.
- E G.-X., Wang G.-D. (2026). Image feature embedding with a deep learning framework improves genome-wide association studies on dog endophenotypes ※ (venue to be verified)
- Ma et al. (2026). [title to be verified; estimating mixed-ancestry proportions from gayal images]. *bioRxiv* ※

**Learning methods and linear structure**

- Gao B.-B. et al. (2017). Deep label distribution learning with label ambiguity. *IEEE Transactions on Image Processing*.
- Geng X. (2016). Label distribution learning. *IEEE Transactions on Knowledge and Data Engineering*.
- Heinz D. C., Chang C.-I. (2001). Fully constrained least squares linear spectral mixture analysis method for material quantification in hyperspectral imagery. *IEEE Transactions on Geoscience and Remote Sensing*.
- Trager M. et al. (2023). Linear spaces of meanings: Compositional structures in vision-language models. *ICCV*.
- Verma V. et al. (2019). Manifold Mixup: Better representations by interpolating hidden states. *ICML*.
- Zhang H. et al. (2018). mixup: Beyond empirical risk minimization. *ICLR*.

**Cue analysis, confidence, and statistics**

- Geifman Y., El-Yaniv R. (2017). Selective classification for deep neural networks. *NeurIPS*.
- Geirhos R. et al. (2019). ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness. *ICLR*.
- Guo C. et al. (2017). On calibration of modern neural networks. *ICML*.
- Holm S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*.
- Wang H. et al. (2020). High-frequency component helps explain the generalization of convolutional neural networks. *CVPR*.
- Yin D. et al. (2019). A Fourier perspective on model robustness in computer vision. *NeurIPS*.

**Models and data**

- Carion N. et al. (2025). SAM 3: Segment anything with concepts. *arXiv* ※
- Csurka G. et al. (2004). Visual categorization with bags of keypoints. *ECCV Workshop on Statistical Learning in Computer Vision*.
- Lowe D. G. (2004). Distinctive image features from scale-invariant keypoints. *International Journal of Computer Vision*.
- Oquab M. et al. (2024). DINOv2: Learning robust visual features without supervision. *Transactions on Machine Learning Research*.
- Radford A. et al. (2021). Learning transferable visual models from natural language supervision. *ICML*.
- Siméoni O. et al. (2025). DINOv3. *arXiv*.
- Zou D.-N. et al. (2020). A new dataset of dog breed images and a benchmark for fine-grained classification. *Computational Visual Media*.

---

## Appendix A. Class Handling in Tsinghua Dogs

- We counted the images per class from the public split files and recorded all 130 classes in `tsinghua_breeds.csv` `[TBD: move into a table]`.
- The 3 excluded classes and the reasons for excluding them are as in Table 1.
- Implementation notes: the `Fila Braziliero` class has a space in its folder name. The split files contain a BOM. The body-box tag in the annotation files is `bodybndbox`.

## Appendix B. Breed Mapping Rules

**Rules.**

1. Map each breed name in the DNA table to 0 or more Tsinghua classes.
2. When several classes map to one breed, sum the predicted proportions (for example, teddy, toy_poodle, miniature_poodle, standard_poodle → Poodle).
3. A breed with no matching class is treated as outside the training breeds and left out of the coverage calculation.
4. A "similar-breed mapping" (mapping a breed to a close breed that has a different name) is marked separately, and we produce results both with and without these mappings.

**Unsettled items (draft mapping).** `[TBD: to be settled with owner A]`

- 6 similar-breed mappings
- Whether to merge teddy into Poodle
- The breeds to which the Black_sable and Eskimo_dog classes map
- Whether to include dingo
- Notation problems in the DNA table: 2 entries with garbled proportion notation, and 2 dogs whose proportions sum to more than 100%. We will define a handling rule and record it.

## Appendix C. Run Names and Log Format

- Run name: `RQ-extractor-method-seed`. Examples: `RQ1-dinov3-proto-r1`, `RQ1-dinov3-mix-s1`. Feature extraction uses `FEAT-extractor-data`.
- Features: a folder for each extractor and dataset, holding shards of 5,000 images and an index file.
- Predictions: a folder for each split, with one CSV per run (per dog, Tsinghua class space, rows sum to 1).
- Logs: `logs/runs.jsonl`, one run per line.

## Appendix D. Reproduction Procedure

`[TBD: owner B. Move the run order from the repository README: environment setup → data check → feature extraction → per-method runs → scoring]`

We do not redistribute the UF material. To reproduce the work, one must download it directly from the UF page, and the repository contains only the download and preprocessing scripts.
