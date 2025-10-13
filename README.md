1. Executive Summary
This report details a multimodal price prediction solution developed for the ML Challenge 2025. Our approach integrates fine-tuned text embeddings from a BERT model and image embeddings from a CLIP ViT-B/32 model, augmented with engineered numerical features. These multimodal features are processed by a LightGBM regressor trained with a 5-fold cross-validation strategy. The resulting model is robust and stable, achieving a cross-validation Symmetric Mean Absolute Percentage Error (SMAPE) of 30.81% and a Mean Absolute Error (MAE) of approximately $6.74 on the original price scale.
2. Methodology
2.1 Problem Analysis and Feature Engineering
The task involved predicting product prices using a dataset of 75,000 training samples, each containing product catalog content, an image link, and a price. Exploratory Data Analysis (EDA) revealed key patterns:
• Target Distribution: The price variable was highly right-skewed. We applied log1p transformation for stability. • Textual Data: Catalog content contained brand names and quantities, offering strong predictive signals. • Image Data: Product images provided complementary visual cues but were noisy when used alone. • Numerical Features: Features such as total_weight_g, pack_qty, and pieces were extracted using regex.
2.2 Solution Pipeline
Our hybrid architecture combines deep learning encoders and a gradient boosting regressor: 1. Text Embedding: Fine-tuned BERT for catalog_content producing 768D embeddings. 2. Image Embedding: CLIP ViT-B/32 encoder fine-tuned for visual similarity, producing 512D embeddings. 3. Feature Fusion: Weighted concatenation (Text: 0.6, Image: 0.32, Numerical: 0.08). 4. Regression: LightGBM trained on fused features with 5-fold CV, predicting log1p(price).
3. Model Architecture Details
3.1 Text Pipeline
Encoder: bert-base-uncased, fine-tuned for price regression. Preprocessing: Lowercasing with minimal cleaning. Tokenization: Max length 128. Embedding: [CLS] token representation (768D).
3.2 Image Pipeline
Encoder: open_clip ViT-B/32 fine-tuned on product domain. Preprocessing: Standard CLIP transformations. Fallback: White image for missing data. Embedding: 512D vector extracted reliably in chunks.
3.3 Numerical Feature Pipeline
Features: Extracted using regex (total_weight_g, pack_qty, pieces, percent_value). Scaling: StandardScaler applied for magnitude consistency.
3.4 Final Regression Model
Model: LightGBM Regressor on log1p(price). Cross-Validation: 5-fold, stratified by price quantiles. Key Parameters: learning_rate=0.05, num_leaves=127, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8.
4. Performance and Results
The model achieved consistent 5-fold CV results: • Overall SMAPE: 30.81% • Overall MAE: $6.74 (on original scale) • Fold SMAPE: 30.73%, 30.79%, 30.87%, 31.04%, 30.63% The SMAPE standard deviation of 0.14% indicates excellent stability and generalization.
5. Conclusion
We developed a reproducible, multimodal pipeline combining text, image, and numerical data for price prediction. By fine-tuning domain-specific encoders and fusing their representations for LightGBM regression, we achieved a stable SMAPE of 30.8%. Future work can explore advanced multimodal fusion and richer numerical feature extraction.
