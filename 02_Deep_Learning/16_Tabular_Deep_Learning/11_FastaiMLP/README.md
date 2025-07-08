# FastaiMLP: Fast.ai Multi-Layer Perceptron for Tabular Data

## Overview
FastaiMLP is a fast.ai framework implementation of multi-layer perceptrons designed for tabular data. Ranking 10th in TabArena with an Elo score of 1300, it leverages the fast.ai library's high-level API and best practices for accessible yet powerful deep learning on structured data.

## Key Features
- **Fast.ai Framework**: Built on fast.ai's high-level abstractions
- **Easy to Use**: Simplified API for rapid experimentation
- **Best Practices**: Incorporates fast.ai's proven training techniques
- **Transfer Learning**: Benefits from fast.ai's transfer learning capabilities

## Performance Characteristics
- **TabArena Ranking**: #10 with Elo score of 1300
- **Default Performance**: Reasonable baseline (~1000 Elo)
- **Tuning Benefits**: Good improvement with optimization (+200 points)
- **Ensemble Gains**: Solid ensemble contribution (+300 points from default)

## Architecture
Fast.ai tabular model architecture:
- Automated preprocessing pipelines
- Embedding layers for categorical variables
- Configurable neural network depth
- Built-in regularization techniques
- One-cycle learning rate scheduling

## Use Cases
- **Rapid Prototyping**: Quick experimentation with tabular deep learning
- **Educational**: Learning deep learning with high-level APIs
- **Fast.ai Ecosystem**: Integration with fast.ai workflows
- **Beginner Friendly**: Accessible entry point to tabular neural networks

## Implementation
Leverages fast.ai's tabular module:
- Automatic data preprocessing
- TabularModel architecture
- Built-in data augmentation
- Transfer learning capabilities
- Integrated model interpretation tools

## Fast.ai Advantages
- **High-Level API**: Simplified model creation and training
- **Automatic Optimization**: Built-in best practices
- **Rich Ecosystem**: Integration with fast.ai tools and workflows
- **Educational Resources**: Excellent documentation and courses

## Mathematical Framework

### Data Preprocessing Pipeline
FastaiMLP uses fast.ai's automated preprocessing:

**Categorical Processing:**
```
x_cat_processed = Categorify(x_cat) → LabelEncoder
E_cat = Embedding(x_cat_processed) ∈ ℝᵈᵉᵐᵇ
```

**Numerical Processing:**
```
x_num_processed = Normalize(FillMissing(x_num))
x_num = (x_num - μ) / σ
```

**Combined Input:**
```
X_input = Concat(x_num_processed, E_cat₁, E_cat₂, ..., E_catₘ)
```

### TabularModel Architecture
Fast.ai's TabularModel implements a sophisticated MLP:

**Layer Configuration:**
```
layers = [input_dim, layer₁_dim, layer₂_dim, ..., output_dim]
```

**Hidden Layer Computation:**
```
h⁽ˡ⁾ = Dropout(Activation(BatchNorm1d(Linear(h⁽ˡ⁻¹⁾))))
```

Where the full layer is:
```
h⁽ˡ⁾ = Dropout(ReLU(BN(W⁽ˡ⁾h⁽ˡ⁻¹⁾ + b⁽ˡ⁾)), p_drop)
```

### Embedding Dimension Calculation
Fast.ai uses a heuristic for embedding dimensions:

```
emb_dim = min(50, cardinality//2)
```

For categorical feature with cardinality c:
```
d_emb = min(50, ⌊c/2⌋ + 1)
```

### Batch Normalization (1D)
Applied to tabular data:

```
BN1d(x) = γ · (x - μ_batch) / √(σ²_batch + ε) + β
```

With momentum update:
```
μ_running = momentum · μ_running + (1-momentum) · μ_batch
σ²_running = momentum · σ²_running + (1-momentum) · σ²_batch
```

### One Cycle Learning Rate Policy
FastaiMLP uses the one-cycle learning rate schedule:

**Phase 1 (0 to pct_start):**
```
η(t) = η_min + (η_max - η_min) · t/pct_start
```

**Phase 2 (pct_start to 1):**
```
η(t) = η_max - (η_max - η_min) · (t - pct_start)/(1 - pct_start)
```

**Momentum Schedule (inverse of learning rate):**
```
momentum(t) = momentum_max - (momentum_max - momentum_min) · η_ratio(t)
```

### Loss Functions with Label Smoothing
FastaiMLP optionally applies label smoothing:

**Standard Cross-Entropy:**
```
L_CE = -∑ᵢ ∑ⱼ yᵢⱼ log(ŷᵢⱼ)
```

**Label Smoothing:**
```
y_smooth = (1 - α) · y_true + α/K
L_smooth = -∑ᵢ ∑ⱼ y_smooth_ij log(ŷᵢⱼ)
```

Where α is the smoothing parameter and K is the number of classes.

### Weight Decay Regularization
Applied to all parameters except biases and batch norm:

```
W_new = W - η(∇L + λ_wd · W)
```

### Mixup Data Augmentation
For tabular data augmentation:

```
x_mixed = λ · x_i + (1-λ) · x_j
y_mixed = λ · y_i + (1-λ) · y_j
```

Where λ ~ Beta(α, α) with α typically 0.4.

### Progressive Resizing/Layer Unfreezing
Fast.ai implements progressive training:

**Layer Unfreezing Schedule:**
```
For epoch e in unfreezing_schedule:
    unfreeze_layers(model, up_to_layer=layer_schedule[e])
```

### Discriminative Learning Rates
Different learning rates for different layer groups:

```
η_layer_group_i = η_base / (discriminative_factor^i)
```

### Automatic Mixed Precision (AMP)
Using PyTorch's AMP:

```
with autocast():
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
```

### Feature Importance via Permutation
Fast.ai implements permutation importance:

```
Importance_i = Score_original - Score_permuted_i
```

Where feature i is randomly permuted.

### Model Interpretation
**Partial Dependence:**
```
PD_i(x) = E[f(x₁, ..., xᵢ=x, ..., xₙ)]
```

**SHAP Values Integration:**
```
SHAP_i = φᵢ = ∑_{S⊆F\{i}} (|S|!(|F|-|S|-1)!)/|F|! · (f(S∪{i}) - f(S))
```

### Transfer Learning for Tabular Data
Pre-training on larger datasets:

```
θ_pretrained → fine_tune(θ_pretrained, task_data, η_reduced)
```

### Ensemble via Test Time Augmentation (TTA)
```
ŷ_ensemble = (1/K) ∑ₖ₌₁ᴷ f(Augment_k(X_test))
```

### Gradient Clipping
For stable training:

```
∇θ_clipped = ∇θ · min(1, clip_norm / ||∇θ||₂)
```

## Advanced Fast.ai Features

### DataLoaders and Transforms
```python
# Fast.ai's tabular data processing
procs = [Categorify, FillMissing, Normalize]
dls = TabularDataLoaders.from_df(
    df, path=path, 
    procs=procs,
    cat_names=cat_names,
    cont_names=cont_names,
    y_names=target,
    bs=batch_size
)
```

### Model Architecture Configuration
```python
model = tabular_learner(
    dls, 
    layers=[200, 100],  # Hidden layer sizes
    emb_drop=0.1,       # Embedding dropout
    y_range=None,       # Output range for regression
    use_bn=True,        # Batch normalization
    bn_final=False,     # BN after final layer
)
```

### Learning Rate Finder
Fast.ai's learning rate finder:

```
lr_finder = learner.lr_find()
optimal_lr = lr_finder.valley  # Steepest gradient point
```

Mathematical formulation:
```
loss_rate = d(loss)/d(lr)
optimal_lr = argmin(loss_rate)
```

### Cross-Validation Integration
```python
def cross_validate(df, folds=5):
    scores = []
    for fold in range(folds):
        # Split data, train model, evaluate
        score = train_fold(df, fold)
        scores.append(score)
    return np.mean(scores), np.std(scores)
```

## Advantages

### Ease of Use
- **High-Level API**: Minimal code for complex functionality
- **Automatic Preprocessing**: Built-in data cleaning and preparation
- **Sensible Defaults**: Optimized hyperparameters out of the box
- **Integrated Workflow**: End-to-end pipeline from data to deployment

### Training Optimizations
- **One-Cycle Learning**: Proven fast training methodology
- **Progressive Training**: Gradual complexity increase
- **Mixed Precision**: Automatic FP16 training for speed
- **Data Augmentation**: Built-in tabular data augmentation

### Model Interpretation
- **Feature Importance**: Multiple importance calculation methods
- **Partial Dependence**: Built-in partial dependence plots
- **SHAP Integration**: Native SHAP value computation
- **Model Debugging**: Comprehensive model inspection tools

## Limitations

### Framework Constraints
- **Fast.ai Dependency**: Tied to fast.ai ecosystem
- **Limited Customization**: Less flexibility than raw PyTorch
- **API Changes**: Subject to fast.ai version updates
- **Learning Curve**: Requires fast.ai knowledge

### Performance Considerations
- **Memory Usage**: Higher memory usage than minimal implementations
- **Overhead**: Framework abstractions add computational cost
- **Model Size**: Larger models due to framework components

## Implementation Examples

### Basic Model Training
```python
from fastai.tabular.all import *

# Load and preprocess data
path = Path('data')
df = pd.read_csv(path/'train.csv')

procs = [Categorify, FillMissing, Normalize]
dls = TabularDataLoaders.from_df(
    df, path=path,
    procs=procs,
    cat_names=['cat_col1', 'cat_col2'],
    cont_names=['num_col1', 'num_col2'],
    y_names='target',
    valid_idx=valid_idx,
    bs=512
)

# Create and train model
learn = tabular_learner(dls, layers=[200, 100])
learn.fit_one_cycle(10, lr_max=1e-2)
```

### Advanced Configuration
```python
# Custom model with advanced settings
learn = tabular_learner(
    dls,
    layers=[400, 200, 100],  # Deeper network
    emb_drop=0.1,            # Embedding dropout
    ps=[0.2, 0.3, 0.4],      # Layer-specific dropout
    use_bn=True,             # Batch normalization
    bn_final=False,          # No BN on output
    lin_first=False,         # Embedding first
    y_range=(0, 1)           # Output range constraint
)

# Advanced training
learn.fit_one_cycle(
    20,                      # Epochs
    lr_max=slice(1e-4, 1e-2), # Discriminative LR
    wd=0.1,                  # Weight decay
    pct_start=0.3            # One-cycle schedule
)
```

### Model Interpretation
```python
# Feature importance
interp = TabularInterpretation.from_learner(learn)
interp.plot_importance()

# Partial dependence
interp.plot_partial_dependence('feature_name')

# SHAP values
interp.plot_shap_values(df.iloc[:100])
```

## Hyperparameter Guidelines

### Architecture Parameters
- **Layers**: [200, 100] for small datasets, [800, 400, 200] for large
- **Embedding Dropout**: 0.04-0.1 depending on categorical cardinality
- **Hidden Dropout**: 0.1-0.5, increasing with layer depth
- **Batch Normalization**: Generally True for tabular data

### Training Parameters
- **Learning Rate**: Use lr_find(), typically 1e-3 to 1e-2
- **Weight Decay**: 0.01-0.1 for regularization
- **Batch Size**: 256-1024 depending on memory and dataset size
- **Epochs**: 10-50 with early stopping

### Data Processing
- **Missing Value Handling**: FillMissing with median/mode
- **Normalization**: StandardScaler via Normalize transform
- **Categorical Encoding**: Automatic via Categorify

## Production Deployment

### Model Export
```python
# Export trained model
learn.export('model.pkl')

# Load for inference
learn_inf = load_learner('model.pkl')
predictions = learn_inf.predict(new_data)
```

### ONNX Export
```python
# Convert to ONNX for cross-platform deployment
dummy_input = torch.randn(1, input_size)
torch.onnx.export(
    learn.model,
    dummy_input,
    'fastai_model.onnx',
    export_params=True
)
```

### Batch Inference
```python
# Efficient batch prediction
test_dl = learn.dls.test_dl(test_df)
preds, targets = learn.get_preds(dl=test_dl)
```

## Integration with Fast.ai Ecosystem

### Callback System
```python
# Custom callbacks for monitoring
class CustomCallback(Callback):
    def before_epoch(self):
        # Custom logic before each epoch
        pass
    
    def after_batch(self):
        # Custom logic after each batch
        pass

learn.fit(10, cbs=[CustomCallback()])
```

### Weights & Biases Integration
```python
import wandb
from fastai.callback.wandb import WandbCallback

wandb.init(project="tabular_project")
learn.fit(10, cbs=[WandbCallback()])
```

## Research Applications

### Hyperparameter Optimization
```python
from fastai.callback.tensorboard import TensorBoardCallback

# Grid search with fast.ai
for lr in [1e-4, 1e-3, 1e-2]:
    for wd in [0.01, 0.1, 1.0]:
        learn = tabular_learner(dls, layers=[200, 100])
        learn.fit(10, lr, wd=wd, cbs=[TensorBoardCallback()])
```

### Transfer Learning Research
```python
# Pre-train on large dataset
pretrain_learn = tabular_learner(large_dls)
pretrain_learn.fit(50)

# Fine-tune on target dataset
target_learn = tabular_learner(target_dls)
target_learn.model.load_state_dict(pretrain_learn.model.state_dict())
target_learn.fit(10, lr=1e-4)  # Lower learning rate
```

*FastaiMLP provides an accessible, high-level interface to tabular deep learning with the proven methodologies and ease-of-use that fast.ai is known for.* 