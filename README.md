## DNN: Comparative Study on CIFAR-10

This notebook conducts a comparative study of Deep Neural Networks by performing a grid search over various hyperparameters.
The experiments use the CIFAR-10 dataset, consisting of 60,000 color images in 10 classes.

### 1. Exploratory Data Analysis (EDA)

- Visualized random samples from the dataset to verify data integrity.
- ![fig:Dataset Head](report/figures/plot_Dataset%20Head.png)
- Analyzed the distribution of samples across the 10 classes.
- ![fig:Class Distribution](report/figures/plot_Class%20Distribution.png)
- Examined color value distributions (Red, Green, Blue channels).
- ![fig:Color Value Histogram](report/figures/plot_Color%20Value%20Histogram.png)
- Plotted histograms for feature means and standard deviations.
- ![fig:Data Mean Histogram](report/figures/plot_Data%20Mean%20Histogram.png)
- ![fig:Data STD Histogram](report/figures/plot_Data%20STD%20Histogram.png)
- Verified absence of missing values and duplicate samples.
- Checked for outliers including invalid pixels, blank images, and over/under-exposed images.

### 2. Data Preparation

- Reshaped image data to channel-first format (Channels, Height, Width) for PyTorch.
- Scaled pixel values to the range [0, 1] by dividing by 255.
- Split training data into training (45,000 samples) and validation (5,000 samples) sets using stratified sampling.
- Converted data to PyTorch Tensors and wrapped them in TensorDatasets and DataLoaders.

### 3. Models Evaluated

- Custom Convolutional Neural Network (CNN) architectures defined by variable depths.
- **Convolutional Layers:** Grid search over counts of [2, 3, 4, 5].
- **Fully Connected Layers:** Grid search over counts of [2, 3, 4, 5].
- **Activation Functions:** Comparison of ReLU, Tanh, and Sigmoid.
- **Optimization Strategies:** Comparison of None (SGD), Momentum, AdaptiveLR (RMSprop), and Both (Adam).
- **Batch Size:** Analyzed impact of sizes [1, 4, 16, 64, 256, 1024, 4096].

### 4. Model Evaluation

- Evaluated models based on Train Loss, Train Accuracy, Validation Loss, Validation Accuracy, and Total Training Time.
- Generated training reports showing loss and accuracy curves for specific runs.
- For example, for 3 FC Layers, 5 Conv Layers, ReLU Activation, and Adam Optimization:
  ![fig:Training Report](report/figures/report%203fc%205cv%20relu%20adam.png)
- Analyzed the impact of batch size on training time and stability.
- ![fig:Total Time per Epoch](report/figures/comp%20batch%20size.png)
- Visualized performance variations across different hyperparameter combinations (FC Count, Conv Count, Activation, Optimization).
- For example, for 5 FC Layers, 5 Conv Layers, Sigmoid Activation, and Momentum Optimization:
  ![fig:Variation over different Hyperparameters](report/figures/comp%20against%205fc%205cv%20sigm%20mom.png)
- Generated heatmaps to analyze the relationship between network depth (FC vs. Conv counts) and Validation Loss.
- ![fig:Depth to Val Loss](report/figures/comp%20fc%20vs%20cv.png) (Single Heatmap)
- ![fig:Depth to Val Loss](report/figures/comp%20all%20hyp.png) (Grid of Heatmaps by Activation/Optimizer)

### 5. Final Results

- Analyzed convergence speeds for different configurations.
- ![fig:ReLU + AdaptiveLR](report/figures/top%20perf%20all.png)
- Analyzed convergence speeds across relu and adaptive LR (fastest ones from last plot).
- ![fig:ReLU + AdaptiveLR](report/figures/top%20perf%20relu+alr.png)
- Compared mean convergence speeds across all activation and optimization pairs.
- ![fig:Mean Convergence Speed](report/figures/res%20mean%20convergence%20speed.png)
- Identified the recommended architecture: **2 Convolutional Layers, 4 Fully Connected Layers, ReLU Activation, and 'Both' Optimization**.
- ![fig:Training Report](report/figures/plot_%7Bconv_count=2,fc_count=4,activation=ReLU,batch_size=1024,optim_strategy=Both,learning_rate=0.001,epochs=25,TotalTime=124.39sec%7D.png) (Final Recommended Architecture)
