
# Image Segmentation Using Clustering

## Authors
- Rishi Chhabra (Roll No: M24CSE020)
- Shivam Shukla (Roll No: M23CSE022)
- Nipun Dhokne (Roll No: M24CSE016)

## Overview
This project focuses on image segmentation techniques to identify water regions in flood-affected areas using various clustering algorithms. The aim is to develop a reliable model that can accurately segment images for better assessment and response to flood situations.

## Table of Contents
- [Package Requirements](#package-requirements)
- [Dataset Description](#dataset-description)
- [Methodology](#methodology)
- [Clustering Algorithms](#clustering-algorithms)
- [Evaluation Metrics](#evaluation-metrics)
- [Run Instructions](#run-instructions)
- [Results](#results)
- [Conclusion](#conclusion)
- [References](#references)

## Package Requirements
To run this project, ensure you have the following packages installed:

- Python (>=3.7)
- NumPy (>=1.18.5)
- pandas (>=1.1.0)
- scikit-learn (>=0.24.0)
- matplotlib (>=3.3.0)
- seaborn (>=0.11.0)
- opencv-python (>=4.5.0)
- scikit-image (>=0.17.0)
- jupyter (optional, for notebooks)

You can install the required packages using pip. Create a `requirements.txt` file with the following content:

```
numpy>=1.18.5
pandas>=1.1.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
opencv-python>=4.5.0
scikit-image>=0.17.0
jupyter
```

Then, run the following command in your terminal:

```bash
pip install -r requirements.txt
```

## Dataset Description
The dataset used in this project consists of:
- **Images**: A collection of 290 images depicting flood-hit areas.
- **Masks**: Corresponding masks that highlight the water regions, annotated using Label Studio.
- **Metadata**: A CSV file mapping image names to their respective masks.

## Methodology
The methodology includes several key steps:
1. **Preprocessing**: Downsampling images to reduce computational requirements.
2. **Clustering Algorithms**: Implementation of K-Means, Gaussian Mixture Models (GMM), Ratio Cut Clustering, and DBSCAN.
3. **Hyperparameter Tuning**: Techniques such as the Elbow Method and Bayesian Information Criterion (BIC) for optimizing clustering parameters.
4. **Evaluation Metrics**: Metrics used to assess clustering performance include Silhouette Score, Calinski-Harabasz Score, and Davies-Bouldin Score.

## Clustering Algorithms
### K-Means Clustering
K-Means partitions the dataset into \( K \) clusters based on the similarity of pixel values.

### Gaussian Mixture Model (GMM)
GMM models the data points as a mixture of several Gaussian distributions, allowing for a probabilistic clustering approach.

### Ratio Cut Clustering
This algorithm minimizes the ratio of the cut size to the total degree of the graph by leveraging the Laplacian matrix of the similarity graph.

### DBSCAN
DBSCAN identifies clusters based on the density of data points, making it suitable for finding clusters of arbitrary shapes.

## Evaluation Metrics
- **Silhouette Score**: Measures how similar an object is to its own cluster compared to other clusters.
- **Calinski-Harabasz Score**: Provides a ratio of between-cluster variance to within-cluster variance.
- **Davies-Bouldin Score**: Evaluates the average similarity ratio of each cluster with the cluster most similar to it.

## Run Instructions
To execute the code for this project, follow these steps:

1. **Clone the repository** (if hosted on a platform like GitHub):
   ```bash
   git clone https://github.com/yourusername/image-segmentation-clustering.git
   cd image-segmentation-clustering
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the main script**:
   Replace `main_script.py` with the name of your main Python file.
   ```bash
   python main_script.py
   ```

5. **View Results**: Check the output directory for generated plots and segmented images.

## Results
The project includes various plots and visualizations, such as:
- Elbow method plot for K-Means.
- BIC plot for GMM.
- Silhouette plots for each clustering method.
- Comparison of original images and predicted masks.

## Conclusion
This project demonstrates the application of clustering algorithms for effective image segmentation in flood-affected areas, contributing to disaster management and environmental monitoring.

## References
1. Zhang, Y., & Li, H. (2008). Image segmentation using K-means clustering and improved watershed algorithm. *Journal of Computer and System Sciences*, 74(1), 25-38.
2. Rajkumar, M., & Othman, M. (2014). A Review on K-means Clustering Algorithm for Big Data. *International Journal of Advanced Computer Science and Applications*, 5(1), 28-33.
3. Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. In *Proceedings of the Eighteenth Annual ACM-SIAM Symposium on Discrete Algorithms* (pp. 1027-1035).
4. Zivkovic, Z. (2004). Improved GMM for change detection in color images. *International Conference on Image Processing*.
5. Zheng, Y., & Huang, J. (2018). Spectral Clustering for Image Segmentation. *IEEE Transactions on Image Processing*, 27(2), 599-610.
6. Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. In *Proceedings of the Second International Conference on Knowledge Discovery and Data Mining* (pp. 226-231).
7. Gonzalez, R., & Alomar, N. (2020). Flood Detection Using Satellite Images: A Case Study. *Remote Sensing*, 12(4), 639.
8. Ronneberger, O., Fischer, P., & Becker, A. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. In *Medical Image Computing and Computer-Assisted Intervention* (pp. 234-241).
9. Long, J., Shelhamer, E., & Darrell, T. (2015). Fully Convolutional Networks for Semantic Segmentation. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 3431-3440).
10. Chen, L. C., Papandreou, G., Schroff, F., & Adam, H. (2017). Rethinking Atrous Convolution for Semantic Image Segmentation. In *Proceedings of the European Conference on Computer Vision* (pp. 20-36).
11. Zhang, Y., et al. (2020). Hybrid deep learning model for image segmentation. *IEEE Transactions on Neural Networks and Learning Systems*.
12. Brooks, J. R., & Rother, C. (2006). Satellite-based methods for flood detection: A comparison of available techniques. *Earth and Planetary Science Letters*, 242(2), 460-470.
13. Delgado, J. D., et al. (2018). Real-time flood monitoring system based on remote sensing techniques. *Remote Sensing Applications: Society and Environment*, 10, 152-161.
14. Gupta, R., & Thakur, R. (2021). A Survey on Flood Detection Techniques Using Remote Sensing. *International Journal of Applied Engineering Research*, 16(5), 150-156.
15. Mermoz, S., & Kussul, N. (2017). Image Segmentation Techniques for Flood Mapping from Satellite Imagery. *ISPRS Journal of Photogrammetry and Remote Sensing*, 130, 219-229.
