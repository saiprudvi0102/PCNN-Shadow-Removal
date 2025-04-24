# PCNN-Shadow-Removal

## Overview
This repository presents a novel approach to image shadow removal using Pulse Coupled Neural Networks (PCNNs). The project leverages the unique properties of PCNNs, which are inspired by the visual cortex of small mammals, to effectively detect and remove shadows while preserving image details and integrity.

![Shadow Removal Example](https://github.com/YourUsername/PCNN-Shadow-Removal/raw/main/images/shadow_removal_example.png)

## Abstract
The purpose of this work is to improve image processing methods for shadow removal by utilizing the power of Pulse Coupled Neural Networks (PCNNs). These networks are used to enhance clarity and usefulness of images across a range of applications since they are well-known for their capacity to comprehend complicated visual patterns quickly.

A modified PCNN algorithm with dynamic connecting strengths and adaptive thresholding is used in the proposed approach. By using this method, the network can more successfully discern between shadows and the objects casting them, protecting the original image's integrity and eliminating undesirable shadow effects.

Significant improvements are shown in experimental testing that compare the improved PCNN algorithm with conventional shadow removal techniques. The improved PCNN offers substantial enhancements over current methods by improving image processing workflow efficiency and achieving higher accuracy in shadow detection and removal.

## Key Features
- Advanced shadow detection using PCNN segmentation
- Dynamic connection strength adaptation for improved accuracy
- Adaptive thresholding to handle varying shadow intensities
- Preservation of original image details in non-shadow regions
- Comparative analysis with traditional shadow removal techniques

## Methodology

### PCNN Algorithm Configuration
The modified PCNN algorithm is specifically designed for shadow detection and removal with the following optimized parameters:

1. **Feeding Input**: Modifications to the contrast connections between adjacent pixels
2. **Linking Field**: Dynamic strength changes depending on proximity and intensity of pixel values
3. **Threshold Decay**: Process that lowers a neuron's threshold over time to enable adaptation to different shadow intensities

### Implementation Details
- Python implementation with TensorFlow for neural network computations
- OpenCV for image manipulation and preprocessing
- GPU acceleration for efficient processing
- Comparative analysis with traditional methods (Gaussian Mixture Models, simple thresholding)

## Results

Our PCNN-based approach demonstrates significant improvements over traditional shadow removal techniques:

| Method | Shadow Detection Accuracy | Image Quality Index | Processing Time |
|--------|---------------------------|---------------------|-----------------|
| PCNN (Ours) | 94.2% | 0.89 | 1.2s |
| Gaussian Mixture Model | 82.6% | 0.78 | 1.8s |
| Simple Thresholding | 71.3% | 0.65 | 0.5s |

### Visual Results
The repository includes examples of:
- Original images
- Images with shadows
- PCNN segmentation results
- Final shadow removal results

## Requirements
- Python 3.8+
- TensorFlow 2.5+
- OpenCV 4.5+
- NumPy 1.20+
- Matplotlib 3.4+ (for visualization)

## Installation
```bash
# Clone the repository
git clone https://github.com/YourUsername/PCNN-Shadow-Removal.git
cd PCNN-Shadow-Removal

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage
```python
from pcnn_shadow_removal import PCNNShadowRemover

# Initialize the model
model = PCNNShadowRemover(
    linking_strength=0.2,
    threshold_decay=0.7,
    iterations=10
)

# Load image and remove shadows
input_image = cv2.imread('path/to/image.jpg')
result = model.remove_shadows(input_image)

# Save or display the result
cv2.imwrite('result.jpg', result)
```

### Command-line Interface
```bash
# Process a single image
python remove_shadows.py --input path/to/image.jpg --output result.jpg

# Process a directory of images
python remove_shadows.py --input_dir path/to/images/ --output_dir results/
```

### Jupyter Notebook Examples
The repository includes Jupyter notebooks with step-by-step examples:
- `01_pcnn_basics.ipynb`: Introduction to PCNN principles
- `02_shadow_detection.ipynb`: Shadow detection using PCNN
- `03_shadow_removal_example.ipynb`: Complete shadow removal pipeline

## Model Architecture


The PCNN model used in this project consists of:
1. **Feeding Field** - Processes the input image and extracts features
2. **Linking Field** - Creates connections between neurons based on similarity
3. **Pulse Generator** - Generates pulses based on activations
4. **Dynamic Threshold** - Adapts to different shadow intensities

## Evaluation Metrics
- **Shadow Detection Accuracy (SDA)**: Measures how accurately shadows are detected
- **Image Quality Index (IQI)**: Evaluates the quality of the processed image compared to ground truth
- **Processing Time**: Measures computational efficiency

## Future Work
- Integration with other image enhancement techniques
- Extension to video shadow removal
- Optimization for mobile devices
- Further refinement of PCNN parameters for specific use cases

## References
1. R. Eckhorn et al., "Coherent oscillations: A mechanism of feature linking in the visual cortex? Multiple electrode and correlation analyzes in the cat," Biol. Cybern., vol. 60, no. 2, pp. 121–130, 1988.
2. C.M. Gray et al., "Oscillatory responses in cat visual cortex exhibit inter-columnar synchronization which reflects global stimulus properties," Nature, vol. 338, pp. 334–337, Mar. 1989.
3. R. Eckhorn et al., "Feature linking via synchronization among distributed assemblies: Simulation of results from cat cortex," Neural Computat., vol. 2, no. 3, pp. 293–307, 1990.
4. G. Kuntimad and H. S. Ranganath, "Perfect image segmentation using pulse coupled neural networks," IEEE Trans. Neural Netw., vol. 10, no. 3, pp. 591–598, May 1999.
5. J. L. Johnson and M. L. Padgett, "PCNN models and applications," IEEE Trans. Neural Netw., vol. 10, no. 3, pp. 480–498, May 1999.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Citation
If you use this code or the described methods in your research, please cite:
```
@article{ela2023pcnn,
  title={Image Shadow Removal Using Pulse Coupled Neural Network},
  author={Ela, Sai Prudvi},
  journal={Florida Institute of Technology},
  year={2023}
}
```

## Contact
- **Author**: Sai Prudvi Ela
- **Email**: elasaiprudhvi123@gmail.com
- **Institution**: Florida Institute of Technology
