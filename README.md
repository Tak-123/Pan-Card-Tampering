# Pan-Card-Tampering
This project focuses on detecting PAN (Permanent Account Number) card tampering using advanced machine learning techniques. The model aims to identify altered or forged PAN card images, helping to prevent identity fraud. The solution involves computer vision methods and deep learning algorithms.
Key Features:
Image Preprocessing: Techniques such as resizing, grayscaling, and noise reduction are applied to prepare images for the model.
Feature Extraction: Uses image processing libraries (like OpenCV) to extract key features from PAN card images, including text, layout, and logo positions.
Tampering Detection: Leverages convolutional neural networks (CNN) or pre-trained models like VGG16/ResNet to classify images as tampered or authentic.
Data Augmentation: Augments the dataset using methods like rotations, flips, and blurring to improve model generalization.
Model Evaluation: Uses accuracy, precision, recall, and F1-score to evaluate the performance of the tampering detection model.
Tech Stack:
Programming Language: Python
Libraries: OpenCV, TensorFlow/Keras, NumPy, Pandas, Matplotlib
Model: Convolutional Neural Network (CNN), or transfer learning models like ResNet/VGG16 for image classification.
Dataset: Custom dataset with authentic and tampered PAN card images (can be synthetically generated or manually collected).
How It Works:
Data Collection: The dataset consists of images of authentic and tampered PAN cards. Tampered images include alterations in text, photos, or layout.
Preprocessing: Images are processed to a consistent size and format, with steps such as noise reduction, grayscaling, and text extraction.
Feature Extraction: Extracts important features from the card, like text integrity, layout consistency, and logo authenticity.
Model Training: A deep learning model (CNN) is trained on labeled data (tampered vs. authentic) using TensorFlow/Keras.
Prediction: The trained model predicts whether a given PAN card image is authentic or tampered based on its learned features.
Dataset:
The dataset can either be created manually by altering authentic PAN card images (e.g., Photoshop tampering) or downloaded from an external source (if available).
If OCR is used, extract text from PAN card images to verify integrity.
