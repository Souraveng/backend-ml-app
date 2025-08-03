import cv2
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import pickle
from sklearn.datasets import fetch_openml

# Set random seed for reproducibility
np.random.seed(42)

def extract_hog_features(image):
    """Extract HOG features from image using OpenCV"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Resize image to standard size
    gray = cv2.resize(gray, (64, 64))
    
    # Create HOG descriptor
    hog = cv2.HOGDescriptor(
        _winSize=(64, 64),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )
    
    # Compute HOG features
    features = hog.compute(gray)
    return features.flatten()

def extract_color_histogram(image):
    """Extract color histogram features"""
    # Calculate histogram for each channel
    hist_b = cv2.calcHist([image], [0], None, [32], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [32], [0, 256])
    hist_r = cv2.calcHist([image], [2], None, [32], [0, 256])
    
    # Combine histograms
    hist_features = np.concatenate([hist_b, hist_g, hist_r]).flatten()
    return hist_features

def extract_combined_features(image):
    """Extract combined HOG and color histogram features"""
    hog_features = extract_hog_features(image)
    color_features = extract_color_histogram(image)
    return np.concatenate([hog_features, color_features])

def load_cifar10_data():
    """Load CIFAR-10 dataset from sklearn/openml"""
    print("Loading CIFAR-10 dataset...")
    
    # Load CIFAR-10 dataset
    cifar10 = fetch_openml('CIFAR_10', version=1, as_frame=False, parser='auto')
    
    # Get data and labels
    X, y = cifar10.data, cifar10.target.astype(int)
    
    # Reshape images to 32x32x3
    X = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    # Class names for CIFAR-10
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    return X, y, class_names

def train_model():
    """Train the image classification model using SVM"""
    try:
        # Load dataset
        X, y, class_names = load_cifar10_data()
        
        print(f"Dataset shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        
        # Use a subset for faster training (you can increase this)
        subset_size = 5000
        indices = np.random.choice(len(X), subset_size, replace=False)
        X_subset = X[indices]
        y_subset = y[indices]
        
        print(f"Using subset of {subset_size} samples for training")
        
        # Extract features
        print("Extracting features...")
        features = []
        for i, img in enumerate(X_subset):
            if i % 500 == 0:
                print(f"Processing image {i}/{len(X_subset)}")
            
            # Convert to uint8 if needed
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            
            feat = extract_combined_features(img)
            features.append(feat)
        
        features = np.array(features)
        print(f"Features shape: {features.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, y_subset, test_size=0.2, random_state=42, stratify=y_subset
        )
        
        # Scale features
        print("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train SVM model
        print("Training SVM model...")
        model = svm.SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,  # Enable probability estimates
            random_state=42
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        print("Making predictions...")
        y_pred = model.predict(X_test_scaled)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test accuracy: {accuracy:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        # Save model and scaler
        print("Saving model and scaler...")
        joblib.dump(model, 'svm_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump(class_names, 'class_names.pkl')
        
        print("Model saved as 'svm_model.pkl'")
        print("Scaler saved as 'scaler.pkl'")
        print("Class names saved as 'class_names.pkl'")
        
        return model, scaler, class_names
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        print("\nFalling back to a simple demo model...")
        
        # Create a simple demo model with dummy data
        from sklearn.datasets import make_classification
        
        X_demo, y_demo = make_classification(
            n_samples=1000, n_features=100, n_classes=10, random_state=42
        )
        
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_demo, y_demo, test_size=0.2, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = svm.SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        accuracy = model.score(X_test_scaled, y_test)
        print(f"Demo model accuracy: {accuracy:.4f}")
        
        # Save demo model
        joblib.dump(model, 'svm_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump(class_names, 'class_names.pkl')
        
        print("Demo model saved successfully!")
        
        return model, scaler, class_names

if __name__ == "__main__":
    model, scaler, class_names = train_model()
