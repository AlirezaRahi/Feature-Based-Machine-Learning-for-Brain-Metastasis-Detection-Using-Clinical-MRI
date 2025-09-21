import os
import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import random
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedRandomForestClassifier

# Settings
np.random.seed(42)
random.seed(42)

class BrainMetsDataset:
    def __init__(self, base_dir, max_patients=None, target_shape=(128, 128, 64)):
        self.base_dir = base_dir
        self.target_shape = target_shape
        self.patient_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) 
                           if os.path.isdir(os.path.join(base_dir, d))]
        self.patient_dirs.sort()
        
        if max_patients:
            self.patient_dirs = self.patient_dirs[:max_patients]
        
        print(f"Found {len(self.patient_dirs)} patient directories.")
        self.data_cache = {}

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        if idx in self.data_cache:
            return self.data_cache[idx]
            
        patient_path = self.patient_dirs[idx]
        
        try:
            files = os.listdir(patient_path)
            
            t1_pre_files = [f for f in files if 'T1pre' in f or 'T1_pre' in f]
            t1_post_files = [f for f in files if 'T1post' in f or 'T1_post' in f]
            seg_files = [f for f in files if 'seg' in f.lower() or 'mask' in f.lower()]
            
            if not t1_pre_files or not t1_post_files or not seg_files:
                return None
            
            t1_pre_path = os.path.join(patient_path, t1_pre_files[0])
            t1_post_path = os.path.join(patient_path, t1_post_files[0])
            seg_path = os.path.join(patient_path, seg_files[0])

            t1_pre_img = nib.load(t1_pre_path).get_fdata()
            t1_post_img = nib.load(t1_post_path).get_fdata()
            seg_img = nib.load(seg_path).get_fdata()

            # Crop or resize
            t1_pre_img = self.crop_or_pad(t1_pre_img)
            t1_post_img = self.crop_or_pad(t1_post_img)
            seg_img = self.crop_or_pad(seg_img)

            # Normalize
            t1_pre_img = self.normalize_image(t1_pre_img)
            t1_post_img = self.normalize_image(t1_post_img)

            input_image = np.stack([t1_pre_img, t1_post_img], axis=-1)
            segmentation = (seg_img > 0).astype(np.float32)

            self.data_cache[idx] = (input_image, segmentation)
            return input_image, segmentation
            
        except Exception as e:
            print(f"Error loading patient {idx}: {e}")
            return None

    def crop_or_pad(self, image):
        if len(image.shape) > 3:
            image = image[..., 0]
        
        current_shape = image.shape
        for i in range(3):
            current_dim = current_shape[i]
            target_dim = self.target_shape[i]
            
            if current_dim >= target_dim:
                start = (current_dim - target_dim) // 2
                end = start + target_dim
                if i == 0:
                    image = image[start:end, :, :]
                elif i == 1:
                    image = image[:, start:end, :]
                else:
                    image = image[:, :, start:end]
            else:
                pad_before = (target_dim - current_dim) // 2
                pad_after = target_dim - current_dim - pad_before
                if i == 0:
                    image = np.pad(image, ((pad_before, pad_after), (0, 0), (0, 0)), mode='constant')
                elif i == 1:
                    image = np.pad(image, ((0, 0), (pad_before, pad_after), (0, 0)), mode='constant')
                else:
                    image = np.pad(image, ((0, 0), (0, 0), (pad_before, pad_after)), mode='constant')
        
        if image.shape != self.target_shape:
            factors = [self.target_shape[i] / image.shape[i] for i in range(3)]
            image = zoom(image, factors, order=1)
        
        return image

    def normalize_image(self, image):
        image = np.clip(image, np.percentile(image, 1), np.percentile(image, 99))
        return (image - np.mean(image)) / (np.std(image) + 1e-8)

    def get_all_data(self):
        all_images = []
        all_masks = []
        
        for i in range(len(self)):
            result = self[i]
            if result is not None:
                image, mask = result
                all_images.append(image)
                all_masks.append(mask)
            
        return np.array(all_images), np.array(all_masks)

def extract_advanced_features(volumes):
    """Extract advanced and meaningful features"""
    features_list = []
    
    for volume in volumes:
        t1_pre = volume[..., 0]
        t1_post = volume[..., 1]
        
        features = []
        
        # 1. Intensity-based features
        for channel, name in zip([t1_pre, t1_post], ['T1pre', 'T1post']):
            features.extend([
                np.mean(channel), np.std(channel), np.median(channel),
                np.min(channel), np.max(channel), 
                np.percentile(channel, 25), np.percentile(channel, 75),
                np.percentile(channel, 90), np.percentile(channel, 95),
                np.mean(channel > np.mean(channel))  # Ratio of pixels above mean
            ])
        
        # 2. Enhancement features
        enhancement = t1_post - t1_pre
        features.extend([
            np.mean(enhancement), np.std(enhancement), np.max(enhancement),
            np.mean(enhancement > 0),  # Ratio of positive enhancement pixels
            np.mean(enhancement > np.mean(enhancement))  # Strong enhancement
        ])
        
        # 3. Texture features (gradients)
        gradients = []
        for dim in range(3):
            grad = np.gradient(t1_post)[dim]
            gradients.extend([np.mean(np.abs(grad)), np.std(grad), np.max(np.abs(grad))])
        features.extend(gradients)
        
        # 4. Histogram features
        hist_post, _ = np.histogram(t1_post, bins=20, density=True)
        features.extend(hist_post[:10])  # First 10 bins
        
        features_list.append(features)
    
    return np.array(features_list)

def manual_data_augmentation(X_features, y_labels, augmentation_factor=5):
    """Manual augmentation of minority class data"""
    minority_indices = np.where(y_labels == 0)[0]
    majority_indices = np.where(y_labels == 1)[0]
    
    if len(minority_indices) == 0:
        return X_features, y_labels
    
    X_minority = X_features[minority_indices]
    y_minority = y_labels[minority_indices]
    
    X_augmented = [X_minority]
    y_augmented = [y_minority]
    
    # Create synthetic samples by adding noise
    for i in range(augmentation_factor):
        noise = np.random.normal(0, 0.1, X_minority.shape)
        X_augmented.append(X_minority + noise)
        y_augmented.append(y_minority)
    
    X_minority_aug = np.vstack(X_augmented)
    y_minority_aug = np.concatenate(y_augmented)
    
    # Combine with original data
    X_balanced = np.vstack([X_features[majority_indices], X_minority_aug])
    y_balanced = np.concatenate([y_labels[majority_indices], y_minority_aug])
    
    return X_balanced, y_balanced

def train_with_cross_validation():
    """Training with Cross-Validation and advanced methods"""
    
    base_directory = r"C:\Alex The Great\Project\medai-env\datasets\Brain\UCSF_BrainMetastases_TRAIN"
    
    print("Loading dataset...")
    dataset = BrainMetsDataset(base_dir=base_directory, max_patients=50, target_shape=(128, 128, 64))
    
    all_images, all_masks = dataset.get_all_data()
    print(f"Loaded {len(all_images)} samples")
    
    # Create labels
    y_labels = np.array([1 if np.any(mask > 0) else 0 for mask in all_masks])
    print(f"Class distribution: {dict(zip(['Healthy', 'Metastasis'], np.bincount(y_labels)))}")
    
    # Extract advanced features
    print("Extracting advanced features...")
    X_features = extract_advanced_features(all_images)
    print(f"Extracted {X_features.shape[1]} features per sample")
    
    # Manual augmentation of healthy data
    print("Augmenting minority class...")
    X_balanced, y_balanced = manual_data_augmentation(X_features, y_labels, augmentation_factor=10)
    
    print(f"After augmentation: {dict(zip(['Healthy', 'Metastasis'], np.bincount(y_balanced)))}")
    
    # Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    cv_predictions = []
    
    print("\nStarting 5-Fold Cross Validation...")
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_balanced, y_balanced)):
        X_train, X_test = X_balanced[train_idx], X_balanced[test_idx]
        y_train, y_test = y_balanced[train_idx], y_balanced[test_idx]
        
        # Normalize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Different models
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
            'SVM': SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        best_score = 0
        best_model = None
        best_name = ""
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            score = accuracy_score(y_test, y_pred)
            
            if score > best_score:
                best_score = score
                best_model = model
                best_name = name
        
        # Save results
        y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
        cv_scores.append(best_score)
        cv_predictions.append((y_test, y_pred_proba))
        
        print(f"Fold {fold+1}: {best_name} - Accuracy: {best_score:.4f}")
    
    # Average results
    mean_accuracy = np.mean(cv_scores)
    print(f"\nMean Cross-Validation Accuracy: {mean_accuracy:.4f}")
    
    # Collect all predictions for overall evaluation
    all_y_test = np.concatenate([pred[0] for pred in cv_predictions])
    all_y_pred_proba = np.concatenate([pred[1] for pred in cv_predictions])
    all_y_pred = (all_y_pred_proba > 0.5).astype(int)
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION RESULTS")
    print("="*60)
    
    accuracy = accuracy_score(all_y_test, all_y_pred)
    print(f"Overall Accuracy: {accuracy:.4f}")
    
    if len(np.unique(all_y_test)) > 1:
        roc_auc = roc_auc_score(all_y_test, all_y_pred_proba)
        print(f"ROC AUC: {roc_auc:.4f}")
        
        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(all_y_test, all_y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Cross Validation')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig('roc_curve_cv.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Confusion Matrix
    cm = confusion_matrix(all_y_test, all_y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Metastasis', 'Metastasis'],
                yticklabels=['No Metastasis', 'Metastasis'])
    plt.title('Confusion Matrix - Cross Validation')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_cv.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(all_y_test, all_y_pred, 
                              target_names=['No Metastasis', 'Metastasis'], 
                              zero_division=0))
    
    # Feature importance analysis
    final_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    final_model.fit(StandardScaler().fit_transform(X_balanced), y_balanced)
    
    feature_importance = final_model.feature_importances_
    top_features = np.argsort(feature_importance)[-15:]
    
    print("\nTop 15 Important Features:")
    for i, idx in enumerate(top_features[::-1]):
        print(f"{i+1}. Feature {idx}: {feature_importance[idx]:.4f}")
    
    # Save final model
    joblib.dump(final_model, 'final_model.pkl')
    print("\nFinal model saved as 'final_model.pkl'")
    
    return final_model, (all_y_test, all_y_pred, all_y_pred_proba)

if __name__ == "__main__":
    try:
        print("Starting Advanced Training with Cross-Validation...")
        model, results = train_with_cross_validation()
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()