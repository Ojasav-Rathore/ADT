"""
Specific Defense Implementations: AIBD, ABL, CBD, DBD, NAD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional
from tqdm import tqdm
import copy
from .base_defense import BaseDefense


class AIBDDefense(BaseDefense):
    """
    Adversarial-Inspired Backdoor Defense (AIBD)
    Uses adversarial examples to isolate backdoor samples
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda:0',
                 adv_eps: float = 1.0, adv_alpha: float = 1/255, adv_steps: int = 255):
        super().__init__(model, device)
        self.adv_eps = adv_eps / 255  # Normalize epsilon
        self.adv_alpha = adv_alpha
        self.adv_steps = adv_steps
    
    def generate_adversarial_examples(self, images: torch.Tensor, 
                                      labels: torch.Tensor) -> torch.Tensor:
        """Generate adversarial examples using PGD attack"""
        images_adv = images.detach().clone()
        
        for _ in range(self.adv_steps):
            images_adv.requires_grad = True
            
            outputs = self.model(images_adv)
            loss = F.cross_entropy(outputs, labels)
            
            grad = torch.autograd.grad(loss, images_adv)[0]
            
            images_adv = images_adv.detach() + self.adv_alpha * grad.sign()
            images_adv = torch.clamp(images_adv, images - self.adv_eps, images + self.adv_eps)
            images_adv = torch.clamp(images_adv, 0, 1)
        
        return images_adv.detach()
    
    def isolate_backdoor_samples(self, loader, threshold: float = 0.8) -> list:
        """Identify potential backdoor samples"""
        self.model.eval()
        suspicious_indices = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Generate adversarial examples
                images_adv = self.generate_adversarial_examples(images, labels)
                
                # Check if model's predictions change significantly
                outputs_clean = self.model(images)
                outputs_adv = self.model(images_adv)
                
                probs_clean = F.softmax(outputs_clean, dim=1)
                probs_adv = F.softmax(outputs_adv, dim=1)
                
                # Measure difference
                diff = torch.abs(probs_clean - probs_adv).max(dim=1)[0]
                
                # Suspicious if difference is large (indicates backdoor)
                for i, d in enumerate(diff):
                    if d.item() > threshold:
                        suspicious_indices.append(batch_idx * loader.batch_size + i)
        
        return suspicious_indices
    
    def defend(self, train_loader, val_loader, epochs: int = 100, 
              isolation_epochs: int = 20, lr: float = 0.1) -> None:
        """AIBD defense procedure"""
        print("Starting AIBD defense...")
        
        # Stage 1: Isolate backdoor samples
        print("Stage 1: Isolating backdoor samples...")
        suspicious_indices = self.isolate_backdoor_samples(train_loader, threshold=0.5)
        print(f"Found {len(suspicious_indices)} suspicious samples")
        
        # Stage 2: Train on clean samples
        print("Stage 2: Training on isolated clean samples...")
        self.train_model(train_loader, val_loader, isolation_epochs, lr=lr, verbose=True)


class ABLDefense(BaseDefense):
    """
    Anti-Backdoor Learning (ABL)
    Uses activation clustering and unlearning
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda:0',
                 isolation_ratio: float = 0.01, flooding: float = 0.5):
        super().__init__(model, device)
        self.isolation_ratio = isolation_ratio
        self.flooding = flooding
    
    def clustering_isolation(self, train_loader) -> Tuple[list, list]:
        """Isolate samples using activation clustering"""
        print("Performing activation clustering isolation...")
        
        self.model.eval()
        activations = []
        indices = []
        
        # Extract activations
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                
                # Extract features from penultimate layer
                features = self._extract_features(images)
                activations.extend(features.cpu().numpy())
                
                for i in range(len(images)):
                    indices.append((batch_idx, i))
        
        activations = np.array(activations)
        
        # Perform simple clustering (K-means)
        from sklearn.cluster import KMeans
        n_clusters = 2
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(activations)
        
        # Select cluster with anomalous behavior
        cluster_sizes = np.bincount(clusters)
        small_cluster = np.argmin(cluster_sizes)
        
        isolated_indices = np.where(clusters == small_cluster)[0]
        isolated_indices = isolated_indices[:int(len(activations) * self.isolation_ratio)]
        
        return list(isolated_indices), list(activations[isolated_indices])
    
    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from penultimate layer"""
        # Forward pass through model
        x = images
        
        # For simplicity, use global average pooling output
        for layer in list(self.model.children())[:-1]:
            if isinstance(layer, nn.Sequential):
                x = layer(x)
            else:
                x = layer(x)
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        
        return x
    
    def unlearning_phase(self, train_loader, isolated_indices: list, 
                        epochs: int = 5, lr: float = 0.01) -> None:
        """Unlearning phase - gradient ascent on isolated samples"""
        print(f"Unlearning phase for {epochs} epochs...")
        
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                # Apply flooding for gradient ascent on suspicious samples
                loss = torch.abs(loss - self.flooding)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
    
    def defend(self, train_loader, val_loader, epochs: int = 100) -> None:
        """ABL defense procedure"""
        print("Starting ABL defense...")
        
        # Stage 1: Isolation
        isolated_indices, isolated_features = self.clustering_isolation(train_loader)
        print(f"Isolated {len(isolated_indices)} samples")
        
        # Stage 2: Unlearning
        self.unlearning_phase(train_loader, isolated_indices, epochs=5)
        
        # Stage 3: Fine-tuning
        print("Fine-tuning phase...")
        self.train_model(train_loader, val_loader, epochs=20, lr=0.01, verbose=True)


class CBDDefense(BaseDefense):
    """
    Causality-inspired Backdoor Defense (CBD)
    Uses causal inference to remove backdoor effects
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda:0',
                 lambda_param: float = 1.0):
        super().__init__(model, device)
        self.lambda_param = lambda_param
        # Create a second model to capture confounding effects
        self.confounding_model = copy.deepcopy(model).to(device)
    
    def defend(self, train_loader, val_loader, epochs: int = 100, 
              lr: float = 0.1) -> None:
        """CBD defense procedure"""
        print("Starting CBD defense...")
        
        optimizer_main = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        optimizer_conf = optim.SGD(self.confounding_model.parameters(), lr=lr, momentum=0.9)
        
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            self.model.train()
            self.confounding_model.train()
            
            for images, labels in tqdm(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Train confounding model
                optimizer_conf.zero_grad()
                conf_outputs = self.confounding_model(images)
                conf_loss = criterion(conf_outputs, labels)
                conf_loss.backward()
                optimizer_conf.step()
                
                # Train main model with mutual information minimization
                optimizer_main.zero_grad()
                main_outputs = self.model(images)
                main_loss = criterion(main_outputs, labels)
                
                # Add mutual information term
                mi_term = self._estimate_mutual_information(
                    self.model(images), 
                    self.confounding_model(images)
                )
                
                total_loss = main_loss + self.lambda_param * mi_term
                total_loss.backward()
                optimizer_main.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss.item():.4f}")
    
    def _estimate_mutual_information(self, outputs1: torch.Tensor, 
                                    outputs2: torch.Tensor) -> torch.Tensor:
        """Estimate mutual information between two output distributions"""
        p1 = F.softmax(outputs1, dim=1)
        p2 = F.softmax(outputs2, dim=1)
        
        # KL divergence-based MI estimation
        mi = F.kl_div(p1.log(), p2, reduction='mean')
        
        return mi


class DBDDefense(BaseDefense):
    """
    Backdoor Defense via Decoupling the Training Process (DBD)
    Separates training into supervised and self-supervised phases
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda:0'):
        super().__init__(model, device)
    
    def defend(self, train_loader, val_loader, epochs: int = 100,
              supervised_epochs: int = 50, self_supervised_epochs: int = 50,
              lr: float = 0.1) -> None:
        """DBD defense procedure"""
        print("Starting DBD defense...")
        
        # Phase 1: Supervised training
        print("Phase 1: Supervised learning...")
        self.train_model(train_loader, val_loader, supervised_epochs, lr=lr, verbose=True)
        
        # Phase 2: Self-supervised learning (using contrastive loss)
        print("Phase 2: Self-supervised learning...")
        self._self_supervised_train(train_loader, val_loader, self_supervised_epochs, lr=lr)
    
    def _self_supervised_train(self, train_loader, val_loader, epochs: int, 
                              lr: float) -> None:
        """Self-supervised training phase using contrastive learning"""
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for images, labels in tqdm(train_loader):
                images = images.to(self.device)
                
                # Create two augmented views
                images1 = self._augment(images)
                images2 = self._augment(images)
                
                optimizer.zero_grad()
                
                features1 = self._extract_features(images1)
                features2 = self._extract_features(images2)
                
                # Contrastive loss
                loss = self._contrastive_loss(features1, features2)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
    
    def _augment(self, images: torch.Tensor) -> torch.Tensor:
        """Simple data augmentation"""
        return images + torch.randn_like(images) * 0.1
    
    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from penultimate layer"""
        x = images
        for layer in list(self.model.children())[:-1]:
            if isinstance(layer, nn.Sequential):
                x = layer(x)
            else:
                x = layer(x)
        
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x
    
    def _contrastive_loss(self, z_i: torch.Tensor, z_j: torch.Tensor,
                         temperature: float = 0.5) -> torch.Tensor:
        """NT-Xent loss for contrastive learning"""
        batch_size = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)
        
        # Normalize
        z = F.normalize(z, dim=1)
        
        # Similarity matrix
        similarity_matrix = torch.mm(z, z.t())
        
        # Create mask for positive pairs
        mask = torch.eye(batch_size, dtype=torch.bool, device=self.device)
        mask = torch.cat([
            torch.cat([torch.zeros_like(mask), mask], dim=1),
            torch.cat([mask, torch.zeros_like(mask)], dim=1)
        ], dim=0)
        
        # Compute loss
        pos = torch.exp(similarity_matrix / temperature)
        neg = torch.exp(similarity_matrix / temperature)
        
        loss = -torch.log(pos.sum(dim=1) / neg.sum(dim=1))
        return loss.mean()


class NADDefense(BaseDefense):
    """
    Neural Attention Distillation (NAD)
    Uses attention-based knowledge distillation from clean teacher model
    """
    
    def __init__(self, model: nn.Module, teacher_model: nn.Module = None,
                 device: str = 'cuda:0', temperature: float = 4.0):
        super().__init__(model, device)
        self.teacher_model = teacher_model
        self.temperature = temperature
        
        if teacher_model is not None:
            self.teacher_model = teacher_model.to(device)
    
    def defend(self, train_loader, val_loader, clean_loader, epochs: int = 20,
              lr: float = 0.1) -> None:
        """NAD defense procedure"""
        print("Starting NAD defense...")
        
        # If no teacher provided, use standard training
        if self.teacher_model is None:
            print("No teacher model provided, using standard training")
            self.train_model(train_loader, val_loader, epochs, lr=lr, verbose=True)
            return
        
        self.teacher_model.eval()
        
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for images, labels in tqdm(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Student model
                student_outputs = self.model(images)
                student_loss = criterion(student_outputs, labels)
                
                # Teacher model
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(images)
                
                # Knowledge distillation loss
                kd_loss = self._kl_divergence_loss(student_outputs, teacher_outputs)
                
                total_loss_val = student_loss + kd_loss
                
                total_loss_val.backward()
                optimizer.step()
                
                total_loss += total_loss_val.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
    
    def _kl_divergence_loss(self, student_outputs: torch.Tensor,
                           teacher_outputs: torch.Tensor) -> torch.Tensor:
        """KL divergence for knowledge distillation"""
        p = F.log_softmax(student_outputs / self.temperature, dim=1)
        q = F.softmax(teacher_outputs / self.temperature, dim=1)
        
        kl_loss = F.kl_div(p, q, reduction='mean')
        return kl_loss


def get_defense_method(defense_name: str, model: nn.Module,
                       teacher_model: nn.Module = None,
                       device: str = 'cuda:0') -> BaseDefense:
    """Get defense instance by name"""
    defenses = {
        'aibd': AIBDDefense,
        'abl': ABLDefense,
        'cbd': CBDDefense,
        'dbd': DBDDefense,
        'nad': NADDefense
    }
    
    defense_class = defenses.get(defense_name.lower())
    if defense_class is None:
        raise ValueError(f"Unknown defense: {defense_name}")
    
    if defense_name.lower() == 'nad' and teacher_model is not None:
        return defense_class(model, teacher_model, device)
    elif defense_name.lower() == 'nad':
        return defense_class(model, None, device)
    else:
        return defense_class(model, device)
